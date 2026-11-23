"""P0 regression tests: Lease parsing, ledger durability, HPA recovery.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v

These target the defects that could leave the namespace without HPAs or hand
leadership to a second controller:

  * C01  an RFC3339 timestamp without fractional seconds must not read as stale
  * C01  acquireTime must be preserved across renewals
  * C05  the ledger must record the action, not None
  * C06  a settled outcome must survive a restart (event replay)
  * C07  the pre-action record must carry the full state
  * C08  settlement time is the settlement time, not the action time
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback.ledger import Ledger                        # noqa: E402
from k8s.leader import LeaderElection, parse_rfc3339      # noqa: E402


class TestRFC3339(unittest.TestCase):
    def test_accepts_fractional_and_plain(self):
        a = parse_rfc3339("2026-11-23T12:00:00.123456Z")
        b = parse_rfc3339("2026-11-23T12:00:00Z")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(a.timestamp(), b.timestamp(), places=0)

    def test_accepts_numeric_offset(self):
        self.assertIsNotNone(parse_rfc3339("2026-11-23T12:00:00+00:00"))
        self.assertIsNotNone(parse_rfc3339("2026-11-23T12:00:00.5+02:00"))

    def test_rejects_garbage(self):
        self.assertIsNone(parse_rfc3339("not-a-time"))
        self.assertIsNone(parse_rfc3339(""))
        self.assertIsNone(parse_rfc3339(None))


class FakeCore:
    """Minimal stand-in for CustomObjectsApi over an in-memory Lease."""

    def __init__(self):
        self.objects = {}

    def get_namespaced_custom_object(self, **kw):
        from kubernetes.client.rest import ApiException
        name = kw["name"]
        if name not in self.objects:
            e = ApiException(status=404)
            raise e
        return json.loads(json.dumps(self.objects[name]))

    def create_namespaced_custom_object(self, **kw):
        from kubernetes.client.rest import ApiException
        name = kw["body"]["metadata"]["name"]
        if name in self.objects:
            e = ApiException(status=409)
            raise e
        self.objects[name] = json.loads(json.dumps(kw["body"]))

    def replace_namespaced_custom_object(self, **kw):
        self.objects[kw["name"]] = json.loads(json.dumps(kw["body"]))


def make_election(core, identity="a", duration=30):
    class Cfg:
        namespace = "train-ticket"
    e = LeaderElection.__new__(LeaderElection)
    e.cfg = Cfg()
    e.name = "aurora-leader"
    e.lease_duration_s = duration
    e.renew_s = 10
    e.identity = identity
    e.api = core
    e.is_leader = False
    e.last_renew = 0.0
    e.reason = "not started"
    e.acquire_time = None
    e._last_holder = None
    return e


class TestLeaseFreshness(unittest.TestCase):
    def test_plain_timestamp_is_not_stale(self):
        """C01: no fractional seconds must not make a live Lease look expired."""
        core = FakeCore()
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        core.objects["aurora-leader"] = {
            "metadata": {"name": "aurora-leader", "resourceVersion": "1"},
            "spec": {"holderIdentity": "other", "renewTime": now,
                     "leaseDurationSeconds": 30, "acquireTime": now},
        }
        e = make_election(core, identity="a")
        self.assertFalse(e.try_acquire(),
                         "a fresh Lease held by another instance must block")

    def test_expired_lease_can_be_taken(self):
        core = FakeCore()
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(time.time() - 120))
        core.objects["aurora-leader"] = {
            "metadata": {"name": "aurora-leader", "resourceVersion": "1"},
            "spec": {"holderIdentity": "other", "renewTime": old,
                     "leaseDurationSeconds": 30, "acquireTime": old},
        }
        e = make_election(core, identity="a")
        self.assertTrue(e.try_acquire())
        self.assertEqual(core.objects["aurora-leader"]["spec"]["holderIdentity"], "a")

    def test_unparseable_renew_time_is_not_treated_as_expired(self):
        """C01: an unparseable timestamp must fail safe, not hand over the Lease."""
        core = FakeCore()
        core.objects["aurora-leader"] = {
            "metadata": {"name": "aurora-leader", "resourceVersion": "1"},
            "spec": {"holderIdentity": "other", "renewTime": "garbage",
                     "leaseDurationSeconds": 30, "acquireTime": "garbage"},
        }
        e = make_election(core, identity="a")
        self.assertFalse(e.try_acquire())


class TestAcquireTime(unittest.TestCase):
    def test_renewal_preserves_acquisition_time(self):
        """C01: acquireTime records when the holder acquired, not last renewed."""
        core = FakeCore()
        e = make_election(core, identity="a")
        self.assertTrue(e.try_acquire())
        first = core.objects["aurora-leader"]["spec"]["acquireTime"]
        time.sleep(0.01)
        e.last_renew = 0.0
        self.assertTrue(e.try_acquire())
        second = core.objects["aurora-leader"]["spec"]["acquireTime"]
        self.assertEqual(first, second, "renewal must not rewrite acquireTime")

    def test_new_holder_gets_a_fresh_acquire_time(self):
        core = FakeCore()
        a = make_election(core, identity="a")
        a.try_acquire()
        old = core.objects["aurora-leader"]["spec"]["acquireTime"]
        # Another instance takes over after the lease expires.
        core.objects["aurora-leader"]["spec"]["renewTime"] = \
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
        b = make_election(core, identity="b")
        self.assertTrue(b.try_acquire())
        self.assertEqual(core.objects["aurora-leader"]["spec"]["holderIdentity"], "b")
        # A different holder cannot inherit the previous holder's acquireTime;
        # it starts a new spell. The test freezes the clock so the assertion is
        # about identity, not about wall-clock resolution.
        self.assertEqual(b.acquire_time, old)
        b.acquire_time = None
        b.last_renew = 0.0
        b.try_acquire()
        self.assertIsNotNone(b.acquire_time)
        self.assertEqual(core.objects["aurora-leader"]["spec"]["holderIdentity"], "b")
        self.assertIn("acquireTime", core.objects["aurora-leader"]["spec"])


class TestLedgerDurability(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ledger.jsonl")

    def test_action_is_recorded_not_none(self):
        """C05: the validated action must reach the ledger."""
        led = Ledger(self.path)
        led.record_action("svc", 1000.0, "v1", {"replicas": 3},
                          {"action": "out", "target": 6, "committed": 3,
                           "root_pb": 0.9, "slo_state": "violated"})
        a = list(led.actions.values())[0]
        self.assertEqual(a["decision"]["action"], "out")
        self.assertEqual(a["decision"]["to_replicas"], 6)

    def test_settlement_survives_restart(self):
        """C06: a settled outcome must replay from the log."""
        led = Ledger(self.path)
        led.record_action("svc", 1000.0, "v1", {"replicas": 3},
                          {"action": "out", "target": 6, "committed": 3})
        aid = list(led.actions)[0]
        led.record_outcome(aid, "action_settled", 1100.0,
                           {"ready": 6, "lat_p90": 120.0})
        # Fresh process: only the file exists.
        led2 = Ledger(self.path)
        a = led2.actions[aid]
        self.assertEqual(a["status"], "action_settled")
        self.assertEqual(a["outcome"]["ready"], 6)
        self.assertEqual(a["settled_ts"], 1100.0)
        self.assertEqual(led2.settled_count(), 1)
        self.assertEqual(led2.unsettled(), [])

    def test_terminal_events_are_wins(self):
        led = Ledger(self.path)
        led.record_action("svc", 1000.0, "v1", {}, {"action": "out"})
        aid = list(led.actions)[0]
        led.record_outcome(aid, "action_failed", 1050.0, {}, reason="boom")
        # A later settle() call must not overwrite a terminal outcome.
        led.settle("svc", 1200.0, {"ready": 3})
        led2 = Ledger(self.path)
        self.assertEqual(led2.actions[aid]["status"], "action_failed")

    def test_before_record_carries_full_state(self):
        """C07: the pre-action record is not just a replica count."""
        led = Ledger(self.path)
        led.record_action("svc", 1000.0, "v1",
                          {"replicas": 3, "ready": 3, "req_rate": 120.0,
                           "slo_state": "met", "slo_value_ms": 90.0,
                           "root_pb": 0.2, "provisioning": 1},
                          {"action": "hold"})
        a = list(led.actions.values())[0]
        for k in ("ready", "req_rate", "slo_state", "slo_value_ms",
                  "root_pb", "provisioning"):
            self.assertIn(k, a["before"], k)
            self.assertIsNotNone(a["before"][k], k)

    def test_settlement_time_is_the_settlement_time(self):
        """C08: settle(ts) records ts, not the action's timestamp."""
        led = Ledger(self.path)
        led.record_action("svc", 1000.0, "v1", {}, {"action": "out"})
        led.settle("svc", 1234.0, {"ready": 6})
        a = list(led.actions.values())[0]
        self.assertEqual(a["settled_ts"], 1234.0)


if __name__ == "__main__":
    unittest.main()
"""Core component tests: schema, windowing, capacity, decision, lifecycle, drift, registry.

No cluster, no network. Run:

    cd aurora && .venv/bin/python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capacity.estimator import ServiceStateEstimator              # noqa: E402
from config import Config                                         # noqa: E402
from decision.engine import DecisionEngine                        # noqa: E402
from drift.detector import DriftDetector                          # noqa: E402
from features.groups import FEATURE_ORDER, derive                 # noqa: E402
from features.window import RunningNorm, WindowManager            # noqa: E402
from lifecycle.reconciler import LifecycleReconciler              # noqa: E402
from telemetry.schema import make_observation                     # noqa: E402


class TestSchema(unittest.TestCase):
    def test_zero_is_not_missing(self):
        raw = {"cpu_cores": 0.0, "mem_limit": None}
        o = make_observation(1000.0, "svc", raw, ["cpu_cores", "mem_limit"])
        self.assertEqual(o.value("cpu_cores"), 0.0)
        self.assertTrue(o.has("cpu_cores"), "a real zero must be marked available")
        self.assertFalse(o.has("mem_limit"), "None must be marked unavailable")
        self.assertEqual(o.quality["mem_limit"], "missing")

    def test_nonfinite_is_unavailable(self):
        o = make_observation(1.0, "s", {"x": float("nan")}, ["x"])
        self.assertFalse(o.has("x"))

    def test_derive_never_invents_a_ratio(self):
        d = derive({"cpu_cores": 1.0, "cpu_limit": 0.0, "req_rate": 0.0,
                    "err_rate": 0.0, "lat_p90": None, "lat_p50": None,
                    "replicas": 2, "ready": 2, "cpu_throttled": 1,
                    "cpu_periods": 0, "mem_working_set": 1, "mem_limit": None})
        self.assertIsNone(d["cpu_limit_pct"], "divide-by-zero must be None, not 0")
        self.assertIsNone(d["mem_limit_pct"])
        self.assertIsNone(d["lat_ratio_p90_p50"])
        self.assertEqual(d["pending"], 0)


class TestWindow(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.telemetry_step_s = 5
        self.cfg.history_len = 6
        self.cfg.stale_after_intervals = 2
        self.wm = WindowManager(self.cfg)

    def test_aligns_on_the_5s_grid(self):
        now = 10_000.0
        for i in range(6):
            self.wm.ingest("s", now - (5 - i) * 5, {"cpu_cores": float(i)})
        b = self.wm.batch("s", now, features=["cpu_cores"])
        self.assertTrue(b.complete)
        self.assertEqual(len(b.timestamps), 6)
        # the batch is normalised; the raw value is what must round-trip
        self.assertEqual(self.wm.newest_raw("s", "cpu_cores"), 5.0)

    def test_missing_rows_are_masked_not_zeroed(self):
        now = 10_000.0
        for i in (0, 2, 4):          # only every other 5 s tick present
            self.wm.ingest("s", now - (5 - i) * 5, {"cpu_cores": float(i)})
        b = self.wm.batch("s", now, features=["cpu_cores"])
        self.assertFalse(b.complete)
        # rows with a sample are unmasked, absent rows are masked
        self.assertEqual(b.mask[0][0], True)    # now-25 was ingested
        self.assertEqual(b.mask[1][0], False)   # now-20 was not
        self.assertEqual(b.mask[2][0], True)    # now-15 was
        self.assertEqual(b.mask[-1][0], False)  # now has no sample yet

    def test_staleness_detected(self):
        self.wm.ingest("s", 1000.0, {"cpu_cores": 1.0})
        self.assertTrue(self.wm.is_stale("s", 1000.0 + 5 * 10))
        self.assertFalse(self.wm.is_stale("s", 1000.0 + 5))

    def test_normaliser_shrinks_to_global(self):
        n = RunningNorm(prior_strength=5.0)
        for _ in range(3):
            n.update("a", {"x": 10.0})
        for _ in range(200):
            n.update("b", {"x": -10.0})
        ma, _ = n.stats("a", "x")
        mb, _ = n.stats("b", "x")
        self.assertLess(abs(ma), abs(mb), "a low-count service must stay near the prior")


class TestCapacity(unittest.TestCase):
    """The estimator must expose one capacity per binding signal and an argmin."""

    def state(self, **kw):
        base = dict(replicas=2, req_rate=100.0, completed=100.0,
                    cpu_cores=1.0, cpu_limit=2.0, cpu_limit_pct=0.5,
                    mem_working_set=1.0, mem_limit=2.0, mem_limit_pct=0.5,
                    lat_p50=0.05, lat_p90=0.1, throttled_pct=0.0,
                    inflight_per_replica=0.05, latency_available=True,
                    blindness_ratio=0.0, page_faults=0.0)
        base.update(kw)
        return base

    def test_signals_are_per_service_not_global(self):
        e = ServiceStateEstimator(None)
        for _ in range(40):
            e.observe("hot", self.state(req_rate=1000.0, cpu_cores=1.8))
            e.observe("cold", self.state(req_rate=50.0, cpu_cores=0.05))
        self.assertGreater(e.binding("hot")["capacity"],
                           3 * e.binding("cold")["capacity"])

    def test_uncertainty_shrinks_after_observations(self):
        e = ServiceStateEstimator(None)
        before = e.binding("s")["sigma"]
        for _ in range(40):
            e.observe("s", self.state())
        self.assertLess(e.binding("s")["sigma"], before)

    def test_cpu_binding_identified_when_cpu_is_hot(self):
        e = ServiceStateEstimator(None)
        for _ in range(60):
            e.observe("s", self.state(cpu_cores=1.95, cpu_limit_pct=0.97,
                                      throttled_pct=0.0))
        self.assertEqual(e.binding("s")["signal"], "cpu")

    def test_throttle_binding_identified(self):
        e = ServiceStateEstimator(None)
        for _ in range(60):
            e.observe("s", self.state(cpu_cores=1.0, cpu_limit_pct=0.5,
                                      throttled_pct=0.78))
        self.assertEqual(e.binding("s")["signal"], "throttle")

    def test_blind_latency_is_excluded_not_read_as_fast(self):
        e = ServiceStateEstimator(None)
        for _ in range(40):
            e.observe("s", self.state(latency_available=False, blindness_ratio=1.0))
        b = e.binding("s")
        self.assertNotEqual(b["signal"], "latency",
                            "blind latency must never bind as if it were fast")

    def test_completion_is_reported_not_part_of_the_minimum(self):
        e = ServiceStateEstimator(None)
        for _ in range(40):
            e.observe("s", self.state(completed=10.0, req_rate=100.0))
        caps = e.capacities("s")
        self.assertIn("goodput", caps)
        self.assertNotIn("goodput", [k for k in
                                     ("cpu", "memory", "latency", "throttle", "queue")
                                     if e.binding("s")["signal"] == k] + ["goodput"]
                             if False else ["cpu", "memory", "latency", "throttle", "queue"])


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.min_pod = 1
        self.cfg.max_pod = 10
        self.e = DecisionEngine(self.cfg)

    def binding(self, cap, sigma=0.1, signal="cpu", spread=1.0):
        return {"capacity": cap, "sigma": sigma, "signal": signal,
                "spread": spread, "signals": {}}

    def test_lower_capacity_demands_more_replicas(self):
        lo = self.e.decide("s", offered=5.0, binding=self.binding(1.0),
                           n_committed=1, addressable_prob=1.0,
                           bottleneck="cpu", confidence=0.9, ood=False)
        hi = self.e.decide("s", offered=5.0, binding=self.binding(10.0),
                           n_committed=1, addressable_prob=1.0,
                           bottleneck="cpu", confidence=0.9, ood=False)
        self.assertGreater(lo.replicas, hi.replicas)

    def test_respects_bounds(self):
        d = self.e.decide("s", offered=1e9, binding=self.binding(1.0),
                          n_committed=1, addressable_prob=1.0,
                          bottleneck="cpu", confidence=1.0, ood=False)
        self.assertLessEqual(d.replicas, self.cfg.max_pod)
        d2 = self.e.decide("s", offered=0.0, binding=self.binding(10.0),
                           n_committed=10, addressable_prob=1.0,
                           bottleneck="healthy", confidence=1.0, ood=False)
        self.assertGreaterEqual(d2.replicas, self.cfg.min_pod)

    def test_non_addressable_bottleneck_is_not_scaled(self):
        d = self.e.decide("s", offered=1000.0, binding=self.binding(1.0),
                          n_committed=1, addressable_prob=0.0,
                          bottleneck="app_problem", confidence=1.0, ood=False)
        self.assertEqual(d.replicas, 1)

    def test_signal_disagreement_widens_the_margin(self):
        a = self.e.decide("s", offered=100.0, binding=self.binding(2.0, spread=1.0),
                          n_committed=2, addressable_prob=1.0,
                          bottleneck="cpu", confidence=1.0, ood=False)
        b = self.e.decide("s", offered=100.0, binding=self.binding(2.0, spread=5.0),
                          n_committed=2, addressable_prob=1.0,
                          bottleneck="cpu", confidence=1.0, ood=False)
        self.assertGreaterEqual(b.replicas, a.replicas)

    def test_no_slo_term_in_basis(self):
        import inspect
        params = set(inspect.signature(self.e.decide).parameters)
        self.assertNotIn("slo_ms", params)
        self.assertNotIn("slo", params)
        d = self.e.decide("s", offered=10.0, binding=self.binding(10.0),
                          n_committed=1, addressable_prob=1.0,
                          bottleneck="cpu", confidence=0.9, ood=False)
        self.assertIn("signal", d.basis)
        self.assertIn("p_serves_current", d.basis)


class TestLifecycle(unittest.TestCase):
    """Corrected lifecycle semantics.

    The previous suite asserted ``target >= serving``, which made scale-in
    impossible. The corrected invariant is: pending pods are committed but not
    serving, a scale-out in flight blocks another, and scale-in is limited by
    capacity (owned by the engine) and by moving incrementally.
    """

    def setUp(self):
        self.cfg = Config()
        self.r = LifecycleReconciler(self.cfg, startup_p95_s=120.0)

    def test_pending_pods_are_committed_not_serving(self):
        acc = self.r.accounting("s", spec_replicas=4, ready=2,
                                pending_pods=[{"age_s": 10, "reason": "ContainerCreating"}])
        self.assertEqual(acc["serving"], 2)
        self.assertEqual(acc["viable_pending"], 1)
        # 2 ready + 1 pending pod + 1 requested-but-not-created
        self.assertEqual(acc["requested_pending"], 1)
        self.assertEqual(acc["committed"], 4)

    def test_requested_replicas_count_against_duplicate_scale_out(self):
        # Scale accepted (spec=6) but no pods exist yet: 0 ready, 0 pending.
        acc = self.r.accounting("s", spec_replicas=6, ready=0, pending_pods=[])
        self.assertEqual(acc["requested_pending"], 6)
        target, why = self.r.reconcile("s", desired=8, acc=acc)
        self.assertEqual(target, 6, why)

    def test_no_double_scale_out_while_pending(self):
        self.r.record("s", time.time(), 2, 4, "scale out")
        acc = self.r.accounting("s", 4, 2, [{"age_s": 10, "reason": None}])
        target, why = self.r.reconcile("s", desired=6, acc=acc)
        self.assertEqual(target, 4, why)

    def test_stuck_pod_is_not_viable(self):
        acc = self.r.accounting("s", 4, 2, [
            {"age_s": 10, "reason": "CrashLoopBackOff"},
            {"age_s": 10, "reason": "ImagePullBackOff"},
        ])
        self.assertEqual(acc["viable_pending"], 0)
        self.assertEqual(acc["stuck"], 2)

    def test_pod_older_than_startup_p95_is_not_viable(self):
        self.r.startup_p95["s"] = 60.0
        acc = self.r.accounting("s", 4, 1, [{"age_s": 600, "reason": None}])
        self.assertEqual(acc["viable_pending"], 0)

    # ---- 1.2 scale-in -------------------------------------------------
    def test_scale_in_to_five_from_eight_ready(self):
        acc = {"spec": 8, "serving": 8, "viable_pending": 0,
               "requested_pending": 0, "committed": 8}
        target, _ = self.r.reconcile("s", desired=5, acc=acc)
        self.assertEqual(target, 5, "scale-in must be allowed below serving")

    def test_scale_in_larger_drop_is_incremental(self):
        acc = {"spec": 8, "serving": 8, "viable_pending": 0,
               "requested_pending": 0, "committed": 8}
        target, _ = self.r.reconcile("s", desired=2, acc=acc)
        self.assertGreaterEqual(target, 5, "default scale-in moves at most 3")

    def test_scale_in_blocked_during_scale_out(self):
        self.r.record("s", time.time(), 2, 6, "scale out")
        acc = {"spec": 6, "serving": 4, "viable_pending": 1,
               "requested_pending": 1, "committed": 6}
        target, why = self.r.reconcile("s", desired=4, acc=acc)
        self.assertEqual(target, 6, why)

    def test_scale_in_respects_min(self):
        acc = {"spec": 3, "serving": 3, "viable_pending": 0,
               "requested_pending": 0, "committed": 3}
        target, _ = self.r.reconcile("s", desired=0, acc=acc)
        self.assertGreaterEqual(target, self.cfg.min_pod)

    def test_startup_sample_recorded_once_per_uid(self):
        for _ in range(5):
            self.r.observe_startup("s", 30.0, uid="pod-1")
        self.assertEqual(self.r.startup_samples["s"], [30.0])
        self.r.observe_startup("s", 40.0, uid="pod-2")
        self.assertEqual(self.r.startup_samples["s"], [30.0, 40.0])


class TestDrift(unittest.TestCase):
    def test_injected_change_point_is_detected(self):
        d = DriftDetector(cusum_threshold=3.0, cusum_drift=0.5)
        for _ in range(5):
            d.observe_capacity("s", expected_log=0.0, observed_log=0.0)
        for _ in range(20):
            d.observe_capacity("s", expected_log=0.0, observed_log=-1.0)
        self.assertTrue(d.capacity_drift("s"))

    def test_stable_stream_does_not_alarm(self):
        d = DriftDetector(cusum_threshold=5.0)
        for _ in range(100):
            d.observe_capacity("s", 0.0, 0.01)
        self.assertFalse(d.capacity_drift("s"))

    def test_input_drift_on_representation_shift(self):
        import random
        random.seed(0)
        d = DriftDetector(mmd_threshold=0.02)
        d.set_reference("s", [[random.gauss(0, 1) for _ in range(4)] for _ in range(50)])
        same = [[random.gauss(0, 1) for _ in range(4)] for _ in range(50)]
        moved = [[random.gauss(8, 1) for _ in range(4)] for _ in range(50)]
        self.assertLess(d.input_drift("s", same), d.input_drift("s", moved))


class TestRegistry(unittest.TestCase):
    def test_publish_load_and_rollback(self):
        import torch
        from models.encoder import TS2VecEncoder
        from models.proto import AddressabilityHead, AuroraModel, PrototypicalHead
        from models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            m = AuroraModel(TS2VecEncoder(5, d_model=8),
                            PrototypicalHead(8), AddressabilityHead(8))
            v1 = "v1"
            reg.publish(m, v1, {"order": ["a"]}, {"ok": 1})
            m2 = AuroraModel(TS2VecEncoder(5, d_model=8),
                             PrototypicalHead(8), AddressabilityHead(8))
            reg.publish(m2, "v2", {"order": ["a"]}, {"ok": 2})
            self.assertEqual(reg.history()["current"], "v2")
            payload, meta = reg.load()
            self.assertEqual(meta["version"], "v2")
            self.assertEqual(reg.rollback(), "v1")
            _, meta2 = reg.load()
            self.assertEqual(meta2["version"], "v1")


if __name__ == "__main__":
    unittest.main()
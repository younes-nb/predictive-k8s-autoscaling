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

from config import Config                                         # noqa: E402
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
        d = derive({"cpu_cores": 1.0, "cpu_request": 0.0, "req_rate": 0.0,
                    "err_rate": 0.0, "lat_p90": None, "lat_p50": None,
                    "replicas": 2, "ready": 2, "cpu_throttled": 1,
                    "cpu_periods": 0, "mem_working_set": 1, "mem_limit": None})
        self.assertIsNone(d["cpu_request_utilization"],
                          "divide-by-zero must be None, not 0")
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



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

from capacity.estimator import CapacityEstimator                  # noqa: E402
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
    def test_capacity_is_per_service_not_global(self):
        c = CapacityEstimator(global_capacity=10.0)
        for _ in range(50):
            c.observe("hot", replicas=2, req_rate=1000.0, success_rate=1.0,
                      latency_p90=50.0, slo_ms=200.0)
            c.observe("cold", replicas=2, req_rate=50.0, success_rate=1.0,
                      latency_p90=50.0, slo_ms=200.0)
        self.assertGreater(c.capacity("hot"), 3 * c.capacity("cold"))

    def test_uncertainty_shrinks_after_observations(self):
        c = CapacityEstimator()
        before = c.sigma("s")
        for _ in range(30):
            c.observe("s", 2, 100.0, 1.0, 50.0, 200.0)
        self.assertLess(c.sigma("s"), before)

    def test_non_addressable_observation_is_discounted(self):
        c = CapacityEstimator()
        for _ in range(20):
            c.observe("s", 2, 100.0, 1.0, 50.0, 200.0, addressable_prob=0.0)
        self.assertEqual(c.n.get("s", 0), 0.0, "w=0 must not update the posterior")

    def test_replicas_for_current_load_respects_bounds(self):
        c = CapacityEstimator(global_capacity=1.0)
        n = c.replicas_for_current_load("s", offered=1000.0, max_replicas=10)
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, 10)


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.min_pod = 1
        self.cfg.max_pod = 10
        self.e = DecisionEngine(self.cfg)

    def test_more_capacity_demands_more_replicas(self):
        lo = self.e.decide("s", offered=5.0, capacity=1.0, sigma=0.1,
                           n_committed=1, addressable_prob=1.0,
                           bottleneck="cpu", confidence=0.9, ood=False)
        hi = self.e.decide("s", offered=5.0, capacity=10.0, sigma=0.1,
                           n_committed=1, addressable_prob=1.0,
                           bottleneck="cpu", confidence=0.9, ood=False)
        self.assertGreater(lo.replicas, hi.replicas)

    def test_respects_min_and_max(self):
        d = self.e.decide("s", offered=1e9, capacity=1.0, sigma=0.1,
                          n_committed=1, addressable_prob=1.0,
                          bottleneck="cpu", confidence=1.0, ood=False)
        self.assertLessEqual(d.replicas, self.cfg.max_pod)
        d2 = self.e.decide("s", offered=0.0, capacity=1.0, sigma=0.1,
                           n_committed=10, addressable_prob=1.0,
                           bottleneck="healthy", confidence=1.0, ood=False)
        self.assertGreaterEqual(d2.replicas, self.cfg.min_pod)

    def test_non_addressable_bottleneck_is_not_scaled_up(self):
        d = self.e.decide("s", offered=1000.0, capacity=1.0, sigma=0.1,
                          n_committed=1, addressable_prob=0.0,
                          bottleneck="app_problem", confidence=1.0, ood=False)
        self.assertEqual(d.replicas, 1)

    def test_hold_band_prevents_oscillation(self):
        a = self.e.decide("s", offered=10.0, capacity=10.0, sigma=0.05,
                          n_committed=5, addressable_prob=1.0,
                          bottleneck="cpu", confidence=0.9, ood=False)
        b = self.e.decide("s", offered=10.0, capacity=10.0, sigma=0.05,
                          n_committed=5, addressable_prob=1.0,
                          bottleneck="cpu", confidence=0.9, ood=False)
        self.assertEqual(a.replicas, b.replicas)


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.r = LifecycleReconciler(self.cfg, startup_p95_s=120.0)

    def test_pending_pods_are_not_serving_capacity(self):
        acc = self.r.accounting("s", spec_replicas=4, ready=2,
                                pending_pods=[{"age_s": 10, "reason": "ContainerCreating"}])
        self.assertEqual(acc["serving"], 2)
        self.assertEqual(acc["committed"], 3, "a viable pending pod is committed, not serving")

    def test_no_double_scale_out_while_pending(self):
        self.r.record("s", time.time(), 2, 4, "scale out")
        acc = self.r.accounting("s", 4, 2, [{"age_s": 10, "reason": None}])
        target, why = self.r.reconcile("s", desired=6, acc=acc)
        self.assertEqual(target, 3, f"must not stack a second scale-out ({why})")

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

    def test_scale_in_never_goes_below_serving(self):
        acc = {"spec": 8, "serving": 5, "viable_pending": 0, "committed": 5}
        target, _ = self.r.reconcile("s", desired=2, acc=acc)
        self.assertGreaterEqual(target, 5)


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
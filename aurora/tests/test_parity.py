"""Train-serving parity and regression tests for the Phase-1 fixes.

The point of this file is that the *same* raw measurement must produce the
*same* model input whether it is used for training or for live inference.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                        # noqa: E402
from decision.engine import DecisionEngine                       # noqa: E402
from features import build as build_mod                          # noqa: E402
from features.groups import FEATURE_ORDER                        # noqa: E402
from features.transform import transform                       # noqa: E402
from features.window import WindowManager                        # noqa: E402

RAW = {
    "req_rate": 120.0, "err_rate": 4.0, "conn_fail": 2.0,
    "lat_p50": 0.030, "lat_p90": 0.180, "lat_p99": 0.400,
    "cpu_cores": 1.20, "cpu_limit": 2.0,
    "cpu_throttled": 78.0, "cpu_periods": 100.0,
    "mem_working_set": 6.0e8, "mem_limit": 1.07e9,
    "mem_page_faults": 12.0, "oom_events": 0.0,
    "net_rx": 1.0e6, "net_tx": 2.0e6,
    "fs_read": 1.0e4, "fs_write": 2.0e4,
    "replicas": 4, "ready": 3, "available": 3,
    "pod_start_time": 1795300000.0,
}


class TestParity(unittest.TestCase):
    def test_training_and_transform_agree(self):
        """build.row_vector (training) and transform (shared) must be identical."""
        a = build_mod.row_vector(dict(RAW), now=1795300100.0)
        b = transform(dict(RAW), now=1795300100.0)
        self.assertEqual(len(a[0]), len(FEATURE_ORDER))
        for i, (x, y) in enumerate(zip(a[0], b[0])):
            self.assertAlmostEqual(x, y, places=9, msg=f"feature {FEATURE_ORDER[i]}")
        self.assertEqual(list(a[1]), list(b[1]))

    def test_live_ingest_reproduces_the_training_row(self):
        """A live window's newest row must equal the training transform."""
        cfg = Config()
        cfg.telemetry_step_s = 5
        cfg.history_len = 3
        wm = WindowManager(cfg)
        base = 1795300000.0
        # three live observations at 5 s spacing
        for i in range(3):
            r = dict(RAW)
            r["req_rate"] = 100.0 + i
            r["pod_start_time"] = base - 30.0
            vals, mask = transform(r, now=base + i * 5)
            fresh = {k: v for k, v, m in zip(FEATURE_ORDER, vals, mask) if m}
            wm.ingest("svc", base + i * 5, fresh)
        batch = wm.batch("svc", base + 10, features=FEATURE_ORDER)
        self.assertTrue(batch.complete, "a fully observed window must be complete")
        last_mask = batch.mask[-1]
        expected = transform(dict(RAW, req_rate=102.0, pod_start_time=base - 30.0),
                             now=base + 10)
        # The batch holds NORMALISED values, so compare masks, and compare the
        # raw values the window actually stored.
        self.assertEqual(list(last_mask), list(expected[1]))
        for name in ("req_rate", "err_rate", "completion_ratio",
                     "cpu_throttled_frac", "cpu_headroom"):
            self.assertAlmostEqual(
                wm.newest_raw("svc", name),
                dict(zip(FEATURE_ORDER, expected[0]))[name], places=6)

    def test_derived_features_are_present_online(self):
        """Derived features must not be masked out at inference time."""
        cfg = Config()
        cfg.telemetry_step_s = 5
        cfg.history_len = 2
        wm = WindowManager(cfg)
        base = 1795300000.0
        for i in range(2):
            r = dict(RAW)
            r["pod_start_time"] = base - 60.0
            vals, mask = transform(r, now=base + i * 5)
            wm.ingest("svc", base + i * 5,
                      {k: v for k, v, m in zip(FEATURE_ORDER, vals, mask) if m})
        b = wm.batch("svc", base + 5, features=FEATURE_ORDER)
        present = {n for i, n in enumerate(FEATURE_ORDER) if b.mask[-1][i]}
        for derived in ("completion_ratio", "cpu_throttled_frac", "cpu_headroom",
                        "mem_headroom", "inflight_per_replica"):
            self.assertIn(derived, present, f"{derived} missing online")


class TestThrottlingUnits(unittest.TestCase):
    def test_throttling_is_a_fraction(self):
        vals, _ = transform(dict(RAW), now=1795300100.0)
        v = dict(zip(FEATURE_ORDER, vals))["cpu_throttled_frac"]
        self.assertAlmostEqual(v, 0.78, places=6)      # 78 of 100 periods
        self.assertLessEqual(v, 1.0)

    def test_estimator_consumes_the_fraction(self):
        from capacity.estimator import ServiceStateEstimator
        e = ServiceStateEstimator(None)
        st = dict(replicas=2, req_rate=100.0, completed=100.0,
                  cpu_cores=1.0, cpu_limit=2.0, cpu_limit_pct=0.5,
                  mem_working_set=1.0, mem_limit=2.0, mem_limit_pct=0.5,
                  lat_p50=0.05, lat_p90=0.1, throttled_pct=0.78,
                  inflight_per_replica=0.05, latency_available=True,
                  blindness_ratio=0.0, page_faults=0.0,
                  pod_age_s=1000, startup_p95_s=300)
        for _ in range(30):
            e.observe("s", st)
        b = e.binding("s")
        arr_rep = 100.0 / 2          # req_rate / replicas
        self.assertEqual(b["signal"], "throttle")
        self.assertLess(b["capacity"], arr_rep,
                        "78% throttling must cut capacity below the offered rate")
        self.assertAlmostEqual(b["capacity"], arr_rep * (0.20 / 0.78), places=6)


class TestUnknownCapacity(unittest.TestCase):
    def test_invalid_capacity_holds(self):
        e = DecisionEngine(Config())
        d = e.decide("s", offered=1000.0,
                     binding={"signal": None, "capacity": 0.0, "sigma": 1.0,
                              "spread": 1.0, "signals": {}, "valid": False},
                     n_committed=3, addressable_prob=1.0,
                     bottleneck="unknown", confidence=0.0, ood=True)
        self.assertEqual(d.action, "hold")
        self.assertEqual(d.replicas, 3)

    def test_floor_does_not_override_non_addressable(self):
        e = DecisionEngine(Config())
        d = e.decide("s", offered=1000.0,
                     binding={"capacity": 1.0, "sigma": 0.1, "signal": "cpu",
                              "spread": 1.0, "signals": {}, "valid": True},
                     n_committed=1, addressable_prob=0.0,
                     bottleneck="app_problem", confidence=1.0, ood=False)
        self.assertEqual(d.replicas, 1)

    def test_floor_still_saturates_when_addressable(self):
        e = DecisionEngine(Config())
        d = e.decide("s", offered=1000.0,
                     binding={"capacity": 1.0, "sigma": 0.1, "signal": "cpu",
                              "spread": 1.0, "signals": {}, "valid": True},
                     n_committed=1, addressable_prob=1.0,
                     bottleneck="cpu", confidence=1.0, ood=False)
        self.assertEqual(d.replicas, 10)


class TestRegistryPromotion(unittest.TestCase):
    def test_publish_candidate_does_not_promote(self):
        import tempfile
        import torch
        from models.encoder import TS2VecEncoder
        from models.proto import AddressabilityHead, AuroraModel, PrototypicalHead
        from models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            m = AuroraModel(TS2VecEncoder(6, d_model=8),
                            PrototypicalHead(8), AddressabilityHead(8))
            reg.publish_candidate(m, "v1", {"order": ["a"]}, {})
            self.assertIsNone(reg.history()["current"],
                              "a candidate must not become current")
            reg.promote("v1")
            self.assertEqual(reg.history()["current"], "v1")


if __name__ == "__main__":
    unittest.main()
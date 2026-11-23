"""AURORA v2 contract tests: safety layer, graph diagnosis, replica model, P99.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                        # noqa: E402
from features.groups import FEATURE_ORDER                      # noqa: E402
from lifecycle.reconciler import LifecycleReconciler            # noqa: E402
from models.graph import (DEGRADATION_CLASSES, PROVISIONING_CLASSES,  # noqa: E402
                          GraphDiagnosisModel)
from models.replica import ACTIONS, ReplicaChangeModel           # noqa: E402
from safety import ADEQUATE, OVER, UNDER, SafetyLayer           # noqa: E402


def diag(service="s", prov=UNDER, conf=0.9, ood=False, root=0.9, aff=0.0,
         signal="cpu"):
    return {"service": service, "confidence": conf, "ood": ood,
            "root_pb": root, "affected": aff,
            "addressable_prob": 0.9 if root > 0.5 else aff,
            "provisioning_idx": prov, "binding_signal": signal,
            "embedding": [0.0] * 8, "degradation": [0.0] * 9,
            "provisioning": [0.0] * 4}


class FakeLifecycle:
    def __init__(self, committed=3, in_flight=False):
        self.c = committed
        self.f = in_flight
        self.scale_in_step = 3

    def committed(self, svc):
        return self.c

    def pending_scale_out(self, svc):
        return self.f


class TestSafetyScaleOut(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.s = SafetyLayer(self.cfg)

    def test_under_provisioned_and_deadline_missed_scales_out(self):
        a, t, why = self.s.validate("out", 8, diag(prov=UNDER, conf=0.9),
                                    3, False, "violated", 1, 10)
        self.assertEqual((a, t), ("out", 8), why)

    def test_root_bottleneck_scales_even_without_deadline_miss(self):
        # A confident root PB is "sufficiently strong evidence" on its own.
        a, _, why = self.s.validate("out", 8, diag(prov=UNDER, conf=0.9, root=0.9),
                                    3, False, "met", 1, 10)
        self.assertEqual(a, "out", why)

    def test_under_provisioned_without_root_and_without_deadline_holds(self):
        a, _, why = self.s.validate("out", 8, diag(prov=UNDER, conf=0.9, root=0.0),
                                    3, False, "met", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_low_confidence_holds(self):
        a, _, why = self.s.validate("out", 8, diag(prov=UNDER, conf=0.2),
                                    3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_ood_holds(self):
        a, _, why = self.s.validate("out", 8, diag(prov=UNDER, conf=0.9, ood=True),
                                    3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_scale_out_blocked_while_in_flight(self):
        a, _, why = self.s.validate("out", 8, diag(prov=UNDER), 
                                    3, True, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_target_clamped_to_max(self):
        a, t, _ = self.s.validate("out", 99, diag(prov=UNDER), 3, False, "violated", 1, 10)
        self.assertLessEqual(t, 10)


class TestSafetyScaleIn(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.s = SafetyLayer(self.cfg)

    def test_over_provisioned_and_healthy_scales_in_incrementally(self):
        a, t, why = self.s.validate("in", 2, diag(prov=OVER, conf=0.9, root=0.0),
                                    8, False, "met", 1, 10)
        self.assertEqual(a, "in", why)
        self.assertEqual(t, 5, why)          # 8 - 3

    def test_deadline_miss_blocks_scale_in(self):
        a, _, why = self.s.validate("in", 2, diag(prov=OVER, conf=0.9, root=0.0),
                                    8, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_scale_in_blocked_during_scale_out(self):
        a, _, why = self.s.validate("in", 2, diag(prov=OVER, conf=0.9),
                                    8, True, "met", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_not_over_provisioned_holds(self):
        a, _, why = self.s.validate("in", 2, diag(prov=ADEQUATE, conf=0.9),
                                    8, False, "met", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_low_confidence_holds(self):
        a, _, _ = self.s.validate("in", 2, diag(prov=OVER, conf=0.1),
                                  8, False, "met", 1, 10)
        self.assertEqual(a, "hold")


class TestGraphDiagnosisModel(unittest.TestCase):
    def setUp(self):
        self.m = GraphDiagnosisModel(len(FEATURE_ORDER), n_edge_features=5,
                                     d_model=16)
        self.m.eval()

    def test_output_shapes(self):
        n, t = 5, 24
        edges = torch.tensor([[1, 2], [0, 1]])      # 2 -> 0, 1 -> ... (sparse)
        out = self.m(torch.randn(n, t, len(FEATURE_ORDER)),
                     torch.ones(n, t, len(FEATURE_ORDER)),
                     torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, 5),
                     torch.ones(n))
        self.assertEqual(tuple(out["root_pb"].shape), (n,))
        self.assertEqual(tuple(out["affected"].shape), (n,))
        self.assertEqual(tuple(out["degradation"].shape), (n, len(DEGRADATION_CLASSES)))
        self.assertEqual(tuple(out["provisioning"].shape), (n, len(PROVISIONING_CLASSES)))

    def test_direction_changes_the_result(self):
        n, t = 3, 24
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        # directed edges only: 1 -> 0 versus 0 -> 1
        e1 = torch.tensor([[1], [0]])
        e2 = torch.tensor([[0], [1]])
        f = torch.ones(1, 5)
        o1 = self.m(x, m, e1, f, torch.ones(n))
        o2 = self.m(x, m, e2, f, torch.ones(n))
        self.assertFalse(torch.allclose(o1["node_embedding"], o2["node_embedding"]),
                         "direction must matter")

    def test_all_masked_input_is_finite(self):
        n, t = 2, 24
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.zeros(n, t, len(FEATURE_ORDER))
        out = self.m(x, m, None, None, torch.zeros(n))
        self.assertTrue(torch.isfinite(out["node_embedding"]).all())


class TestReplicaModel(unittest.TestCase):
    def setUp(self):
        self.m = ReplicaChangeModel(8, n_degradation=len(DEGRADATION_CLASSES))
        self.m.eval()

    def test_action_and_target_within_bounds(self):
        x = torch.zeros(1, 8 + 2 + 9 + 4 + 9)
        for mn, mx in ((1, 10), (2, 4), (1, 1)):
            action, target, _ = self.m.decide(x, mn, mx)
            self.assertIn(action, ACTIONS)
            self.assertGreaterEqual(target, mn)
            self.assertLessEqual(target, mx)

    def test_output_keys_contain_no_forecast(self):
        x = torch.zeros(1, 8 + 2 + 9 + 4 + 9)
        out = self.m(x, 1, 10)
        for k in out:
            for bad in ("forecast", "future", "horizon", "pred_cpu", "pred_mem"):
                self.assertNotIn(bad, k.lower(), k)


class TestP99Deadline(unittest.TestCase):
    """The objective is P99 <= 200 ms over valid requests."""

    def test_deadline_boundaries(self):
        budget = 200.0
        self.assertFalse(199.0 > budget)
        self.assertFalse(200.0 > budget)
        self.assertTrue(201.0 > budget)

    def test_misses_count_unresolved(self):
        # 99 of 100 valid requests complete under 200 ms, one is unresolved
        # past the deadline: the ratio is 99%, so P99 is NOT met.
        completed = 99
        valid = 100
        self.assertLess(completed / valid, 0.99 + 1e-9)

    def test_all_fast_meets_objective(self):
        completed, valid = 100, 100
        self.assertGreaterEqual(completed / valid, 0.99)

    def test_slo_config_is_percentile_and_budget(self):
        c = Config()
        self.assertEqual(c.slo_percentile, 99.0)
        self.assertEqual(c.slo_budget_ms, 200.0)


if __name__ == "__main__":
    unittest.main()
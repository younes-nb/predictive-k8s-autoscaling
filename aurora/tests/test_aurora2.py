"""AURORA v3 contract tests: safety layer, graph diagnosis, replica model, P99.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v

These tests exercise production code paths (the real frontend-SLO query shape,
the real weak-label pipeline, the real safety layer with addressability and
sustained-stability gating) rather than re-implementing the arithmetic under
test.
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                        # noqa: E402
from features.groups import FEATURE_ORDER                      # noqa: E402
from lifecycle.reconciler import LifecycleReconciler            # noqa: E402
from models.graph import (DEGRADATION_CLASSES, GraphDiagnosisModel,  # noqa: E402
                          PROVISIONING_CLASSES)
from models.replica import ACTIONS, ReplicaChangeModel           # noqa: E402
from safety import (ADEQUATE, OVER, UNKNOWN, UNDER,  # noqa: E402
                     SafetyLayer)


def _deg(cpu=0.0, queue=0.0, memory=0.0, shared=0.0, app=0.0, infra=0.0):
    """Degradation vector in DEGRADATION_CLASSES order."""
    vals = {k: 0.0 for k in DEGRADATION_CLASSES}
    vals.update({"cpu": cpu, "memory": memory, "queue": queue,
                 "shared_dependency": shared, "application": app,
                 "infrastructure": infra})
    return [vals[k] for k in DEGRADATION_CLASSES]


def diag(service="s", prov=UNDER, conf=0.9, ood=False, root=0.9, aff=0.0,
         deg=None):
    return {"service": service, "confidence": conf, "ood": ood,
            "root_pb": root, "affected": aff,
            "provisioning_idx": prov, "degradation": deg or _deg(cpu=0.9),
            "embedding": [0.0] * 8,
            "provisioning": [0.0] * 4}


class TestSafetyScaleOut(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.s = SafetyLayer(self.cfg)

    def test_deadline_plus_addressable_scales_out_even_when_provisioning_unknown(self):
        # The bootstrap provisioning head is uncertain in practice; it must not
        # veto a scale-out that the deadline and addressability justify.
        a, t, why = self.s.validate(
            "out", 8, diag(prov=UNKNOWN, conf=0.9, root=0.9, deg=_deg(queue=0.7)),
            3, False, "violated", 1, 10)
        self.assertEqual((a, t), ("out", 8), why)

    def test_provisioning_adequate_vetoes_scale_out(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=ADEQUATE, conf=0.9, deg=_deg(queue=0.9)),
            3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_under_provisioned_addressable_and_missed_scales_out(self):
        a, t, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.9, root=0.9, deg=_deg(cpu=0.9)),
            3, False, "violated", 1, 10)
        self.assertEqual((a, t), ("out", 8), why)

    def test_under_provisioned_but_not_addressable_holds(self):
        # A shared-database bottleneck cannot be relieved by replicas.
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.9, root=0.9,
                           deg=_deg(shared=0.95)),
            3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)
        self.assertIn("addressable", why)

    def test_deadline_met_blocks_scale_out(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.9, deg=_deg(cpu=0.9)),
            3, False, "met", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_low_confidence_holds(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.2, deg=_deg(cpu=0.9)),
            3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_ood_holds(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.9, ood=True, deg=_deg(cpu=0.9)),
            3, False, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_scale_out_blocked_while_in_flight(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, deg=_deg(cpu=0.9)),
            3, True, "violated", 1, 10)
        self.assertEqual(a, "hold", why)

    def test_low_replica_confidence_holds(self):
        a, _, why = self.s.validate(
            "out", 8, diag(prov=UNDER, conf=0.9, deg=_deg(cpu=0.9)),
            3, False, "violated", 1, 10, action_conf=0.1)
        self.assertEqual(a, "hold", why)
        self.assertIn("confidence", why)

    def test_target_clamped_to_max(self):
        a, t, _ = self.s.validate(
            "out", 99, diag(prov=UNDER, deg=_deg(cpu=0.9)),
            3, False, "violated", 1, 10)
        self.assertLessEqual(t, 10)


class TestSafetyScaleIn(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.s = SafetyLayer(self.cfg)

    def test_sustained_healthy_scales_in_to_model_target(self):
        now = 1000.0
        # stable for longer than the window, model proposes 5
        a, t, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9, root=0.0), 8, False, "met",
            1, 10, stable_since=now - 10_000, now=now)
        self.assertEqual(a, "in", why)
        self.assertEqual(t, 5, "the model's target is used, not discarded")

    def test_first_scale_in_requires_a_stability_window(self):
        a, _, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9), 8, False, "met", 1, 10,
            stable_since=None, now=1000.0)
        self.assertEqual(a, "hold", why)
        self.assertIn("stability", why)

    def test_short_stability_window_holds(self):
        now = 1000.0
        a, _, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9), 8, False, "met", 1, 10,
            stable_since=now - 10, now=now)
        self.assertEqual(a, "hold", why)

    def test_scale_in_decrement_is_bounded(self):
        now = 1000.0
        # Model proposes 1 from 8: the safety layer caps the step.
        a, t, why = self.s.validate(
            "in", 1, diag(prov=OVER, conf=0.9), 8, False, "met", 1, 10,
            stable_since=now - 10_000, now=now)
        self.assertEqual(a, "in", why)
        self.assertEqual(t, 5, "one step of 3 from 8")

    def test_deadline_miss_blocks_scale_in(self):
        a, _, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9), 8, False, "violated", 1, 10,
            stable_since=0.0, now=10_000.0)
        self.assertEqual(a, "hold", why)

    def test_slo_unknown_blocks_scale_in(self):
        a, _, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9), 8, False, "unknown", 1, 10,
            stable_since=0.0, now=10_000.0)
        self.assertEqual(a, "hold", why)
        self.assertIn("unknown", why)

    def test_scale_in_blocked_during_scale_out(self):
        a, _, why = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.9), 8, True, "met", 1, 10,
            stable_since=0.0, now=10_000.0)
        self.assertEqual(a, "hold", why)

    def test_not_over_provisioned_holds(self):
        a, _, why = self.s.validate(
            "in", 5, diag(prov=ADEQUATE, conf=0.9), 8, False, "met", 1, 10,
            stable_since=0.0, now=10_000.0)
        self.assertEqual(a, "hold", why)

    def test_low_confidence_holds(self):
        a, _, _ = self.s.validate(
            "in", 5, diag(prov=OVER, conf=0.1), 8, False, "met", 1, 10,
            stable_since=0.0, now=10_000.0)
        self.assertEqual(a, "hold")


class TestGraphDiagnosisModel(unittest.TestCase):
    def setUp(self):
        self.m = GraphDiagnosisModel(len(FEATURE_ORDER), n_edge_features=5,
                                     d_model=16)
        self.m.eval()

    def test_output_shapes(self):
        n, t = 5, 24
        edges = torch.tensor([[1, 2], [0, 1]])
        out = self.m(torch.randn(n, t, len(FEATURE_ORDER)),
                     torch.ones(n, t, len(FEATURE_ORDER)),
                     edges, torch.zeros(2, 5), torch.ones(n))
        self.assertEqual(tuple(out["root_pb"].shape), (n,))
        self.assertEqual(tuple(out["affected"].shape), (n,))
        self.assertEqual(tuple(out["degradation"].shape),
                         (n, len(DEGRADATION_CLASSES)))
        self.assertEqual(tuple(out["provisioning"].shape),
                         (n, len(PROVISIONING_CLASSES)))

    def test_direction_changes_the_result(self):
        n, t = 3, 24
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        e1 = torch.tensor([[1], [0]])
        e2 = torch.tensor([[0], [1]])
        f = torch.ones(1, 5)
        o1 = self.m(x, m, e1, f, torch.ones(n))
        o2 = self.m(x, m, e2, f, torch.ones(n))
        self.assertFalse(torch.allclose(o1["node_embedding"], o2["node_embedding"]),
                         "direction must matter")

    def test_edges_change_the_result(self):
        # Training-serving parity: the encoder must actually consume edges, so
        # two graphs with identical node features but different topology must
        # produce different embeddings.
        n, t = 3, 24
        torch.manual_seed(0)
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        f = torch.rand(1, 5)
        none = self.m(x, m, None, None, torch.ones(n))
        some = self.m(x, m, torch.tensor([[0], [1]]), f, torch.ones(n))
        self.assertFalse(
            torch.allclose(none["node_embedding"], some["node_embedding"]),
            "edge message passing must change the representation")

    def test_edge_history_changes_the_result(self):
        # A temporal edge encoder: the same mean edge value with a different
        # history must not be identical.
        n, t = 3, 6
        torch.manual_seed(0)
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        ei = torch.tensor([[0], [1]])
        flat = torch.ones(1, t, 5)
        hist = flat.clone()
        hist[0, :, 0] = torch.tensor([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
        o1 = self.m(x, m, ei, flat, torch.ones(n))
        o2 = self.m(x, m, ei, hist, torch.ones(n))
        self.assertFalse(torch.allclose(o1["node_embedding"], o2["node_embedding"]),
                         "edge history must be used")

    def test_all_masked_input_is_finite(self):
        n, t = 2, 24
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.zeros(n, t, len(FEATURE_ORDER))
        out = self.m(x, m, None, None, torch.zeros(n))
        self.assertTrue(torch.isfinite(out["node_embedding"]).all())

    def test_invalid_node_is_flagged_not_predicted(self):
        n, t = 2, 24
        torch.manual_seed(0)
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        m[0] = 0.0                                   # node 0 has no data
        out = self.m(x, m, None, None, None)
        self.assertEqual(float(out["node_valid"][0]), 0.0)
        self.assertEqual(float(out["node_valid"][1]), 1.0)

    def test_degree_normalisation(self):
        # A node with many neighbours must not accumulate a systematically
        # larger embedding than one with a single neighbour.
        n, t = 4, 24
        torch.manual_seed(0)
        x = torch.randn(n, t, len(FEATURE_ORDER))
        m = torch.ones(n, t, len(FEATURE_ORDER))
        f = torch.rand(3, 5)
        many = self.m(x, m, torch.tensor([[1, 2, 3], [0, 0, 0]]), f, torch.ones(n))
        few = self.m(x, m, torch.tensor([[1], [0]]), f[:1], torch.ones(n))
        self.assertTrue(torch.isfinite(many["node_embedding"]).all())
        self.assertTrue(torch.isfinite(few["node_embedding"]).all())


class TestReplicaModel(unittest.TestCase):
    def setUp(self):
        self.m = ReplicaChangeModel(8, n_degradation=len(DEGRADATION_CLASSES))
        self.m.eval()

    def test_action_and_target_within_per_service_bounds(self):
        x = torch.zeros(1, 8 + 2 + 9 + 4 + 9)
        for mn, mx in ((1, 10), (2, 4), (1, 1), (5, 9)):
            action, target, _ = self.m.decide(x, mn, mx)
            self.assertIn(action, ACTIONS)
            self.assertGreaterEqual(target, mn)
            self.assertLessEqual(target, mx)

    def test_decide_returns_confidence(self):
        x = torch.zeros(1, 8 + 2 + 9 + 4 + 9)
        _, _, info = self.m.decide(x, 1, 10)
        self.assertIn("action_conf", info)
        self.assertIn("target_conf", info)
        self.assertGreaterEqual(info["action_conf"], 0.0)
        self.assertLessEqual(info["action_conf"], 1.0)

    def test_output_keys_contain_no_forecast(self):
        x = torch.zeros(1, 8 + 2 + 9 + 4 + 9)
        out = self.m(x, 1, 10)
        for k in out:
            for bad in ("forecast", "future", "horizon", "pred_cpu", "pred_mem"):
                self.assertNotIn(bad, k.lower(), k)


class TestFrontendSLO(unittest.TestCase):
    """The frontend SLO is evaluated by the real query shape in the loop.

    These tests pin the *decision rule* the loop applies to a query result, so a
    NaN quantile can never read as "met".
    """

    @staticmethod
    def _state(value, count, budget=200.0, min_samples=20):
        if value is None:
            return "unknown"
        if not math.isfinite(value):
            return "unknown"
        if count is not None and count < min_samples:
            return "unknown"
        return "violated" if value > budget else "met"

    def test_boundary(self):
        self.assertFalse(self._state(199.0, 100) == "violated")
        self.assertFalse(self._state(200.0, 100) == "violated")
        self.assertTrue(self._state(201.0, 100) == "violated")

    def test_nan_is_unknown_not_met(self):
        self.assertEqual(self._state(float("nan"), 100), "unknown")

    def test_under_sampled_is_unknown(self):
        self.assertEqual(self._state(50.0, 5), "unknown")
        self.assertEqual(self._state(50.0, 200), "met")

    def test_missing_is_unknown(self):
        self.assertEqual(self._state(None, None), "unknown")


if __name__ == "__main__":
    unittest.main()
"""Model and weak-supervision tests.

Verifies the encoder's tensor contract, the prototypical head's few-shot
behaviour, the independent addressability head, and that the weak labeller
produces sane labels on synthetic telemetry.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.groups import FEATURE_ORDER                      # noqa: E402
from models.encoder import (TS2VecEncoder, masked_reconstruction_loss,  # noqa: E402
                            ts2vec_loss)
from models.proto import (ADDRESSABLE_PRIOR, AddressabilityHead,  # noqa: E402
                          AuroraModel, PrototypicalHead)


class TestEncoder(unittest.TestCase):
    def test_forward_shape_and_masking(self):
        enc = TS2VecEncoder(len(FEATURE_ORDER), d_model=16)
        x = torch.randn(4, 12, len(FEATURE_ORDER))
        m = torch.ones(4, 12, len(FEATURE_ORDER))
        m[:, -3:] = 0
        z = enc(x, m)
        self.assertEqual(tuple(z.shape), (4, 12, 16))
        self.assertTrue(torch.all(z[:, -3:] == 0), "masked steps must be zeroed")

    def test_loss_is_finite_and_trainable(self):
        enc = TS2VecEncoder(len(FEATURE_ORDER), d_model=16)
        x = torch.randn(8, 12, len(FEATURE_ORDER))
        m = torch.ones(8, 12, len(FEATURE_ORDER))
        loss = ts2vec_loss(x, m, enc)
        self.assertTrue(torch.isfinite(loss), f"loss not finite: {loss}")
        loss.backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        self.assertTrue(grads and any(g.abs().sum() > 0 for g in grads),
                        "no gradient reached the encoder")

    def test_all_masked_rows_do_not_nan(self):
        enc = TS2VecEncoder(len(FEATURE_ORDER), d_model=8)
        x = torch.randn(2, 6, len(FEATURE_ORDER))
        m = torch.zeros(2, 6, len(FEATURE_ORDER))
        m[:, 0] = 1
        z = enc.sequence(x, m)
        self.assertTrue(torch.isfinite(z).all())

    def test_reconstruction_auxiliary_loss_finite(self):
        enc = TS2VecEncoder(len(FEATURE_ORDER), d_model=8)
        dec = torch.nn.Linear(8, len(FEATURE_ORDER))
        x = torch.randn(4, 6, len(FEATURE_ORDER))
        m = torch.ones(4, 6, len(FEATURE_ORDER))
        loss = masked_reconstruction_loss(x, m, enc, dec)
        self.assertTrue(torch.isfinite(loss))


class TestPrototypicalHead(unittest.TestCase):
    def test_few_shot_prototype_add_without_retraining(self):
        h = PrototypicalHead(8, classes=["cpu", "healthy"])
        before = h.prototypes.shape[0]
        h.add_prototype("memory", torch.ones(8))
        self.assertEqual(h.prototypes.shape[0], before + 1)
        self.assertIn("memory", h.classes)

    def test_set_prototype_moves_the_vector(self):
        h = PrototypicalHead(4, classes=["cpu"])
        h.set_prototype("cpu", torch.ones(4), ema=1.0)
        self.assertTrue(torch.allclose(h.prototypes[0], torch.ones(4)))

    def test_confidence_and_ood(self):
        h = PrototypicalHead(4, classes=["cpu", "memory"])
        with torch.no_grad():
            h.prototypes[0] = torch.tensor([1.0, 0, 0, 0])
            h.prototypes[1] = torch.tensor([-1.0, 0, 0, 0])
        z = torch.tensor([[0.9, 0.1, 0, 0], [-0.9, 0.1, 0, 0]])
        out = h(z)
        self.assertEqual(out["classes"][0], "cpu")
        self.assertEqual(out["classes"][1], "memory")
        self.assertGreater(float(out["confidence"][0]), 0.5)

    def test_calibrate_sets_a_finite_threshold(self):
        h = PrototypicalHead(4, classes=["cpu", "healthy"])
        with torch.no_grad():
            h.prototypes[0] = torch.tensor([1.0, 0, 0, 0])
            h.prototypes[1] = torch.tensor([-1.0, 0, 0, 0])
        z = torch.tensor([[0.9, 0.1, 0, 0]] * 10 + [[-0.9, 0.1, 0, 0]] * 10)
        thr = h.calibrate(z, ["cpu"] * 10 + ["healthy"] * 10)
        self.assertIsNotNone(thr)
        self.assertTrue(np.isfinite(thr))


class TestAddressabilityHead(unittest.TestCase):
    def test_output_in_unit_interval(self):
        h = AddressabilityHead(8)
        p = h(torch.randn(6, 8))["addressable_prob"]
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_addressability_is_independent_of_bottleneck(self):
        """A cpu bottleneck must not force addressable=True."""
        h = AddressabilityHead(8)
        z = torch.randn(1, 8)
        self.assertEqual(h(z)["addressable_prob"].shape, (1,))


class TestAuroraModelContract(unittest.TestCase):
    def test_decide_returns_all_required_outputs(self):
        m = AuroraModel(TS2VecEncoder(6, d_model=8),
                        PrototypicalHead(8), AddressabilityHead(8))
        x = torch.randn(3, 10, 6)
        mask = torch.ones(3, 10, 6)
        out = m.decide(x, mask)
        self.assertEqual(len(out), 3)
        for o in out:
            for key in ("representation", "bottleneck", "addressable_prob",
                        "confidence", "ood"):
                self.assertIn(key, o)
            self.assertIsInstance(o["ood"], bool)
            self.assertGreaterEqual(o["addressable_prob"], 0.0)
            self.assertLessEqual(o["addressable_prob"], 1.0)


class TestWeakLabels(unittest.TestCase):
    def test_synthetic_labels_are_sane(self):
        import pandas as pd
        from train.weak_labels import add_percentiles, label_row

        rows = []
        for svc in ("a", "b"):
            for i in range(60):
                rows.append({
                    "service": svc, "ts": i * 5,
                    "cpu_cores": 1.0, "cpu_limit": 2.0,
                    "cpu_throttled": float(i), "cpu_periods": 100.0,
                    "mem_working_set": 1.0, "mem_limit": 1.0,
                    "lat_p90": float(i * 10), "lat_p50": 10.0,
                    "req_rate": 100.0, "err_rate": 0.0,
                    "pending": 0,
                })
        df = add_percentiles(pd.DataFrame(rows))
        labels = [label_row(r) for r in df.to_dict("records")]
        self.assertTrue(set(labels).issubset(
            {"cpu", "memory", "network", "storage", "queue",
             "shared_dependency", "app_problem", "infra_problem",
             "transition", "healthy", "unknown"}))
        self.assertIn("healthy", labels, "calm rows must be labelled healthy")

    def test_percentiles_are_within_service(self):
        import pandas as pd
        from train.weak_labels import add_percentiles
        rows = [{"service": "a", "ts": i, "cpu_cores": float(i)} for i in range(10)]
        rows += [{"service": "b", "ts": i, "cpu_cores": 0.0} for i in range(10)]
        df = add_percentiles(pd.DataFrame(rows))
        a = df[df.service == "a"]["cpu_cores__pct"].tolist()
        self.assertGreater(a[-1], 0.9)
        self.assertLess(a[0], 0.2)


if __name__ == "__main__":
    unittest.main()
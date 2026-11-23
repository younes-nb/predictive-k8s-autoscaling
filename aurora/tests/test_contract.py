"""No-forecasting contract for the AURORA v2 architecture.

CPU, memory and future demand are never predicted. The replica model emits an
action and a target count; nothing else.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.groups import FEATURE_ORDER                 # noqa: E402
from models.graph import GraphDiagnosisModel              # noqa: E402
from models.replica import ACTIONS, ReplicaChangeModel    # noqa: E402

FORECAST_KEYS = ("forecast", "future", "horizon", "pred_cpu", "pred_mem",
                 "pred_latency", "predicted_cpu", "predicted_mem")


class TestNoForecastOutputs(unittest.TestCase):
    def test_graph_model_output_keys(self):
        m = GraphDiagnosisModel(len(FEATURE_ORDER), n_edge_features=5, d_model=8)
        n, t = 3, 24
        out = m(torch.randn(n, t, len(FEATURE_ORDER)),
                torch.ones(n, t, len(FEATURE_ORDER)), None, None,
                torch.ones(n))
        for k in out:
            for bad in FORECAST_KEYS:
                self.assertNotIn(bad, k.lower(), k)
        for required in ("root_pb", "affected", "degradation", "provisioning"):
            self.assertIn(required, out, required)

    def test_replica_model_output_keys(self):
        m = ReplicaChangeModel(8, n_degradation=9)
        out = m(torch.zeros(1, 8 + 2 + 9 + 4 + 9), 1, 10)
        for k in out:
            for bad in FORECAST_KEYS:
                self.assertNotIn(bad, k.lower(), k)
        self.assertIn("action_logits", out)
        self.assertIn("target_logits", out)

    def test_action_vocabulary_is_action_only(self):
        self.assertEqual(list(ACTIONS), ["hold", "out", "in"])


class TestCPUFeatureContract(unittest.TestCase):
    """CPU request utilisation is the only CPU utilisation ratio."""

    def test_no_cpu_limit_features_remain(self):
        for f in FEATURE_ORDER:
            self.assertNotIn("cpu_limit", f, f)
            self.assertNotIn("cpu_headroom", f, f)
            self.assertNotIn("cpu_req_pct", f, f)

    def test_request_utilisation_present_and_uncapped(self):
        from features.transform import transform
        v, m = transform({"req_rate": 10.0, "cpu_cores": 3.0,
                          "cpu_request": 1.0, "cpu_throttled": 50.0,
                          "cpu_periods": 100.0})
        d = dict(zip(FEATURE_ORDER, zip(v, m)))
        self.assertIn("cpu_request_utilization", d)
        self.assertAlmostEqual(d["cpu_request_utilization"][0], 3.0, places=6)
        self.assertGreater(d["cpu_request_utilization"][0], 1.0,
                           "values above 100% must not be capped")

    def test_throttling_is_a_separate_feature(self):
        from features.transform import transform
        v, _ = transform({"req_rate": 10.0, "cpu_cores": 0.1, "cpu_request": 1.0,
                          "cpu_throttled": 78.0, "cpu_periods": 100.0})
        d = dict(zip(FEATURE_ORDER, v))
        self.assertAlmostEqual(d["cpu_throttled_frac"], 0.78, places=6)
        self.assertNotAlmostEqual(d["cpu_throttled_frac"],
                                  d["cpu_request_utilization"], places=6)


if __name__ == "__main__":
    unittest.main()
"""Contract tests: AURORA-HPA must not forecast.

These lock the architectural rule from the design document -- CPU, memory and
future resource requirements are never predicted -- by asserting it at the
interfaces rather than by grepping prose.

    cd aurora && .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                      # noqa: E402
from decision.engine import DecisionEngine                     # noqa: E402
from features.groups import FEATURE_ORDER                      # noqa: E402
from models.encoder import TS2VecEncoder                       # noqa: E402
from models.proto import (AddressabilityHead, AuroraModel,     # noqa: E402
                          PrototypicalHead)

FORECAST_KEYS = ("forecast", "future", "horizon", "predicted_cpu",
                 "predicted_mem", "pred_cpu", "pred_mem")


class TestNoForecastOutputs(unittest.TestCase):
    def test_decide_emits_no_future_values(self):
        m = AuroraModel(TS2VecEncoder(len(FEATURE_ORDER), d_model=8),
                        PrototypicalHead(8), AddressabilityHead(8))
        out = m.decide(torch.randn(2, 10, len(FEATURE_ORDER)),
                       torch.ones(2, 10, len(FEATURE_ORDER)))
        for o in out:
            for k in o:
                for bad in FORECAST_KEYS:
                    self.assertNotIn(bad, k.lower(),
                                     f"model output {k!r} looks like a forecast")

    def test_representations_are_not_named_as_predictions(self):
        m = AuroraModel(TS2VecEncoder(len(FEATURE_ORDER), d_model=8),
                        PrototypicalHead(8), AddressabilityHead(8))
        o = m.decide(torch.randn(1, 10, len(FEATURE_ORDER)))[0]
        self.assertIn("representation", o)
        self.assertIn("bottleneck", o)
        self.assertIn("addressable_prob", o)


class TestNoForecastInputs(unittest.TestCase):
    def test_decision_engine_takes_current_load_only(self):
        import inspect
        sig = inspect.signature(DecisionEngine.decide)
        params = set(sig.parameters)
        for banned in ("future_offered", "forecast_offered", "horizon",
                       "predicted_load"):
            self.assertNotIn(banned, params)
        # the offered workload must be a required, current value
        self.assertIn("offered", params)
        d = DecisionEngine(Config()).decide(
            "s", offered=10.0, capacity=10.0, sigma=0.1, n_committed=1,
            addressable_prob=1.0, bottleneck="cpu", confidence=0.9, ood=False)
        self.assertNotIn("forecast", d.basis)
        self.assertIn("p_serves_current", d.basis)

    def test_basis_has_no_future_field(self):
        d = DecisionEngine(Config()).decide(
            "s", offered=10.0, capacity=10.0, sigma=0.1, n_committed=1,
            addressable_prob=1.0, bottleneck="cpu", confidence=0.9, ood=False)
        for k in d.basis:
            for bad in FORECAST_KEYS:
                self.assertNotIn(bad, k.lower(), f"basis key {k!r}")


class TestNoForecastInModules(unittest.TestCase):
    """A coarse structural guard: no module may define a forecasting head."""

    def test_no_forecast_module(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        banned_files = {"forecast.py", "forecaster.py", "predictor.py"}
        found = []
        for dirpath, _, names in os.walk(root):
            if ".venv" in dirpath or "__pycache__" in dirpath:
                continue
            for n in names:
                if n in banned_files:
                    found.append(os.path.join(dirpath, n))
        self.assertEqual(found, [], f"forecasting module(s) present: {found}")


if __name__ == "__main__":
    unittest.main()

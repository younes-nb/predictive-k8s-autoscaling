#!/usr/bin/env python

import numpy as np
import torch
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

try:
    from tsbootstrap import EnbPIEnsemble, ACI, AgACI, NexCP
    from tsbootstrap.methods import MovingBlock, StationaryBlock
    TSBOOTSTRAP_AVAILABLE = True
except ImportError:
    TSBOOTSTRAP_AVAILABLE = False
    warnings.warn("tsbootstrap not available. Install: pip install 'tsbootstrap[models,accel,uq]'")


class ConformalCalibrator:

    CALIBRATORS = {
        "enbpi": "EnbPI",
        "aci": "ACI",
        "agaci": "AgACI", 
        "nexcp": "NexCP",
    }

    def __init__(
        self,
        calibrator_type: str = "aci",
        alpha: float = 0.1,
        n_bootstraps: int = 999,
        block_length: Union[str, int] = "auto",
        learning_rate: float = 0.1,
        horizon: int = 5,
        num_targets: int = 2,
    ):
        if not TSBOOTSTRAP_AVAILABLE:
            raise RuntimeError("tsbootstrap required: pip install 'tsbootstrap[models,accel,uq]'")

        self.calibrator_type = calibrator_type.lower()
        self.alpha = alpha
        self.n_bootstraps = n_bootstraps
        self.block_length = block_length
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.num_targets = num_targets

        self.calibrators = {}
        self.fitted = False

    def fit(self, residuals: np.ndarray, predictions: np.ndarray) -> "ConformalCalibrator":
        N, H, T = residuals.shape
        assert T == self.num_targets

        self.calibrators = {}

        for t_idx in range(self.num_targets):
            target_residuals = residuals[:, :, t_idx]
            target_predictions = predictions[:, :, t_idx]

            horizon_calibrators = []
            for h in range(self.horizon):
                res_h = target_residuals[:, h]
                pred_h = target_predictions[:, h]

                mask = ~(np.isnan(res_h) | np.isnan(pred_h))
                if mask.sum() < 50:
                    warnings.warn(f"Insufficient data for target {t_idx}, horizon {h}")
                    horizon_calibrators.append(None)
                    continue

                res_h = res_h[mask]
                pred_h = pred_h[mask]

                if self.calibrator_type == "aci":
                    cal = ACI(learning_rate=self.learning_rate)
                    cal.fit(pred_h, pred_h + res_h, alpha=self.alpha)
                elif self.calibrator_type == "agaci":
                    cal = AgACI(learning_rates=[self.learning_rate * 0.1, self.learning_rate, self.learning_rate * 10])
                    cal.fit(pred_h, pred_h + res_h, alpha=self.alpha)
                elif self.calibrator_type == "nexcp":
                    cal = NexCP()
                    cal.fit(pred_h, pred_h + res_h, alpha=self.alpha)
                elif self.calibrator_type == "enbpi":
                    cal = EnbPIEnsemble(
                        n_bootstraps=self.n_bootstraps,
                        alpha=self.alpha,
                        block_length=self.block_length,
                        calibrator=ACI(learning_rate=self.learning_rate)
                    )
                    cal.fit(pred_h, pred_h + res_h, alpha=self.alpha)
                else:
                    raise ValueError(f"Unknown calibrator: {self.calibrator_type}")

                horizon_calibrators.append(cal)

            self.calibrators[t_idx] = horizon_calibrators

        self.fitted = True
        return self

    def predict_intervals(
        self, 
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        N, H, T = predictions.shape
        lower = np.zeros_like(predictions)
        upper = np.zeros_like(predictions)

        for t_idx in range(self.num_targets):
            horizon_calibrators = self.calibrators.get(t_idx, [])
            for h in range(self.horizon):
                cal = horizon_calibrators[h]
                if cal is None:
                    lower[:, h, t_idx] = predictions[:, h, t_idx]
                    upper[:, h, t_idx] = predictions[:, h, t_idx]
                    continue

                pred_h = predictions[:, h, t_idx]

                if hasattr(cal, 'predict'):
                    lower[:, h, t_idx], upper[:, h, t_idx] = cal.predict(pred_h, alpha=self.alpha)
                else:
                    lower[:, h, t_idx] = pred_h
                    upper[:, h, t_idx] = pred_h

        return lower, upper

    def get_upper_bound(self, predictions: np.ndarray) -> np.ndarray:
        _, upper = self.predict_intervals(predictions)
        return upper

    def save(self, path: str):
        state = {
            "calibrator_type": self.calibrator_type,
            "alpha": self.alpha,
            "n_bootstraps": self.n_bootstraps,
            "block_length": self.block_length,
            "learning_rate": self.learning_rate,
            "horizon": self.horizon,
            "num_targets": self.num_targets,
            "calibrators": self.calibrators,
            "fitted": self.fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        cal = cls(
            calibrator_type=state["calibrator_type"],
            alpha=state["alpha"],
            n_bootstraps=state["n_bootstraps"],
            block_length=state["block_length"],
            learning_rate=state["learning_rate"],
            horizon=state["horizon"],
            num_targets=state["num_targets"],
        )
        cal.calibrators = state["calibrators"]
        cal.fitted = state["fitted"]
        return cal


class EnbPIPipeline:

    def __init__(
        self,
        quantile_model,
        calibrator_type: str = "aci",
        alpha: float = 0.1,
        horizon: int = 5,
        num_targets: int = 2,
    ):
        self.quantile_model = quantile_model
        self.calibrator = ConformalCalibrator(
            calibrator_type=calibrator_type,
            alpha=alpha,
            horizon=horizon,
            num_targets=num_targets,
        )
        self.device = next(quantile_model.parameters()).device

    def calibrate(self, val_loader) -> "EnbPIPipeline":
        self.quantile_model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(self.device).float()
                preds = self.quantile_model(x)
                median_idx = len(self.quantile_model.quantiles) // 2
                median_pred = preds[:, :, :, median_idx].cpu().numpy()
                all_preds.append(median_pred)
                all_targets.append(y.numpy())

        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)

        residuals = targets_arr - preds_arr

        self.calibrator.fit(residuals, preds_arr)
        return self

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.quantile_model.eval()
        with torch.no_grad():
            preds = self.quantile_model(x.to(self.device))
            median_idx = len(self.quantile_model.quantiles) // 2
            median = preds[:, :, :, median_idx].cpu().numpy()

        lower, upper = self.calibrator.predict_intervals(median)
        return median, lower, upper

    def get_upper_for_scaling(self, x: torch.Tensor) -> np.ndarray:
        self.quantile_model.eval()
        with torch.no_grad():
            preds = self.quantile_model(x.to(self.device))
            median_idx = len(self.quantile_model.quantiles) // 2
            median = preds[:, :, :, median_idx].cpu().numpy()
        return self.calibrator.get_upper_bound(median)


def make_enbpi_pipeline(
    quantile_model,
    calibrator_type: str = "aci",
    alpha: float = 0.1,
    horizon: int = 5,
    num_targets: int = 2,
) -> EnbPIPipeline:
    return EnbPIPipeline(
        quantile_model=quantile_model,
        calibrator_type=calibrator_type,
        alpha=alpha,
        horizon=horizon,
        num_targets=num_targets,
    )

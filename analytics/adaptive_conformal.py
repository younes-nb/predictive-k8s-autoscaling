from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AdaptiveConformalState:
    alpha_upper: float
    alpha_lower: float
    target_alpha: float
    window_size: int
    scores_upper: list
    scores_lower: list
    eta: float
    alpha_min: float
    alpha_max: float


class AdaptiveUpperConformal:

    def __init__(
        self,
        window_size: int = 500,
        alpha: float = 0.05,
        eta: float = 0.01,
        alpha_min: float = 0.01,
        alpha_max: float = 0.20,
    ):
        self.window_size = window_size
        self.target_alpha = alpha
        self.eta = eta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.scores_upper = deque(maxlen=window_size)
        self.scores_lower = deque(maxlen=window_size)

        self.alpha_upper = alpha
        self.alpha_lower = alpha

    def get_correction_upper(self, q95: float) -> float:
        if not self.scores_upper:
            return 0.0
        scores = np.asarray(self.scores_upper)
        q_conf = np.quantile(scores, 1.0 - self.alpha_upper, method="higher")
        return float(q95 + q_conf)

    def get_correction_lower(self, q10: float) -> float:
        if not self.scores_lower:
            return 0.0
        scores = np.asarray(self.scores_lower)
        q_conf = np.quantile(scores, 1.0 - self.alpha_lower, method="higher")
        return float(q10 - q_conf)

    def get_interval(self, q10: float, q95: float) -> Tuple[float, float]:
        lower = self.get_correction_lower(q10)
        upper = self.get_correction_upper(q95)
        return lower, upper

    def update(self, y: float, q10: float, q95: float) -> None:
        score_upper = max(0.0, y - q95)
        score_lower = max(0.0, q10 - y)

        upper_bound = self.get_correction_upper(q95)
        lower_bound = self.get_correction_lower(q10)

        upper_miss = float(y > upper_bound)
        lower_miss = float(y < lower_bound)

        self.scores_upper.append(score_upper)
        self.scores_lower.append(score_lower)

        self.alpha_upper = np.clip(
            self.alpha_upper + self.eta * (self.target_alpha - upper_miss),
            self.alpha_min, self.alpha_max
        )
        self.alpha_lower = np.clip(
            self.alpha_lower + self.eta * (self.target_alpha - lower_miss),
            self.alpha_min, self.alpha_max
        )

    def get_state(self) -> AdaptiveConformalState:
        return AdaptiveConformalState(
            alpha_upper=self.alpha_upper,
            alpha_lower=self.alpha_lower,
            target_alpha=self.target_alpha,
            window_size=self.window_size,
            scores_upper=list(self.scores_upper),
            scores_lower=list(self.scores_lower),
            eta=self.eta,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )

    def reset(self):
        self.scores_upper.clear()
        self.scores_lower.clear()
        self.alpha_upper = self.target_alpha
        self.alpha_lower = self.target_alpha


class AdaptiveUpperConformalPerTarget:

    def __init__(self, num_targets: int, **kwargs):
        self.calibrators = [
            AdaptiveUpperConformal(**kwargs) for _ in range(num_targets)
        ]
        self.target_names = ["cpu", "memory"][:num_targets]

    @property
    def states(self):
        return {name: self.calibrators[i] for i, name in enumerate(self.target_names)}

    def get_interval(self, q10: np.ndarray, q95: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q10 = np.asarray(q10)
        q95 = np.asarray(q95)

        if q10.ndim == 1:
            lower = np.zeros_like(q10)
            upper = np.zeros_like(q95)
            for t_idx, cal in enumerate(self.calibrators):
                lower[t_idx] = np.clip(cal.get_correction_lower(q10[t_idx]), 0.0, 1.0)
                upper[t_idx] = np.clip(cal.get_correction_upper(q95[t_idx]), 0.0, 1.0)
        else:
            lower = np.zeros_like(q10)
            upper = np.zeros_like(q95)
            for t_idx, cal in enumerate(self.calibrators):
                lower[:, t_idx] = [np.clip(cal.get_correction_lower(q10[h, t_idx]), 0.0, 1.0)
                                   for h in range(q10.shape[0])]
                upper[:, t_idx] = [np.clip(cal.get_correction_upper(q95[h, t_idx]), 0.0, 1.0)
                                   for h in range(q95.shape[0])]
        return lower, upper

    def update(self, y: np.ndarray, q10: np.ndarray, q95: np.ndarray) -> None:
        for t_idx, cal in enumerate(self.calibrators):
            cal.update(float(y[t_idx]), float(q10[t_idx]), float(q95[t_idx]))

    def get_alphas(self) -> Tuple[np.ndarray, np.ndarray]:
        alpha_u = np.array([c.alpha_upper for c in self.calibrators])
        alpha_l = np.array([c.alpha_lower for c in self.calibrators])
        return alpha_u, alpha_l

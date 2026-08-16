"""Online Adaptive Conformal Prediction for asymmetric errors.

Implements separate upper/lower adaptive conformal calibration as per the spec:
- Upper tail prioritized for CPU spike protection
- Rolling calibration window (W=500)
- ACI-style alpha adaptation
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AdaptiveConformalState:
    """State for adaptive conformal prediction."""
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
    """Separate upper/lower adaptive conformal for asymmetric errors.
    
    Maintains rolling buffers of conformity scores for upper and lower tails,
    updates alpha parameters online using ACI-style adaptation.
    """
    
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
        
        # Separate buffers for upper and lower conformity scores
        self.scores_upper = deque(maxlen=window_size)
        self.scores_lower = deque(maxlen=window_size)
        
        # Current alpha values
        self.alpha_upper = alpha
        self.alpha_lower = alpha
    
    def get_correction_upper(self, q95: float) -> float:
        """Get conformal correction for upper bound: q95 + Q_{1-alpha}(scores_upper)."""
        if not self.scores_upper:
            return 0.0
        scores = np.asarray(self.scores_upper)
        q_conf = np.quantile(scores, 1.0 - self.alpha_upper, method="higher")
        return float(q95 + q_conf)
    
    def get_correction_lower(self, q10: float) -> float:
        """Get conformal correction for lower bound: q10 - Q_{1-alpha}(scores_lower)."""
        if not self.scores_lower:
            return 0.0
        scores = np.asarray(self.scores_lower)
        q_conf = np.quantile(scores, 1.0 - self.alpha_lower, method="higher")
        return float(q10 - q_conf)
    
    def get_interval(self, q10: float, q95: float) -> Tuple[float, float]:
        """Get calibrated prediction interval [lower, upper]."""
        lower = self.get_correction_lower(q10)
        upper = self.get_correction_upper(q95)
        return lower, upper
    
    def update(self, y: float, q10: float, q95: float) -> None:
        """Update buffers and adapt alphas.
        
        Order is critical: compute scores from PREVIOUS state, then add to buffers,
        then update alphas based on whether the interval missed.
        """
        # Compute scores BEFORE adding to buffer (no label leakage)
        score_upper = max(0.0, y - q95)
        score_lower = max(0.0, q10 - y)
        
        # Current corrections (from old buffer)
        upper_bound = self.get_correction_upper(q95)
        lower_bound = self.get_correction_lower(q10)
        
        # Check for misses
        upper_miss = float(y > upper_bound)
        lower_miss = float(y < lower_bound)
        
        # Add scores to buffers
        self.scores_upper.append(score_upper)
        self.scores_lower.append(score_lower)
        
        # Update alphas: if miss -> decrease alpha (wider interval)
        self.alpha_upper = np.clip(
            self.alpha_upper + self.eta * (self.target_alpha - upper_miss),
            self.alpha_min, self.alpha_max
        )
        self.alpha_lower = np.clip(
            self.alpha_lower + self.eta * (self.target_alpha - lower_miss),
            self.alpha_min, self.alpha_max
        )
    
    def get_state(self) -> AdaptiveConformalState:
        """Get current state for logging/checkpointing."""
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
        """Reset buffers and alphas."""
        self.scores_upper.clear()
        self.scores_lower.clear()
        self.alpha_upper = self.target_alpha
        self.alpha_lower = self.target_alpha


class AdaptiveUpperConformalPerTarget:
    """Adaptive conformal for multiple targets (CPU, Memory)."""
    
    def __init__(self, num_targets: int, **kwargs):
        self.calibrators = [
            AdaptiveUpperConformal(**kwargs) for _ in range(num_targets)
        ]
        self.target_names = ["cpu", "memory"][:num_targets]
    
    @property
    def states(self):
        """Dict-like access to per-target states (for compatibility with deploy code)."""
        return {name: self.calibrators[i] for i, name in enumerate(self.target_names)}
    
    def get_interval(self, q10: np.ndarray, q95: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibrated intervals for all targets.
        
        Args:
            q10: (num_targets,) or (H, num_targets) lower quantiles
            q95: (num_targets,) or (H, num_targets) upper quantiles
        Returns:
            lower, upper: same shape as inputs
        """
        q10 = np.asarray(q10)
        q95 = np.asarray(q95)
        
        if q10.ndim == 1:
            lower = np.zeros_like(q10)
            upper = np.zeros_like(q95)
            for t_idx, cal in enumerate(self.calibrators):
                lower[t_idx] = cal.get_correction_lower(q10[t_idx])
                upper[t_idx] = cal.get_correction_upper(q95[t_idx])
        else:
            # Horizon dimension present
            lower = np.zeros_like(q10)
            upper = np.zeros_like(q95)
            for t_idx, cal in enumerate(self.calibrators):
                lower[:, t_idx] = [cal.get_correction_lower(q10[h, t_idx]) for h in range(q10.shape[0])]
                upper[:, t_idx] = [cal.get_correction_upper(q95[h, t_idx]) for h in range(q95.shape[0])]
        return lower, upper
    
    def update(self, y: np.ndarray, q10: np.ndarray, q95: np.ndarray) -> None:
        """Update all target calibrators."""
        for t_idx, cal in enumerate(self.calibrators):
            cal.update(float(y[t_idx]), float(q10[t_idx]), float(q95[t_idx]))
    
    def get_alphas(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current alpha values for logging."""
        alpha_u = np.array([c.alpha_upper for c in self.calibrators])
        alpha_l = np.array([c.alpha_lower for c in self.calibrators])
        return alpha_u, alpha_l
"""Adaptive Conformal State for online prediction interval calibration.

Replaces AR-based residual correction with conformal prediction intervals.
Maintains rolling buffers of conformity scores and adapts alpha parameters online.
"""

import json
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple


class AdaptiveConformalState:
    """Per-target conformal state with rolling buffers and adaptive alpha.
    
    Maintains separate upper/lower conformity score buffers and adapts
    alpha parameters using ACI-style online updates.
    """
    
    def __init__(
        self,
        window_size: int = 500,
        target_alpha: float = 0.05,
        eta: float = 0.01,
        alpha_min: float = 0.01,
        alpha_max: float = 0.20,
    ):
        self.window_size = window_size
        self.target_alpha = target_alpha
        self.eta = eta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Rolling buffers for conformity scores
        self.scores_upper = deque(maxlen=window_size)
        self.scores_lower = deque(maxlen=window_size)
        
        # Current alpha values
        self.alpha_upper = target_alpha
        self.alpha_lower = target_alpha
    
    def get_interval(self, q10: float, q95: float) -> Tuple[float, float]:
        """Get calibrated prediction interval [L, U].
        
        L = q10 - Q_{1-alpha_lower}(scores_lower)
        U = q95 + Q_{1-alpha_upper}(scores_upper)
        """
        if not self.scores_upper:
            return float(q10), float(q95)
        
        scores_u = np.asarray(self.scores_upper)
        scores_l = np.asarray(self.scores_lower)
        
        q_conf_u = float(np.quantile(scores_u, 1.0 - self.alpha_upper, method="higher"))
        q_conf_l = float(np.quantile(scores_l, 1.0 - self.alpha_lower, method="higher"))
        
        L = float(q10 - q_conf_l)
        U = float(q95 + q_conf_u)
        return L, U
    
    def update(self, y: float, q10: float, q95: float) -> None:
        """Update conformal state with observed value.
        
        Computes conformity scores BEFORE adding to buffer (no leakage).
        Then updates rolling buffers and adapts alphas based on misses.
        """
        # Compute scores BEFORE adding to buffer (no label leakage)
        score_u = max(0.0, y - q95)
        score_l = max(0.0, q10 - y)
        
        # Current interval (before adding current observation)
        L, U = self.get_interval(q10, q95)
        upper_miss = float(y > U)
        lower_miss = float(y < L)
        
        # Add scores to buffers
        self.scores_upper.append(score_u)
        self.scores_lower.append(score_l)
        
        # Update alphas: if miss -> decrease alpha (wider interval)
        self.alpha_upper = np.clip(
            self.alpha_upper + self.eta * (self.target_alpha - upper_miss),
            self.alpha_min, self.alpha_max
        )
        self.alpha_lower = np.clip(
            self.alpha_lower + self.eta * (self.target_alpha - lower_miss),
            self.alpha_min, self.alpha_max
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "window_size": self.window_size,
            "target_alpha": self.target_alpha,
            "eta": self.eta,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "scores_upper": list(self.scores_upper),
            "scores_lower": list(self.scores_lower),
            "alpha_upper": self.alpha_upper,
            "alpha_lower": self.alpha_lower,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptiveConformalState":
        """Deserialize state from dict."""
        state = cls(
            window_size=d["window_size"],
            target_alpha=d["target_alpha"],
            eta=d["eta"],
            alpha_min=d["alpha_min"],
            alpha_max=d["alpha_max"],
        )
        state.scores_upper = deque(d["scores_upper"], maxlen=d["window_size"])
        state.scores_lower = deque(d["scores_lower"], maxlen=d["window_size"])
        state.alpha_upper = d["alpha_upper"]
        state.alpha_lower = d["alpha_lower"]
        return state
    
    def reset(self) -> None:
        """Reset buffers and alphas to initial values."""
        self.scores_upper.clear()
        self.scores_lower.clear()
        self.alpha_upper = self.target_alpha
        self.alpha_lower = self.target_alpha


class ConformalManager:
    """Manages conformal state for multiple targets (CPU, Memory)."""
    
    def __init__(
        self,
        num_targets: int = 2,
        window_size: int = 500,
        target_alpha: float = 0.05,
        eta: float = 0.01,
        alpha_min: float = 0.01,
        alpha_max: float = 0.20,
    ):
        self.target_names = ["cpu", "memory"][:num_targets]
        self.states = {
            name: AdaptiveConformalState(
                window_size=window_size,
                target_alpha=target_alpha,
                eta=eta,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
            for name in self.target_names
        }
    
    def get_interval(self, q10: np.ndarray, q95: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibrated intervals for all targets.
        
        Args:
            q10: (num_targets,) lower quantiles
            q95: (num_targets,) upper quantiles
        Returns:
            L, U: (num_targets,) calibrated bounds
        """
        L = np.zeros_like(q10)
        U = np.zeros_like(q95)
        for i, name in enumerate(self.target_names):
            L[i], U[i] = self.states[name].get_interval(float(q10[i]), float(q95[i]))
        return L, U
    
    def update(self, y: np.ndarray, q10: np.ndarray, q95: np.ndarray) -> None:
        """Update all target states with observed values."""
        for i, name in enumerate(self.target_names):
            self.states[name].update(float(y[i]), float(q10[i]), float(q95[i]))
    
    def get_alphas(self) -> Dict[str, Tuple[float, float]]:
        """Get current alpha values for logging."""
        return {name: (s.alpha_upper, s.alpha_lower) for name, s in self.states.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all states."""
        return {
            "targets": {
                name: state.to_dict() for name, state in self.states.items()
            }
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConformalManager":
        """Deserialize all states."""
        targets = d.get("targets", {})
        if not targets:
            return cls()
        
        # Get config from first target
        first_state = next(iter(targets.values()))
        manager = cls(
            num_targets=len(targets),
            window_size=first_state["window_size"],
            target_alpha=first_state["target_alpha"],
            eta=first_state["eta"],
            alpha_min=first_state["alpha_min"],
            alpha_max=first_state["alpha_max"],
        )
        for name, state_dict in targets.items():
            if name in manager.states:
                manager.states[name] = AdaptiveConformalState.from_dict(state_dict)
        return manager
    
    def save_to_file(self, filepath: str) -> None:
        """Save conformal state to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "ConformalManager":
        """Load conformal state from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))
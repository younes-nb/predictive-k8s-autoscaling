"""Spike-aware evaluation metrics for conformal prediction.

Computes spike coverage, spike miss rate, mean excess error, and average
upper bound during spikes as specified in the conformal prediction plan.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SpikeMetrics:
    """Spike-aware metrics for a single target."""
    spike_threshold: float
    n_spikes: int
    spike_coverage: float
    spike_miss_rate: float
    mean_excess_error: float
    avg_upper_during_spike: float
    avg_upper_during_normal: float
    picp: float  # Overall prediction interval coverage probability
    mpiw: float  # Mean prediction interval width
    
    def to_dict(self, prefix: str = "") -> dict:
        """Convert to dictionary with optional prefix."""
        return {
            f"{prefix}spike_threshold": self.spike_threshold,
            f"{prefix}n_spikes": self.n_spikes,
            f"{prefix}spike_coverage": self.spike_coverage,
            f"{prefix}spike_miss_rate": self.spike_miss_rate,
            f"{prefix}mean_excess_error": self.mean_excess_error,
            f"{prefix}avg_upper_during_spike": self.avg_upper_during_spike,
            f"{prefix}avg_upper_during_normal": self.avg_upper_during_normal,
            f"{prefix}picp": self.picp,
            f"{prefix}mpiw": self.mpiw,
        }


def compute_spike_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spike_threshold: Optional[float] = None,
    spike_quantile: float = 0.95,
) -> SpikeMetrics:
    """Compute spike-aware metrics.
    
    Args:
        y_true: True values (N,)
        lower: Lower bounds of prediction interval (N,)
        upper: Upper bounds of prediction interval (N,)
        spike_threshold: Fixed threshold for spike definition. If None, uses spike_quantile of y_true.
        spike_quantile: Quantile of y_true to use as spike threshold (default 0.95).
    
    Returns:
        SpikeMetrics object with all spike-aware metrics.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    # Determine spike threshold
    if spike_threshold is None:
        spike_threshold = float(np.quantile(y_true, spike_quantile))
    
    # Identify spikes
    is_spike = y_true > spike_threshold
    n_spikes = int(is_spike.sum())
    n_normal = int((~is_spike).sum())
    
    # Overall PICP and MPIW
    in_interval = (y_true >= lower) & (y_true <= upper)
    picp = float(in_interval.mean())
    mpiw = float((upper - lower).mean())
    
    if n_spikes == 0:
        return SpikeMetrics(
            spike_threshold=spike_threshold,
            n_spikes=0,
            spike_coverage=1.0,
            spike_miss_rate=0.0,
            mean_excess_error=0.0,
            avg_upper_during_spike=0.0,
            avg_upper_during_normal=float(upper.mean()),
            picp=picp,
            mpiw=mpiw,
        )
    
    # Spike-specific metrics
    spike_covered = upper[is_spike] >= y_true[is_spike]
    spike_coverage = float(spike_covered.mean())
    spike_miss_rate = 1.0 - spike_coverage
    
    # Mean excess error: how far above upper bound during spike misses
    excess = np.maximum(0, y_true[is_spike] - upper[is_spike])
    mean_excess_error = float(excess.mean())
    
    # Average upper bound during spikes vs normal
    avg_upper_during_spike = float(upper[is_spike].mean())
    avg_upper_during_normal = float(upper[~is_spike].mean()) if n_normal > 0 else 0.0
    
    return SpikeMetrics(
        spike_threshold=spike_threshold,
        n_spikes=n_spikes,
        spike_coverage=spike_coverage,
        spike_miss_rate=spike_miss_rate,
        mean_excess_error=mean_excess_error,
        avg_upper_during_spike=avg_upper_during_spike,
        avg_upper_during_normal=avg_upper_during_normal,
        picp=picp,
        mpiw=mpiw,
    )


def compute_spike_metrics_per_target(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spike_quantile: float = 0.95,
) -> dict:
    """Compute spike metrics for multiple targets.
    
    Args:
        y_true: (N, T) true values
        lower: (N, T) lower bounds
        upper: (N, T) upper bounds
    
    Returns:
        Dictionary with per-target metrics.
    """
    results = {}
    for t_idx in range(y_true.shape[1]):
        metrics = compute_spike_metrics(
            y_true[:, t_idx],
            lower[:, t_idx],
            upper[:, t_idx],
            spike_quantile=spike_quantile,
        )
        target_name = f"target_{t_idx}"
        results[target_name] = metrics.to_dict(prefix=f"{target_name}_")
    return results


def print_spike_metrics(metrics: SpikeMetrics, target_name: str = "") -> None:
    """Pretty print spike metrics."""
    prefix = f"{target_name} " if target_name else ""
    print(f"{prefix}Spike Threshold (q={metrics.spike_threshold:.4f}):")
    print(f"  Spike Coverage:     {metrics.spike_coverage:.2%}  (n_spikes={metrics.n_spikes})")
    print(f"  Spike Miss Rate:    {metrics.spike_miss_rate:.2%}")
    print(f"  Mean Excess Error:  {metrics.mean_excess_error:.4f}")
    print(f"  Avg Upper (spike):  {metrics.avg_upper_during_spike:.4f}")
    print(f"  Avg Upper (normal): {metrics.avg_upper_during_normal:.4f}")
    print(f"  Overall PICP:       {metrics.picp:.2%}")
    print(f"  Overall MPIW:       {metrics.mpiw:.4f}")


def compute_coverage_trajectory(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    window: int = 100,
) -> np.ndarray:
    """Compute rolling coverage for trajectory analysis.
    
    Args:
        y_true: (N,) true values
        lower: (N,) lower bounds
        upper: (N,) upper bounds
        window: Rolling window size
    
    Returns:
        Array of rolling coverage values (N-window+1,)
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    rolling = np.convolve(in_interval.astype(float), np.ones(window)/window, mode='valid')
    return rolling


def compute_alpha_trajectory(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    window: int = 100,
) -> tuple:
    """Compute rolling upper/lower miss rates (proxy for adaptive alpha)."""
    upper_miss = (y_true > upper).astype(float)
    lower_miss = (y_true < lower).astype(float)
    
    upper_rolling = np.convolve(upper_miss, np.ones(window)/window, mode='valid')
    lower_rolling = np.convolve(lower_miss, np.ones(window)/window, mode='valid')
    
    return upper_rolling, lower_rolling
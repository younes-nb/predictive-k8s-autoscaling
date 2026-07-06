import logging
import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
from scipy.stats import norm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)


def _count_extrema(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        return 0
    diff = np.diff(x)
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    count = 0
    for c in crossings:
        left = diff[c]
        right = diff[c + 1] if c + 1 < len(diff) else 0.0
        if left > 0 and right < 0:
            count += 1
        elif left < 0 and right > 0:
            count += 1
    return count


def _extract_one_mode_via_vmd(
    signal: np.ndarray,
    alpha: int,
    tau: float,
    tol: float,
) -> np.ndarray:
    try:
        from vmdpy import VMD
    except ImportError as exc:
        raise RuntimeError(
            "vmdpy not installed. Run:\n"
            "  pip install vmdpy --break-system-packages"
        ) from exc

    signal = np.asarray(signal, dtype=np.float64)

    pad = len(signal) % 2
    if pad:
        signal_padded = np.append(signal, signal[-1])
    else:
        signal_padded = signal

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u, _, _ = VMD(signal_padded, alpha, tau, 1, 0, 1, tol)
        except Exception as exc:
            logger.warning("Single-mode VMD failed (%s); returning signal unchanged.", exc)
            return np.asarray(signal, dtype=np.float64)

    if pad:
        u = u[:, :-1]

    mode = np.asarray(u[:, :len(signal)], dtype=np.float64)
    return mode[0]


def svmd_decompose(
    signal: np.ndarray,
    max_modes: int,
    energy_ratio_tol: float,
    alpha: int,
    tau: float,
    tol: float,
) -> Tuple[List[np.ndarray], np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    residual = signal.copy()
    original_energy = np.sum(signal ** 2) + 1e-12
    modes = []

    for _ in range(max_modes):
        residual_energy = np.sum(residual ** 2)
        if residual_energy / original_energy < energy_ratio_tol:
            logger.debug("SVMD stopping: residual energy below tolerance.")
            break

        if _count_extrema(residual) < 2:
            logger.debug("SVMD stopping: fewer than 2 extrema in residual.")
            break

        mode = _extract_one_mode_via_vmd(residual, alpha, tau, tol)
        modes.append(mode)
        residual = residual - mode

    if not modes:
        modes = [signal.copy()]
        residual = np.zeros_like(signal)

    return modes, residual


def dispersion_entropy(
    x: np.ndarray,
    classes: int = 6,
    embed_dim: int = 2,
    time_delay: int = 1,
) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < embed_dim * time_delay + 1:
        return 0.0

    std_x = float(np.std(x, ddof=1))
    if std_x < 1e-12:
        return 0.0

    z = (x - np.mean(x)) / std_x
    y = norm.cdf(z)

    class_i = np.round(classes * y + 0.5).astype(np.int64)
    class_i = np.clip(class_i, 1, classes)

    n_patterns = n - (embed_dim - 1) * time_delay
    patterns = np.zeros(n_patterns, dtype=object)
    for i in range(n_patterns):
        idxs = [i + d * time_delay for d in range(embed_dim)]
        patterns[i] = tuple(class_i[idxs].tolist())

    _, counts = np.unique(patterns, return_counts=True)
    p = counts.astype(np.float64) / n_patterns
    de = -np.sum(p * np.log(p + 1e-12))

    max_de = np.log(classes ** embed_dim)
    if max_de < 1e-12:
        return 0.0
    de_normalized = de / max_de
    return float(np.clip(de_normalized, 0.0, 1.0))


def band_modes_by_dispersion_entropy(
    modes: List[np.ndarray],
    high_quantile: float,
    cfg,
) -> Tuple[List[np.ndarray], np.ndarray]:
    if not modes:
        return [], np.zeros(cfg.INPUT_LEN, dtype=np.float64)

    if len(modes) == 1:
        return [modes[0].copy()], np.zeros_like(modes[0])

    de_values = np.array([dispersion_entropy(m) for m in modes])
    threshold = float(np.quantile(de_values, high_quantile))

    high_freq_modes = []
    low_freq_list = []
    for i, m in enumerate(modes):
        if de_values[i] >= threshold:
            high_freq_modes.append(m.copy())
        else:
            low_freq_list.append(m.copy())

    if not high_freq_modes:
        high_freq_modes = [modes[-1].copy()]
        if low_freq_list:
            low_freq_list = [m for j, m in enumerate(modes) if j != len(modes) - 1]

    low_freq_signal = sum(low_freq_list) if low_freq_list else np.zeros_like(modes[0])

    return high_freq_modes, low_freq_signal


def decompose_window(window: np.ndarray, cfg) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    n = len(window)

    if n < cfg.INPUT_LEN or np.std(window) < 1e-12:
        logger.warning(
            "Degenerate window (std=%.2e); returning zeros.",
            float(np.std(window)),
        )
        return np.zeros((cfg.TOTAL_CHANNELS, cfg.INPUT_LEN), dtype=np.float32)

    primary_modes, primary_residual = svmd_decompose(
        window, cfg.SVMD_MAX_MODES, cfg.SVMD_ENERGY_RATIO_TOL,
        cfg.SVMD_ALPHA, cfg.SVMD_TAU, cfg.SVMD_TOL,
    )

    high_freq_modes, low_freq_signal = band_modes_by_dispersion_entropy(
        primary_modes, cfg.DE_HIGH_QUANTILE, cfg,
    )
    low_freq_signal = low_freq_signal + primary_residual

    combined_high_freq = sum(high_freq_modes)

    secondary_modes, secondary_residual = svmd_decompose(
        combined_high_freq, cfg.SVMD2_MAX_MODES, cfg.SVMD2_ENERGY_RATIO_TOL,
        cfg.SVMD2_ALPHA, cfg.SVMD2_TAU, cfg.SVMD2_TOL,
    )

    low_freq_signal = low_freq_signal + secondary_residual

    actual_count = len(secondary_modes)
    padded_secondary = []
    for i in range(cfg.SVMD2_MAX_MODES):
        if i < actual_count:
            padded_secondary.append(secondary_modes[i].copy())
        else:
            padded_secondary.append(np.zeros(cfg.INPUT_LEN, dtype=np.float64))

    recon = sum(padded_secondary) + low_freq_signal
    recon_error = window - recon

    channels = [m.astype(np.float32) for m in padded_secondary]
    channels.append(low_freq_signal.astype(np.float32))
    channels.append(recon_error.astype(np.float32))

    result = np.stack(channels, axis=0)
    assert result.shape == (cfg.TOTAL_CHANNELS, cfg.INPUT_LEN), (
        f"Expected ({cfg.TOTAL_CHANNELS}, {cfg.INPUT_LEN}), got {result.shape}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from experiments.sdtnet.config import CFG as CFG

    print("=" * 60)
    print("Scenario A — Decomposition smoke test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, CFG.INPUT_LEN)
    signal = np.sin(2 * np.pi * t / 15) + 0.5 * np.sin(2 * np.pi * t / 5)
    signal += 0.1 * rng.standard_normal(CFG.INPUT_LEN)

    result = decompose_window(signal, CFG)
    print(f"Output shape: {result.shape}")
    assert result.shape == (CFG.TOTAL_CHANNELS, CFG.INPUT_LEN), "Shape mismatch"

    reconstruct = result.sum(axis=0)
    rec_error = np.max(np.abs(reconstruct - signal.astype(np.float32)))
    print(f"Reconstruction max error: {rec_error:.6f}")
    assert np.allclose(reconstruct, signal.astype(np.float32), atol=1e-4), \
        f"Reconstruction error too high: {rec_error}"

    n_primary = [None]
    print("Dispersion entropy tests:")
    const_val = dispersion_entropy(np.ones(CFG.INPUT_LEN))
    print(f"  Constant input DE: {const_val:.4f} (expected ~0.0)")
    assert const_val == 0.0, "Constant signal DE should be 0"

    rand_val = dispersion_entropy(rng.standard_normal(CFG.INPUT_LEN))
    print(f"  Random input DE: {rand_val:.4f} (expected in [0,1])")
    assert 0.0 <= rand_val <= 1.0, f"DE out of range: {rand_val}"

    print("\nAll decomposition smoke checks passed!")

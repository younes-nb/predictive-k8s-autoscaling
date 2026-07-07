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
            u, _, omega = VMD(signal_padded, alpha, tau, 1, 0, 1, tol)
        except Exception as exc:
            logger.warning("Single-mode VMD failed (%s); returning signal unchanged.", exc)
            return np.asarray(signal, dtype=np.float64), 0.0

    if pad:
        u = u[:, :-1]

    mode = np.asarray(u[:, :len(signal)], dtype=np.float64)
    return mode[0], omega[0]


def svmd_decompose_adaptive(
    signal: np.ndarray,
    alpha: int,
    tau: float,
    tol: float,
    max_iter: int = 20,
) -> Tuple[List[np.ndarray], np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    residual = signal.copy()
    original_energy = np.sum(signal ** 2) + 1e-12
    modes = []
    center_freqs = []

    ENERGY_RATIO_TOL = 0.005
    FREQ_DUP_THRESHOLD = 0.01
    NOISE_FLOOR = 0.45

    for _ in range(max_iter):
        residual_energy = np.sum(residual ** 2)
        if residual_energy / original_energy < ENERGY_RATIO_TOL:
            logger.debug("Adaptive SVMD stopping: residual energy below tolerance.")
            break

        if _count_extrema(residual) < 2:
            logger.debug("Adaptive SVMD stopping: fewer than 2 extrema in residual.")
            break

        mode, omega_rad = _extract_one_mode_via_vmd(residual, alpha, tau, tol)
        omega_norm = float(np.asarray(omega_rad).item()) / np.pi

        if len(center_freqs) > 0:
            is_duplicate = any(
                abs(omega_norm - prev_omega) < FREQ_DUP_THRESHOLD
                for prev_omega in center_freqs
            )
            if is_duplicate:
                logger.debug("Adaptive SVMD stopping: duplicate center frequency (%.4f)", omega_norm)
                break

        if omega_norm > NOISE_FLOOR:
            logger.debug("Adaptive SVMD stopping: noise floor exceeded (%.4f > %.4f)",
                         omega_norm, NOISE_FLOOR)
            break

        modes.append(mode)
        center_freqs.append(omega_norm)
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


def otsu_threshold(values: np.ndarray) -> float:
    if len(values) <= 1:
        return float(values[0]) if len(values) == 1 else 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    best_threshold = float(sorted_vals[0])
    best_var = float('inf')

    for i in range(1, n):
        threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
        left = sorted_vals[:i]
        right = sorted_vals[i:]

        w1 = i / n
        w2 = (n - i) / n
        var1 = float(np.var(left, ddof=0))
        var2 = float(np.var(right, ddof=0))

        intra_var = w1 * var1 + w2 * var2

        if intra_var < best_var:
            best_var = intra_var
            best_threshold = threshold

    return best_threshold


def band_modes_by_otsu(
    modes: List[np.ndarray],
    cfg,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not modes:
        return [], []

    if len(modes) == 1:
        return [modes[0].copy()], []

    de_values = np.array([dispersion_entropy(m) for m in modes])
    threshold = otsu_threshold(de_values)

    high_freq_modes = []
    low_freq_modes = []
    for i, m in enumerate(modes):
        if de_values[i] >= threshold:
            high_freq_modes.append(m.copy())
        else:
            low_freq_modes.append(m.copy())

    if not high_freq_modes:
        high_freq_modes = [modes[-1].copy()]
        low_freq_modes = [m for j, m in enumerate(modes) if j != len(modes) - 1]

    if not low_freq_modes:
        low_freq_modes = [modes[-1].copy()]
        high_freq_modes = [m for j, m in enumerate(modes) if j != len(modes) - 1]

    return high_freq_modes, low_freq_modes


def decompose_window(window: np.ndarray, cfg) -> np.ndarray:
    import pywt

    window = np.asarray(window, dtype=np.float64)
    n = len(window)

    if n < cfg.INPUT_LEN or np.std(window) < 1e-12:
        logger.warning(
            "Degenerate window (std=%.2e); returning zeros.",
            float(np.std(window)),
        )
        return np.zeros((cfg.TOTAL_CHANNELS, cfg.INPUT_LEN), dtype=np.float32)

    primary_modes, primary_residual = svmd_decompose_adaptive(
        window, cfg.SVMD_ALPHA, cfg.SVMD_TAU, cfg.SVMD_TOL,
    )

    high_freq_modes, low_freq_modes = band_modes_by_otsu(
        primary_modes, cfg,
    )

    high_composite = sum(high_freq_modes) if high_freq_modes else np.zeros_like(window)
    low_composite = sum(low_freq_modes) + primary_residual if low_freq_modes else primary_residual

    pad_len = 64 - len(high_composite)
    if pad_len > 0:
        high_padded = np.pad(high_composite, (0, pad_len), mode='reflect')
    else:
        high_padded = high_composite
    mra_coeffs = pywt.mra(high_padded, 'sym4', level=3, transform='swt')
    mra_coeffs = [c[:len(high_composite)] for c in mra_coeffs]

    channels = [
        mra_coeffs[0].astype(np.float32),
        mra_coeffs[1].astype(np.float32),
        mra_coeffs[2].astype(np.float32),
        mra_coeffs[3].astype(np.float32),
        low_composite.astype(np.float32),
    ]

    result = np.stack(channels, axis=0)
    assert result.shape == (cfg.TOTAL_CHANNELS, cfg.INPUT_LEN), (
        f"Expected ({cfg.TOTAL_CHANNELS}, {cfg.INPUT_LEN}), got {result.shape}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from experiments.tsdp.config import CFG as CFG

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

    print("Dispersion entropy tests:")
    const_val = dispersion_entropy(np.ones(CFG.INPUT_LEN))
    print(f"  Constant input DE: {const_val:.4f} (expected ~0.0)")
    assert const_val == 0.0, "Constant signal DE should be 0"

    rand_val = dispersion_entropy(rng.standard_normal(CFG.INPUT_LEN))
    print(f"  Random input DE: {rand_val:.4f} (expected in [0,1])")
    assert 0.0 <= rand_val <= 1.0, f"DE out of range: {rand_val}"

    print("\nOtsu threshold test:")
    test_de = np.array([0.01, 0.02, 0.03, 0.40, 0.45, 0.50])
    thr = otsu_threshold(test_de)
    print(f"  Otsu threshold on {test_de}: {thr:.4f}")
    assert 0.03 < thr < 0.40, f"Otsu threshold should separate low and high: {thr}"

    print("\nAdaptive SVMD test:")
    amodes, ares = svmd_decompose_adaptive(signal, CFG.SVMD_ALPHA, CFG.SVMD_TAU, CFG.SVMD_TOL)
    print(f"  Extracted {len(amodes)} modes (adaptive, center-frequency progression)")
    assert len(amodes) >= 1, "Should extract at least 1 mode"

    print("\nAll decomposition smoke checks passed!")

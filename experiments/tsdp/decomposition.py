import logging
import os
import sys
import warnings

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)


def vmd_decompose(
    signal: np.ndarray,
    K: int,
    alpha: int,
    tau: float,
    DC: int,
    init: int,
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
            u, _, _ = VMD(signal_padded, alpha, tau, K, DC, init, tol)
        except Exception as exc:
            logger.warning("VMD failed (%s); returning signal as single mode.", exc)
            return signal[np.newaxis, :]

    if pad:
        u = u[:, :-1]

    return np.asarray(u[:, :len(signal)], dtype=np.float64)


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

    pad_len = 64 - n
    if pad_len > 0:
        padded = np.pad(window, (pad_len, 0), mode='edge')
    else:
        padded = window
    mra_coeffs = pywt.mra(padded, 'sym4', level=3, transform='swt')
    if pad_len > 0:
        mra_coeffs = [c[pad_len:] for c in mra_coeffs]

    D1, D2, D3, A3 = mra_coeffs

    vmd_modes = vmd_decompose(
        D1, K=cfg.VMD_K, alpha=cfg.VMD_ALPHA,
        tau=cfg.VMD_TAU, DC=cfg.VMD_DC,
        init=cfg.VMD_INIT, tol=cfg.VMD_TOL,
    )

    channels = []
    for k in range(vmd_modes.shape[0]):
        channels.append(vmd_modes[k].astype(np.float32))
    channels.append(D2.astype(np.float32))
    channels.append(D3.astype(np.float32))
    channels.append(A3.astype(np.float32))

    result = np.stack(channels, axis=0)
    assert result.shape == (cfg.TOTAL_CHANNELS, cfg.INPUT_LEN), (
        f"Expected ({cfg.TOTAL_CHANNELS}, {cfg.INPUT_LEN}), got {result.shape}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from experiments.tsdp.config import CFG as CFG

    print("=" * 60)
    print("Scenario A — MODWT-VMD Decomposition smoke test")
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

    print("\nVMD fixed-K decomposition test:")
    vmd_modes = vmd_decompose(signal, K=CFG.VMD_K, alpha=CFG.VMD_ALPHA,
                               tau=CFG.VMD_TAU, DC=CFG.VMD_DC,
                               init=CFG.VMD_INIT, tol=CFG.VMD_TOL)
    print(f"  VMD produced {vmd_modes.shape[0]} modes")
    assert vmd_modes.shape[0] == CFG.VMD_K, f"Expected {CFG.VMD_K} modes, got {vmd_modes.shape[0]}"

    print("\nAll decomposition smoke checks passed!")

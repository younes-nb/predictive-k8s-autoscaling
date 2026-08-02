import logging
import os
import sys
import warnings

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_preprocessing_defaults import PREPROCESSING

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


def _expected_channels(cfg) -> int:
    if cfg.NO_VMD:
        return cfg.SWT_LEVEL + 1
    return cfg.VMD_K + cfg.SWT_LEVEL


def decompose_window(window: np.ndarray, cfg) -> np.ndarray:
    import pywt

    window = np.asarray(window, dtype=np.float64)
    n = len(window)

    if n < 2 ** cfg.SWT_LEVEL or np.std(window) < 1e-12:
        return None

    swt_coeffs = pywt.swt(window, 'sym4', level=cfg.SWT_LEVEL, norm=True, trim_approx=True)
    A_ts = swt_coeffs[0]

    if cfg.NO_VMD:
        channels = [swt_coeffs[idx].astype(np.float32) for idx in range(len(swt_coeffs))]
    else:
        vmd_d_idx = cfg.SWT_LEVEL + 1 - cfg.VMD_SWT_LEVEL
        D_ts = swt_coeffs[vmd_d_idx]

        vmd_modes = vmd_decompose(
            D_ts, K=cfg.VMD_K, alpha=cfg.VMD_ALPHA,
            tau=cfg.VMD_TAU, DC=cfg.VMD_DC,
            init=cfg.VMD_INIT, tol=cfg.VMD_TOL,
        )

        channels = [vmd_modes[k].astype(np.float32) for k in range(vmd_modes.shape[0])]
        for idx in range(1, len(swt_coeffs)):
            if idx == vmd_d_idx:
                continue
            channels.append(swt_coeffs[idx].astype(np.float32))
        channels.append(A_ts.astype(np.float32))

    result = np.stack(channels, axis=0)
    expected_channels = _expected_channels(cfg)
    assert result.shape == (expected_channels, n), (
        f"Expected ({expected_channels}, {n}), got {result.shape}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from dataclasses import replace
    from preprocessing.sv.config import CFG as CFG

    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, PREPROCESSING.INPUT_LEN)
    signal = np.sin(2 * np.pi * t / 15) + 0.5 * np.sin(2 * np.pi * t / 5)
    signal += 0.1 * rng.standard_normal(PREPROCESSING.INPUT_LEN)

    print("=" * 60)
    print("Scenario A — SWT+VMD Decomposition smoke test (default D1)")
    print("=" * 60)
    result = decompose_window(signal, CFG)
    expected = CFG.VMD_K + CFG.SWT_LEVEL
    print(f"Output shape: {result.shape}")
    assert result.shape == (expected, PREPROCESSING.INPUT_LEN), "Shape mismatch"
    reconstruct = result.sum(axis=0)
    rec_error = np.max(np.abs(reconstruct - signal.astype(np.float32)))
    print(f"Reconstruction max error: {rec_error:.6f}")

    print("\nVMD fixed-K decomposition test:")
    vmd_modes = vmd_decompose(signal, K=CFG.VMD_K, alpha=CFG.VMD_ALPHA,
                               tau=CFG.VMD_TAU, DC=CFG.VMD_DC,
                               init=CFG.VMD_INIT, tol=CFG.VMD_TOL)
    print(f"  VMD produced {vmd_modes.shape[0]} modes")
    assert vmd_modes.shape[0] == CFG.VMD_K, f"Expected {CFG.VMD_K} modes, got {vmd_modes.shape[0]}"

    print("\n" + "=" * 60)
    print("Scenario B — SWT+VMD with VMD_SWT_LEVEL=2 (D2 into VMD)")
    print("=" * 60)
    cfg_d2 = replace(CFG, VMD_SWT_LEVEL=2)
    result_d2 = decompose_window(signal, cfg_d2)
    expected_d2 = cfg_d2.VMD_K + cfg_d2.SWT_LEVEL
    print(f"Output shape: {result_d2.shape}")
    assert result_d2.shape == (expected_d2, PREPROCESSING.INPUT_LEN), "Shape mismatch"

    print("\n" + "=" * 60)
    print("Scenario C — SWT only (no VMD)")
    print("=" * 60)
    cfg_no_vmd = replace(CFG, NO_VMD=True)
    result_no_vmd = decompose_window(signal, cfg_no_vmd)
    expected_no_vmd = cfg_no_vmd.SWT_LEVEL + 1
    print(f"Output shape: {result_no_vmd.shape}")
    assert result_no_vmd.shape == (expected_no_vmd, PREPROCESSING.INPUT_LEN), "Shape mismatch"
    rec_no_vmd = result_no_vmd.sum(axis=0)
    rec_err_no_vmd = np.max(np.abs(rec_no_vmd - signal.astype(np.float32)))
    print(f"SWT-only reconstruction max error: {rec_err_no_vmd:.6f}")

    print("\nAll decomposition smoke checks passed!")

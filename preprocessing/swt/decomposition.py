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


def _expected_channels(cfg) -> int:
    return cfg.SWT_LEVEL + 1


def decompose_window(window: np.ndarray, cfg) -> np.ndarray:
    import pywt

    window = np.asarray(window, dtype=np.float64)
    n = len(window)

    if n < 2 ** cfg.SWT_LEVEL or np.std(window) < 1e-12:
        return None

    swt_coeffs = pywt.swt(window, 'sym4', level=cfg.SWT_LEVEL, norm=True, trim_approx=True)

    channels = [swt_coeffs[idx].astype(np.float32) for idx in range(len(swt_coeffs))]

    result = np.stack(channels, axis=0)
    expected_channels = _expected_channels(cfg)
    assert result.shape == (expected_channels, n), (
        f"Expected ({expected_channels}, {n}), got {result.shape}"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from preprocessing.swt.config import CFG

    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, PREPROCESSING.INPUT_LEN)
    signal = np.sin(2 * np.pi * t / 15) + 0.5 * np.sin(2 * np.pi * t / 5)
    signal += 0.1 * rng.standard_normal(PREPROCESSING.INPUT_LEN)

    print("=" * 60)
    print("SWT Decomposition smoke test")
    print("=" * 60)
    result = decompose_window(signal, CFG)
    expected = CFG.SWT_LEVEL + 1
    print(f"Output shape: {result.shape}")
    assert result.shape == (expected, PREPROCESSING.INPUT_LEN), "Shape mismatch"
    reconstruct = result.sum(axis=0)
    rec_error = np.max(np.abs(reconstruct - signal.astype(np.float32)))
    print(f"Reconstruction max error: {rec_error:.6f}")

    print("\nAll decomposition smoke checks passed!")

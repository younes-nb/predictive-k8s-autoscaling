import logging
import warnings
from typing import List, Tuple

import numpy as np
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)


def ceemdan_decompose_window(
    window: np.ndarray,
    epsilon: float,
    trials: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    try:
        from PyEMD import CEEMDAN
    except ImportError as exc:
        raise RuntimeError(
            "PyEMD not installed. Run:\n"
            "  pip install EMD-signal --break-system-packages"
        ) from exc

    signal = np.asarray(window, dtype=np.float64)

    c = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    c.noise_seed(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            imfs_all = c.ceemdan(signal)
        except Exception as exc:
            logger.debug("CEEMDAN failed (%s); using raw signal as single IMF.", exc)
            return [signal.copy()], np.zeros_like(signal)

    if imfs_all.shape[0] < 2:
        logger.debug("CEEMDAN returned < 2 rows; degeneracy path.")
        return [signal.copy()], np.zeros_like(signal)

    residue = imfs_all[-1].copy()
    imfs = [imfs_all[i].copy() for i in range(imfs_all.shape[0] - 1)]
    return imfs, residue


def sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r_frac: float = 0.2,
    max_samples: int = 1000,
) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()

    if len(x) > max_samples:
        indices = np.linspace(0, len(x) - 1, max_samples).astype(int)
        x = x[indices]

    N = len(x)
    if N < m + 2:
        return float("nan")

    r = r_frac * float(np.std(x, ddof=1))
    if r == 0.0:
        return float("nan")

    def _count_matches(template_len: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(x, template_len)
        n_templates = len(templates)
        count = 0
        for i in range(n_templates):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            dists[i] = np.inf
            count += int(np.sum(dists < r))
        return count

    Bm = _count_matches(m)
    Am = _count_matches(m + 1)

    if Bm == 0 or Am == 0:
        return float("nan")

    return float(-np.log(Am / Bm))


def _rank_split_fallback(
    imfs: List[np.ndarray],
    se_values: np.ndarray,
    residue: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_imfs = len(imfs)
    if n_imfs == 0:
        return np.zeros_like(residue), residue.copy()

    order = np.argsort(-se_values)
    n_high = max(1, (n_imfs + 1) // 2)
    high_idx = order[:n_high]
    low_idx = order[n_high:]

    high_signal = sum(imfs[i] for i in high_idx)
    low_signal = sum(imfs[i] for i in low_idx) + residue
    return high_signal, low_signal


def hdbscan_cluster_imfs(
    imfs: List[np.ndarray],
    se_values: np.ndarray,
    residue: np.ndarray,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    n_imfs = len(imfs)
    if n_imfs == 0:
        return np.zeros_like(residue), residue.copy()

    se_array = np.asarray(se_values, dtype=np.float64).reshape(-1, 1)

    if n_imfs < 3:
        return _rank_split_fallback(imfs, se_values.ravel(), residue)

    clusterer = HDBSCAN(
        min_cluster_size=cfg.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=cfg.HDBSCAN_MIN_SAMPLES,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = clusterer.fit_predict(se_array)

    unique_labels = set(l for l in labels if l != -1)

    if len(unique_labels) < 2:
        logger.debug(
            "HDBSCAN found < 2 clusters; using SE-rank fallback."
        )
        return _rank_split_fallback(imfs, se_values.ravel(), residue)

    cluster_means = {}
    for label in unique_labels:
        mask = labels == label
        cluster_means[label] = float(se_values.ravel()[mask].mean())

    high_label = max(cluster_means, key=cluster_means.get)

    high_mask = labels == high_label
    low_mask = (labels != high_label) | (labels == -1)

    high_signal = sum(imfs[i] for i in range(n_imfs) if high_mask[i])
    low_signal = sum(imfs[i] for i in range(n_imfs) if low_mask[i]) + residue

    return high_signal, low_signal


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
            logger.warning("VMD failed (%s); tiling signal as K modes.", exc)
            return np.tile(signal, (K, 1))

    if pad:
        u = u[:, :-1]

    return np.asarray(u[:, :len(signal)], dtype=np.float64)


def decompose_window(
    window: np.ndarray,
    cfg,
) -> List[np.ndarray]:
    window = np.asarray(window, dtype=np.float64)
    n = len(window)

    if n < cfg.INPUT_LEN or np.std(window) < 1e-12:
        logger.warning(
            "Degenerate window (std=%.2e); returning zeros.",
            float(np.std(window)),
        )
        return [np.zeros(cfg.INPUT_LEN, dtype=np.float32) for _ in range(cfg.TOTAL_CHANNELS)]

    imfs, residue = ceemdan_decompose_window(
        window, cfg.CEEMDAN_EPSILON, cfg.CEEMDAN_TRIALS, seed=42,
    )

    se_values = []
    for imf in imfs:
        se = sample_entropy(imf, m=cfg.SE_M, r_frac=cfg.SE_R_FRAC, max_samples=cfg.SE_MAX_SAMPLES)
        se_values.append(se)

    se_array = np.array(se_values, dtype=np.float64)
    mask_finite = np.isfinite(se_array)
    if not np.any(mask_finite):
        se_array[:] = 0.5
    else:
        se_array[~mask_finite] = np.nanmedian(se_array)

    high_freq_signal, low_freq_signal = hdbscan_cluster_imfs(
        imfs, se_array, residue, cfg,
    )

    vmd_modes = vmd_decompose(
        high_freq_signal,
        K=cfg.VMD_K,
        alpha=cfg.VMD_ALPHA,
        tau=cfg.VMD_TAU,
        DC=cfg.VMD_DC,
        init=cfg.VMD_INIT,
        tol=cfg.VMD_TOL,
    )

    recon = vmd_modes.sum(axis=0) + low_freq_signal
    recon_error = window - recon
    low_freq_signal = low_freq_signal + recon_error

    channels = [vmd_modes[i].astype(np.float32) for i in range(cfg.VMD_K)]
    channels.append(low_freq_signal.astype(np.float32))

    assert len(channels) == cfg.TOTAL_CHANNELS, \
        f"Expected {cfg.TOTAL_CHANNELS} channels, got {len(channels)}"
    assert all(ch.shape == (cfg.INPUT_LEN,) for ch in channels), \
        f"Channel shape mismatch: expected ({cfg.INPUT_LEN},)"

    return channels

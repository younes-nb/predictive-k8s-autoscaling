
import logging
import warnings
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def ceemdan_decompose(
    signal: np.ndarray, epsilon: float, n_trials: int
) -> Tuple[np.ndarray, np.ndarray]:

    try:
        from PyEMD import CEEMDAN
    except ImportError as exc:
        raise RuntimeError(
            "PyEMD not installed. Run:\n"
            "  pip install EMD-signal --break-system-packages"
        ) from exc

    signal = np.asarray(signal, dtype=np.float64)
    c = CEEMDAN(trials=n_trials, epsilon=epsilon, parallel=False)
    c.noise_seed(42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            imfs_all = c.ceemdan(signal)
        except Exception as exc:
            logger.warning("CEEMDAN failed (%s); using raw signal as single IMF.", exc)
            return signal[np.newaxis, :].copy(), np.zeros_like(signal)

    if imfs_all.shape[0] < 2:

        return np.zeros((0, len(signal)), dtype=np.float64), imfs_all[-1].copy()

    residue = imfs_all[-1].copy()
    imfs = imfs_all[:-1].copy()
    return imfs, residue

def sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r_frac: float = 0.2,
    max_samples: int = 1000,
) -> float:

    x = np.asarray(x, dtype=np.float64).ravel()

    if len(x) > max_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(x), size=max_samples, replace=False))
        x = x[idx]

    N = len(x)
    if N < 2 * (m + 1) + 2:
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

def cluster_imfs(
    imfs: np.ndarray,
    residue: np.ndarray,
    m: int,
    r_frac: float,
    max_se_samples: int,
    n_clusters: int = 3,
) -> List[np.ndarray]:

    n_imfs = imfs.shape[0]

    if n_imfs == 0:

        zero = np.zeros_like(residue)
        return [zero.copy(), zero.copy(), residue.copy()]

    se_values = np.array([
        sample_entropy(imfs[i], m, r_frac, max_se_samples)
        for i in range(n_imfs)
    ])

    valid_mask = np.isfinite(se_values)
    if valid_mask.any():
        se_values[~valid_mask] = float(np.nanmedian(se_values))
    else:
        se_values[:] = 0.5

    k = min(n_clusters, n_imfs)
    if k == 1:
        labels = np.zeros(n_imfs, dtype=int)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            raw_labels = km.fit_predict(se_values.reshape(-1, 1))

        cluster_mean_se = np.full(k, -np.inf, dtype=np.float64)
        for c in range(k):
            mask = raw_labels == c
            if mask.any():
                cluster_mean_se[c] = se_values[mask].mean()

        sorted_order = np.argsort(-cluster_mean_se)
        remap = np.empty(k, dtype=int)
        for new_idx, old_idx in enumerate(sorted_order):
            remap[old_idx] = new_idx
        labels = remap[raw_labels]

    co_imfs: List[np.ndarray] = []
    for c in range(k):
        mask = labels == c
        if mask.any():
            co_imfs.append(imfs[mask].sum(axis=0))
        else:
            co_imfs.append(np.zeros_like(residue))

    while len(co_imfs) < n_clusters:
        co_imfs.append(np.zeros_like(residue))

    co_imfs[-1] = co_imfs[-1] + residue
    return co_imfs[:n_clusters]

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
            logger.warning("VMD failed (%s); using high-freq cluster signal as-is.", exc)
            return signal[np.newaxis, :]

    if pad:
        u = u[:, :-1]

    return np.asarray(u[:, :len(signal)], dtype=np.float64)

def decompose_service_signal(signal: np.ndarray, cfg) -> List[np.ndarray]:

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < cfg.MIN_SIGNAL_LEN:
        logger.warning(
            "Signal too short (%d < %d); returning trivial decomposition.",
            len(signal), cfg.MIN_SIGNAL_LEN,
        )
        zeros = np.zeros_like(signal)
        return [signal.astype(np.float32), zeros.astype(np.float32), zeros.astype(np.float32)]

    imfs, residue = ceemdan_decompose(signal, cfg.CEEMDAN_EPSILON, cfg.CEEMDAN_TRIALS)
    logger.debug("CEEMDAN produced %d IMFs.", imfs.shape[0])

    co_imfs = cluster_imfs(
        imfs,
        residue,
        m=cfg.SE_M,
        r_frac=cfg.SE_R_FRAC,
        max_se_samples=cfg.SE_MAX_SAMPLES,
        n_clusters=cfg.N_CLUSTERS,
    )

    try:
        vmd_modes = vmd_decompose(
            co_imfs[0],
            K=cfg.VMD_K,
            alpha=cfg.VMD_ALPHA,
            tau=cfg.VMD_TAU,
            DC=cfg.VMD_DC,
            init=cfg.VMD_INIT,
            tol=cfg.VMD_TOL,
        )
        co_imfs[0] = vmd_modes.sum(axis=0)
    except Exception as exc:
        logger.warning("VMD step skipped: %s", exc)

    n = len(signal)
    for k in range(len(co_imfs)):
        arr = np.asarray(co_imfs[k], dtype=np.float64)
        if len(arr) != n:
            arr = arr[:n] if len(arr) > n else np.pad(arr, (0, n - len(arr)))
        co_imfs[k] = arr

    reconstruction_error = signal - np.sum(co_imfs, axis=0)
    co_imfs[-1] = co_imfs[-1] + reconstruction_error

    return [np.asarray(arr, dtype=np.float32) for arr in co_imfs[:cfg.N_CLUSTERS]]

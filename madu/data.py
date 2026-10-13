"""CPM time-series loading + windowing for the Madu predictor.

Paper mapping (§4.1.2): x_t = {C_t, t_m, t_h, t_w} with C_t = CPM and
t_m/t_h/t_w the minute/hour/day-of-week at time t. The paper calls t_w
the "week number"; per user instruction it is interpreted as day of week
(0-6) derived from the global minute index.
"""
import os

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 10080  # 7 * 1440; day-of-week = (minute // 1440) % 7


def load_cpm_series(msname, parquet_dir="/dataset/ms_call_counts/per_minute"):
    """Raw (minute, cpm) pairs for one microservice, sorted by minute."""
    q = (
        pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))
        .filter(pl.col("dm") == msname)
        .select(["minute", "call_count"])
        .sort("minute")
    )
    df = q.collect()
    if len(df) == 0:
        raise ValueError(f"No CPM data for msname='{msname}' in {parquet_dir}")
    return (
        df["minute"].to_numpy().astype(np.int64),
        df["call_count"].to_numpy().astype(np.float64),
    )


def build_continuous_series(minutes, cpm, max_minutes=0):
    """Reindex onto a dense minute grid; missing minutes mean 0 calls.

    The per-minute aggregates are complete counts, so a (minute, dm) row
    missing from the table implies the MS received no calls that minute.
    Returns (grid_minutes, grid_cpm). `max_minutes` caps the length
    (keeps the most recent part) for quick pipeline tests.
    """
    lo, hi = int(minutes.min()), int(minutes.max())
    grid = np.arange(lo, hi + 1, dtype=np.int64)
    full = np.zeros(len(grid), dtype=np.float64)
    full[minutes - lo] = cpm
    if max_minutes and len(grid) > max_minutes:
        grid = grid[-max_minutes:]
        full = full[-max_minutes:]
    return grid, full


def time_features(minutes):
    """t_m in [0,1), t_h in [0,1), t_w (day of week) in [0,1)."""
    minutes = np.asarray(minutes, dtype=np.int64)
    tm = (minutes % MINUTES_PER_HOUR) / MINUTES_PER_HOUR
    th = ((minutes // MINUTES_PER_HOUR) % 24) / 24.0
    tw = ((minutes // MINUTES_PER_DAY) % 7) / 7.0
    return np.stack([tm, th, tw], axis=1).astype(np.float32)


def build_feature_matrix(grid_minutes, t_series, mode="paper"):
    """Feature matrix aligned with a (possibly transformed) CPM series.

    mode="paper": [t_cpm, t_m, t_h, t_w] (§4.1.2).
    mode="full": paper + cyclic sin/cos(time) + causal rolling mean/std of
      the transformed CPM (helps bursty series; rolling uses past only).
    """
    grid_minutes = np.asarray(grid_minutes, dtype=np.int64)
    tm = (grid_minutes % MINUTES_PER_HOUR) / MINUTES_PER_HOUR
    th = ((grid_minutes // MINUTES_PER_HOUR) % 24) / 24.0
    tw = ((grid_minutes // MINUTES_PER_DAY) % 7) / 7.0
    base = np.stack([t_series, tm, th, tw], axis=1).astype(np.float32)
    if mode == "paper":
        return base
    if mode != "full":
        raise ValueError(f"unknown feat mode='{mode}'")
    hh = (grid_minutes // MINUTES_PER_HOUR) % 24
    mm = grid_minutes % MINUTES_PER_HOUR
    dw = (grid_minutes // MINUTES_PER_DAY) % 7
    cyc = np.stack([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24),
                    np.sin(2 * np.pi * mm / 60), np.cos(2 * np.pi * mm / 60),
                    np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)],
                   axis=1).astype(np.float32)
    t = np.asarray(t_series, dtype=np.float64)
    c = np.concatenate([[t[0]], t])  # causal padding
    w = 60
    rmean = np.array([c[max(0, i - w + 1):i + 1].mean() for i in range(len(t))])
    rstd = np.array([c[max(0, i - w + 1):i + 1].std() for i in range(len(t))])
    extra = np.stack([rmean, rstd], axis=1).astype(np.float32)
    return np.concatenate([base, cyc, extra], axis=1)


def resample_series(grid_minutes, series, res=1, how="mean"):
    """Coarsen minute series to `res`-minute bins (for bursty MSs whose
    single-minute spikes are unpredictable in isolation).
    how="max": bin peak (autoscaling-relevant: provision for the peak);
    how="mean": bin average. Returns (bin_start_minutes, binned)."""
    if res <= 1:
        return grid_minutes, np.asarray(series, dtype=np.float64)
    grid_minutes = np.asarray(grid_minutes, dtype=np.int64)
    series = np.asarray(series, dtype=np.float64)
    bins = grid_minutes // res
    uniq, inv = np.unique(bins, return_inverse=True)
    agg = np.zeros(len(uniq), dtype=np.float64)
    if how == "max":
        np.maximum.at(agg, inv, series)
    elif how == "mean":
        s = np.zeros(len(uniq)); c = np.zeros(len(uniq))
        np.add.at(s, inv, series); np.add.at(c, inv, 1)
        agg = s / np.maximum(c, 1)
    else:
        raise ValueError(f"unknown resample how='{how}'")
    return (uniq * res).astype(np.int64), agg


def make_windows(cpm, feats, n, m):
    """Sliding windows: X (N, n, F), y_raw (N, m), window starts."""
    n_total = len(cpm)
    n_win = n_total - n - m + 1
    if n_win <= 0:
        raise ValueError(
            f"Series too short ({n_total}) for n={n}, m={m}"
        )
    starts = np.arange(n_win, dtype=np.int64)
    X = np.stack(
        [np.concatenate([cpm[s:s + n, None], feats[s:s + n]], axis=1) for s in starts]
    ).astype(np.float32)
    y_raw = np.stack([cpm[s + n:s + n + m] for s in starts]).astype(np.float64)
    return X, y_raw, starts


def chronological_split(n_windows, train_frac=0.7, val_frac=0.1):
    """Paper (§5.1) trains on 30 days, validates on 10; our trace is ~13 days,
    so default to a 70/10/20 chronological split of windows."""
    n_train = int(n_windows * train_frac)
    n_val = int(n_windows * val_frac)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_windows)
    return train_idx, val_idx, test_idx


class MaduDataset(Dataset):
    def __init__(self, X, y_norm):
        self.X = torch.from_numpy(X)
        self.y_norm = torch.from_numpy(y_norm.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y_norm[i]


def prepare_ms(msname, n, m, parquet_dir="/dataset/ms_call_counts/per_minute",
               train_frac=0.7, val_frac=0.1, max_minutes=0, transform="max",
               feat="paper", resample=1, resample_how="mean"):
    """Full prep for one MS. Normalization uses train data only (no leakage).

    transform="max": CPM / train-max (original; collapses heavy-tailed
      series since typical values become ~1e-5).
    transform="log1p": z-scored log1p(CPM) (handles burst spikes spanning
      orders of magnitude; invert with expm1(z*std+mean)).
    Returns dict with torch datasets + norm stats + raw test targets.
    """
    minutes, cpm = load_cpm_series(msname, parquet_dir)
    grid, full = build_continuous_series(minutes, cpm, max_minutes)
    grid, full = resample_series(grid, full, resample, resample_how)
    X, y_raw, _ = make_windows(full, np.zeros((len(full), 1), np.float32), n, m)
    tr_idx, va_idx, te_idx = chronological_split(len(X), train_frac, val_frac)
    # Normalization stats use train INPUT minutes only: train window starts
    # [0, n_train) cover minutes [0, n_train + n). Extending to n+m would
    # peek at the first (m-1) validation minutes (window overlap).
    n_tr = len(tr_idx)
    train_end_time = (int(tr_idx.max()) + n) if n_tr else (n + m)
    if transform == "log1p":
        t = np.log1p(full)
        mean, std = float(t[:train_end_time].mean()), float(t[:train_end_time].std())
        if std <= 1e-12:
            std = 1.0
        norm = {"mode": "log1p", "mean": mean, "std": std}
        t_full = (t - mean) / std
        y_norm = (np.log1p(y_raw) - mean) / std
    elif transform == "max":
        scale = float(full[:train_end_time].max())
        if scale <= 0:
            scale = 1.0
        norm = {"mode": "max", "scale": scale}
        t_full = full / scale
        y_norm = y_raw / scale
    else:
        raise ValueError(f"unknown transform='{transform}'")
    F = build_feature_matrix(grid, t_full, mode=feat)
    X = np.stack([F[s:s + n] for s in np.arange(len(X))]).astype(np.float32)
    return {
        "msname": msname,
        "norm": norm,
        "scale": norm.get("scale", 1.0),
        "feat": feat,
        "resample": resample,
        "resample_how": resample_how,
        "train": MaduDataset(X[tr_idx], y_norm[tr_idx]),
        "val": MaduDataset(X[va_idx], y_norm[va_idx]),
        "test": MaduDataset(X[te_idx], y_norm[te_idx]),
        "test_raw": y_raw[te_idx],
        "series_minutes": grid,
        "series_cpm": full,
        "n": n,
        "m": m,
    }


def top_ms_names(k, parquet_dir="/dataset/ms_call_counts/per_minute"):
    """Top-k receiver MSs by total calls (UNKNOWN excluded before ranking,
    so the result always holds k names)."""
    q = (
        pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))
        .filter(pl.col("dm") != "UNKNOWN")
        .group_by("dm")
        .agg(pl.sum("call_count").alias("t"))
        .sort("t", descending=True)
        .head(k)
        .collect()
    )
    return q["dm"].to_list()[:k]

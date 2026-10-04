"""LDS-inspired low-std window downsampler (train split only).

Method: Deep Imbalanced Regression (Yang et al., ICML 2021) label-distribution
smoothing adapted to window volatility: score each train window by its temporal
std over target channels, histogram the scores, Gaussian-smooth the density
(LDS), then keep each window with probability ``1/sqrt(density)`` rescaled so
the rarest bin is kept with p=1 and the densest bin with p=``min_keep``.

Val/test splits are copied verbatim so evaluation stays unbiased.
"""
import os
import sys
import argparse
import glob
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import (
    feature_names_for_feature_set,
    target_features_for_feature_set,
)

SIGNATURE_NAME = "_downsample_signature.json"
STATS_NAME = "_downsample_stats.json"


def _gaussian_kernel(sigma):
    radius = max(1, int(round(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (k / sigma) ** 2)
    ker /= ker.sum()
    return ker


def _target_channel_indices(feature_set):
    feats = feature_names_for_feature_set(feature_set)
    targs = target_features_for_feature_set(feature_set)
    idxs = [feats.index(t) for t in targs if t in feats]
    if not idxs:
        idxs = list(range(len(feats)))
    return idxs


def _scores_of(block, chan_idx):
    if block.ndim == 2:
        block = block[:, :, np.newaxis]
    if len(chan_idx) > 1:
        sel = block[:, :, chan_idx]
    else:
        sel = block[:, :, chan_idx[0]:chan_idx[0] + 1]
    return sel.std(axis=1).mean(axis=1)


def _sample_scores_task(task):
    x_path, chan_idx, max_samples = task
    X = np.load(x_path, mmap_mode="r")
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    stride = max(1, n // max_samples)
    sub = X[::stride][:max_samples].astype(np.float32)
    return _scores_of(sub, chan_idx).ravel()


def _hist_task(task):
    x_path, chan_idx, edges, n_bins, chunk = task
    edges = np.asarray(edges)
    X = np.load(x_path, mmap_mode="r")
    counts = np.zeros(n_bins, dtype=np.float64)
    for s in range(0, X.shape[0], chunk):
        scores = _scores_of(X[s:s + chunk].astype(np.float32), chan_idx)
        bins = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
        counts += np.bincount(bins, minlength=n_bins)
    return counts


def _filter_task(task):
    (x_path, out_dir, chan_idx, edges, keep_prob, seed,
     ordinal, n_bins, chunk) = task
    edges = np.asarray(edges)
    keep_prob = np.asarray(keep_prob)
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    base = x_path.replace("_X_train.npy", "")
    X = np.load(x_path)
    y = np.load(base + "_y_train.npy")
    sid = np.load(base + "_sid_train.npy")
    ylast_path = base + "_ylast_train.npy"
    ylast = np.load(ylast_path) if os.path.exists(ylast_path) else None
    n = X.shape[0]
    parts = []
    for s in range(0, n, chunk):
        parts.append(_scores_of(X[s:s + chunk].astype(np.float32), chan_idx))
    scores = np.concatenate(parts)
    bins = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    mask = rng.random(n) < keep_prob[bins]
    if mask.sum() == 0:
        mask[int(np.argmax(scores))] = True  # never emit an empty shard
    out_base = os.path.join(out_dir, os.path.basename(base))
    np.save(f"{out_base}_X_train.npy", X[mask])
    np.save(f"{out_base}_y_train.npy", y[mask])
    np.save(f"{out_base}_sid_train.npy", sid[mask])
    if ylast is not None:
        np.save(f"{out_base}_ylast_train.npy", ylast[mask])
    bin_kept = np.bincount(bins, weights=mask.astype(np.float64),
                           minlength=n_bins).astype(np.int64)
    bin_total = np.bincount(bins, minlength=n_bins).astype(np.int64)
    return os.path.basename(x_path), n, int(mask.sum()), bin_total, bin_kept


def _shard_fingerprint(paths):
    fp = []
    for p in sorted(paths):
        try:
            st = os.stat(p)
            fp.append([os.path.basename(p), st.st_size, st.st_mtime_ns])
        except OSError:
            fp.append([os.path.basename(p), -1, -1])
    return fp


def main():
    p = argparse.ArgumentParser(
        description="Downsample low-std train windows via LDS inverse-density sampling."
    )
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--std_bins", type=int, default=50)
    p.add_argument("--std_sigma", type=float, default=2.0,
                   help="Gaussian smoothing sigma in bins (LDS kernel).")
    p.add_argument("--min_keep", type=float, default=0.1,
                   help="Keep probability floor for the densest bin.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_std_pct", type=float, default=99.9,
                   help="Robust max percentile for linear bin edges.")
    p.add_argument("--chunk", type=int, default=200000)
    p.add_argument("--jobs", type=int, default=16,
                   help="Parallel workers over shards (default: %(default)s).")
    p.add_argument("--recompute", action="store_true",
                   help="Ignore cached signature and rebuild.")
    args = p.parse_args()
    jobs = max(1, args.jobs or 1)

    os.makedirs(args.out_dir, exist_ok=True)

    train_x_files = sorted(glob.glob(os.path.join(args.windows_dir, "part-*_X_train.npy")))
    if not train_x_files:
        print(f"No train X shards in {args.windows_dir}, nothing to do.")
        return

    signature = {
        "feature_set": args.feature_set,
        "std_bins": args.std_bins,
        "std_sigma": args.std_sigma,
        "min_keep": args.min_keep,
        "seed": args.seed,
        "max_std_pct": args.max_std_pct,
        "train_shards": _shard_fingerprint(train_x_files),
    }
    sig_path = os.path.join(args.out_dir, SIGNATURE_NAME)
    if not args.recompute and os.path.exists(sig_path):
        try:
            with open(sig_path) as f:
                if json.load(f) == signature:
                    out_train = glob.glob(os.path.join(args.out_dir, "part-*_X_train.npy"))
                    if len(out_train) == len(train_x_files):
                        print("Downsample cache valid, skipping.")
                        return
        except (OSError, ValueError):
            pass

    chan_idx = _target_channel_indices(args.feature_set)
    print(f"Target channels for std scoring: {chan_idx} (jobs={jobs})")

    # Pass 1a: robust max from a sampled subset of scores (parallel).
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        sample_parts = list(ex.map(
            _sample_scores_task,
            [(xp, chan_idx, 200000) for xp in train_x_files],
        ))
    sample_parts = [a for a in sample_parts if len(a)]
    sample_scores = np.concatenate(sample_parts) if sample_parts \
        else np.zeros(1, dtype=np.float32)
    robust_max = float(np.percentile(sample_scores, args.max_std_pct))
    del sample_scores, sample_parts
    print(f"Robust std max (p{args.max_std_pct}): {robust_max:.6f}")

    n_bins = args.std_bins
    if robust_max < 1e-6:
        print("All windows near-constant; copying train split unchanged.")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        keep_prob = np.ones(n_bins)
    else:
        edges = np.linspace(0.0, robust_max, n_bins + 1)
        # Pass 1b: global histogram of scores (parallel).
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            hist_parts = list(ex.map(
                _hist_task,
                [(xp, chan_idx, edges.tolist(), n_bins, args.chunk)
                 for xp in train_x_files],
            ))
        counts = np.sum(hist_parts, axis=0)
        del hist_parts
        # LDS: Gaussian-smooth the density, keep prob ~ 1/sqrt(density).
        ker = _gaussian_kernel(args.std_sigma)
        density = np.convolve(counts + 1e-3, ker, mode="same")
        w = 1.0 / np.sqrt(density)
        # Normalize by the rarest *non-empty* bin so empty bins (eps floor)
        # cannot collapse every populated bin to min_keep.
        nonzero = counts > 0
        w_max = w[nonzero].max() if nonzero.any() else w.max()
        keep_prob = w / w_max
        keep_prob = np.clip(keep_prob, args.min_keep, 1.0)
        print(f"Bin counts head: {counts[:5].astype(int).tolist()} "
              f"tail: {counts[-5:].astype(int).tolist()}")
        print(f"Keep probs head: {np.round(keep_prob[:5], 3).tolist()} "
              f"tail: {np.round(keep_prob[-5:], 3).tolist()}")

    # Pass 2: sample train shards (parallel; per-shard SeedSequence keeps
    # results deterministic), copy val/test verbatim.
    tasks = []
    for ordinal, x_path in enumerate(train_x_files):
        base = x_path.replace("_X_train.npy", "")
        if not (os.path.exists(base + "_y_train.npy")
                and os.path.exists(base + "_sid_train.npy")):
            print(f"  Skipping {os.path.basename(x_path)}: missing y or sid.")
            continue
        if robust_max < 1e-6:
            kp = np.ones(n_bins)  # near-constant: keep everything
        else:
            kp = keep_prob
        tasks.append((x_path, args.out_dir, chan_idx, edges.tolist(),
                      np.asarray(kp).tolist(), args.seed, ordinal,
                      n_bins, args.chunk))
    total, kept = 0, 0
    bin_total = np.zeros(n_bins, dtype=np.int64)
    bin_kept = np.zeros(n_bins, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, n, k, bt, bk in ex.map(_filter_task, tasks):
            total += n
            kept += k
            bin_total += np.asarray(bt)
            bin_kept += np.asarray(bk)
            print(f"  {name}: kept {k}/{n} ({100.0 * k / max(1, n):.1f}%)")

    for split in ("val", "test"):
        x_files = sorted(glob.glob(os.path.join(args.windows_dir, f"part-*_X_{split}.npy")))
        n_copy = 0
        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            copied = False
            for suffix in (f"_X_{split}.npy", f"_y_{split}.npy",
                           f"_sid_{split}.npy", f"_ylast_{split}.npy"):
                src = base + suffix
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(args.out_dir, os.path.basename(src)))
                    copied = True
            n_copy += 1 if copied else 0
        print(f"Split {split}: copied {n_copy} shards verbatim.")

    stats = {
        "total_train": int(total),
        "kept_train": int(kept),
        "kept_frac": float(kept / max(1, total)),
        "robust_max": robust_max,
        "bin_counts": bin_total.tolist(),
        "bin_kept": bin_kept.tolist(),
        "keep_prob": [float(v) for v in keep_prob],
    }
    with open(os.path.join(args.out_dir, STATS_NAME), "w") as f:
        json.dump(stats, f, indent=2)
    with open(sig_path, "w") as f:
        json.dump(signature, f, indent=2)
    print(f"\nTrain downsampled: {kept}/{total} ({100.0 * kept / max(1, total):.2f}% kept) "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()

import argparse
import ctypes
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import gc
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import get_feature_set
from preprocessing.swt.config import CFG
from preprocessing.swt.decomposition import decompose_window

_PROGRESS = {"windows_done": None, "shards_done": None, "cur_shard_idx": None}


class _TehranFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.getMessage()}"


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "preprocess.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = _TehranFormatter()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    logging.getLogger("preprocessing.swt.decomposition").setLevel(logging.WARNING)
    logging.info("Preprocessing log: %s", log_path)


def _chunk_and_per_worker_mem(input_len, total_channels, budget=0.9e9):
    """Pick a per-worker processing chunk so RAM stays bounded (~budget bytes).

    Each window costs `input_len * (n_in + n_out) * 4` bytes while a chunk is
    being decomposed (input slice + 12-channel output buffer), plus base
    interpreter overhead (~250MB).
    """
    per_window = input_len * (2 + total_channels) * 4
    chunk = max(20_000, min(500_000, int(budget // max(per_window, 1))))
    per_worker = chunk * per_window + 250e6
    return chunk, per_worker


def _memory_aware_workers(requested, per_worker, max_fraction=0.8):
    """Cap workers so total peak RSS stays under ~max_fraction of free RAM."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:
        return requested
    return max(1, min(requested, int(avail * max_fraction / per_worker)))


def _decompose_shard(task, windows_done, shards_done, cur_shard_idx):
    (shard_x_path, shard_y_path, shard_sid_path,
     shard_out_x_path, shard_out_y_path, shard_out_sid_path, shard_out_last_path,
     feature_cfgs, feature_indices, target_indices, shard_idx) = task

    t0 = time.time()

    # Progress objects are passed as arguments (works with spawn on Windows)
    cur_shard_idx.value = shard_idx

    X = np.load(shard_x_path, mmap_mode="r")
    Y_full = np.load(shard_y_path)
    S = np.load(shard_y_path.replace("_y_", "_sid_"))
    N, input_len, n_input_features = X.shape

    # Extract last values for target features
    last_vals = np.asarray(X[:, -1, target_indices], dtype=np.float16)

    # Compute channel counts per feature
    n_channels_per_feat = [cfg.SWT_LEVEL + 1 for cfg in feature_cfgs]
    total_channels = sum(n_channels_per_feat)

    out_dir = os.path.dirname(shard_out_x_path)
    os.makedirs(out_dir, exist_ok=True)

    chunk_size, _ = _chunk_and_per_worker_mem(input_len, total_channels)

    # Stream X_dec into a full-size memmap. All windows are kept.
    tmp_x = shard_out_x_path + ".tmp"
    out_mmap = np.lib.format.open_memmap(
        tmp_x, mode="w+", dtype="float16",
        shape=(N, input_len, total_channels),
    )
    for a in range(0, N, chunk_size):
        b = min(a + chunk_size, N)
        X_chunk = X[a:b]
        m = b - a
        X_dec_chunk = np.zeros((m, input_len, total_channels), dtype=np.float16)
        for i in range(m):
            ch_offset = 0
            for feat_idx, feat_cfg in zip(feature_indices, feature_cfgs):
                n_ch = feat_cfg.SWT_LEVEL + 1
                feat_ch = decompose_window(X_chunk[i, :, feat_idx], feat_cfg)
                if feat_ch is None:
                    feat_ch = np.zeros((n_ch, input_len), dtype=np.float32)
                    feat_ch[0] = X_chunk[i, :, feat_idx].astype(np.float32)
                X_dec_chunk[i, :, ch_offset:ch_offset + n_ch] = feat_ch.T
                ch_offset += n_ch
        out_mmap[a:b] = X_dec_chunk
        windows_done.value += m
        del X_dec_chunk, X_chunk
        gc.collect()

    del out_mmap
    os.replace(tmp_x, shard_out_x_path)

    # Slice Y to only include the target features for this feature set.
    # Y_full has shape (N, pred_horizon, num_targets_total); we take the first
    # len(target_indices) targets, which matches the order of target_features.
    num_targets = len(target_indices)
    Y_subset = Y_full[..., :num_targets].astype(np.float16)

    np.save(shard_out_last_path, last_vals)
    np.save(shard_out_y_path, Y_subset)
    np.save(shard_out_sid_path, S)

    shards_done.value += 1

    elapsed = time.time() - t0
    return (os.path.basename(shard_x_path), N, 0, elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose windowed signals via SWT into channels."
    )
    ap.add_argument("--windows_dir", required=True,
                    help="Path to build_windows output directory")
    ap.add_argument("--out_dir", default="/dataset/swt_preprocess",
                    help="Output directory for decomposed shards")
    ap.add_argument("--feature_set", default="cpu",
                    help="Feature set: 'cpu' for CPU only, 'cpu_mem_both' for CPU + memory, 'cpu_mem_http_rpc' for CPU + memory + http/rpc MCR")
    ap.add_argument("--swt_level", type=int, default=CFG.SWT_LEVEL,
                    help=f"SWT decomposition level for CPU (default: {CFG.SWT_LEVEL})")
    ap.add_argument("--mem_swt_level", type=int, default=CFG.MEM_SWT_LEVEL,
                    help=f"SWT decomposition level for memory (default: {CFG.MEM_SWT_LEVEL})")
    ap.add_argument("--extra_swt_level", type=int, default=None,
                    help=f"SWT decomposition level for extra features (default: same as --swt_level)")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    ap.add_argument("--recompute_preprocessing", action="store_true",
                    help="Recompute the preprocessing approach output, ignoring cached shards")
    args = ap.parse_args()

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_features = spec.get("targets", [spec.get("target")])
    target_indices = [feature_names.index(tf) for tf in target_features]

    extra_swt_level = args.extra_swt_level if args.extra_swt_level is not None else args.swt_level

    # Build config for each feature
    feature_cfgs = []
    for feat in feature_names:
        if feat == "cpu_utilization":
            feature_cfgs.append(replace(CFG, SWT_LEVEL=args.swt_level))
        elif feat == "memory_utilization":
            feature_cfgs.append(replace(CFG, SWT_LEVEL=args.mem_swt_level))
        else:
            feature_cfgs.append(replace(CFG, SWT_LEVEL=extra_swt_level))

    feature_indices = list(range(len(feature_names)))

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)

    input_len = PREPROCESSING.INPUT_LEN
    total_channels = sum(cfg.SWT_LEVEL + 1 for cfg in feature_cfgs)
    _, per_worker = _chunk_and_per_worker_mem(input_len, total_channels)
    capped = _memory_aware_workers(num_workers, per_worker)
    if capped != num_workers:
        logging.info(
            "Memory-aware worker cap: %d -> %d (per-worker ~%.1fGB)",
            num_workers, capped, per_worker / 1e9,
        )
    num_workers = capped

    # Write metadata file describing channel structure
    meta = {
        "feature_set": args.feature_set,
        "features": feature_names,
        "target_features": target_features,
        "target_indices": target_indices,
        "swt_levels": {feat: cfg.SWT_LEVEL for feat, cfg in zip(feature_names, feature_cfgs)},
        "channel_counts": {feat: cfg.SWT_LEVEL + 1 for feat, cfg in zip(feature_names, feature_cfgs)},
        "total_channels": total_channels,
        "input_len": input_len,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote metadata to %s", meta_path)

    splits = ["train", "val", "test"]
    shard_tasks = []
    shard_names = []
    for split in splits:
        x_shards = sorted(glob.glob(os.path.join(args.windows_dir, f"part-*_X_{split}.npy")))
        for x_path in x_shards:
            base = os.path.basename(x_path).replace(f"_X_{split}.npy", "")
            y_path = os.path.join(args.windows_dir, f"{base}_y_{split}.npy")
            sid_path = os.path.join(args.windows_dir, f"{base}_sid_{split}.npy")

            if not os.path.exists(y_path) or not os.path.exists(sid_path):
                logging.warning("Missing y/sid for shard %s, skipping", base)
                continue

            out_x = os.path.join(args.out_dir, f"{base}_X_{split}.npy")
            out_y = os.path.join(args.out_dir, f"{base}_y_{split}.npy")
            out_sid = os.path.join(args.out_dir, f"{base}_sid_{split}.npy")
            out_last = os.path.join(args.out_dir, f"{base}_last_{split}.npy")

            if not args.recompute_preprocessing and os.path.exists(out_x) and os.path.exists(out_last):
                logging.info("Shard %s already done, skipping", base)
                continue

            shard_names.append(os.path.basename(x_path))
            shard_tasks.append((
                x_path, y_path, sid_path,
                out_x, out_y, out_sid, out_last,
                feature_cfgs, feature_indices, target_indices,
            ))

    if not shard_tasks:
        logging.info("No shards to process")
        return

    # Total windows across all shards, read from .npy headers only (cheap).
    total_windows = 0
    for t in shard_tasks:
        total_windows += np.load(t[0], mmap_mode="r").shape[0]

    # Shared progress state: workers update these (inherited at fork), a monitor
    # thread in the parent renders them.
    manager = mp.Manager()
    windows_done = manager.Value(ctypes.c_longlong, 0)
    shards_done = manager.Value(ctypes.c_longlong, 0)
    cur_shard_idx = manager.Value(ctypes.c_longlong, -1)
    global _PROGRESS
    _PROGRESS = {
        "windows_done": windows_done,
        "shards_done": shards_done,
        "cur_shard_idx": cur_shard_idx,
    }
    shard_tasks = [t + (i,) for i, t in enumerate(shard_tasks)]

    logging.info("Processing %d shards (%d windows) with %d workers",
                 len(shard_tasks), total_windows, num_workers)
    logging.info("Feature set: %s, features: %s, total_channels: %d",
                 args.feature_set, feature_names, total_channels)

    t_start = time.time()
    kept_windows = 0
    total_skipped = 0

    mp_context = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp_context) as executor:
        futures = {
            executor.submit(_decompose_shard, t, windows_done, shards_done, cur_shard_idx): t
            for t in shard_tasks
        }
        pbar = tqdm(
            total=total_windows, desc="SWT Decomposition",
            unit="", unit_scale=True,
            bar_format=("{desc}: {percentage:5.1f}%|{bar}| "
                        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                        "{rate_fmt}{postfix}]"),
        )

        stop_monitor = threading.Event()

        def _monitor():
            while not stop_monitor.is_set():
                idx = cur_shard_idx.value
                cur = shard_names[idx] if 0 <= idx < len(shard_names) else ""
                pbar.set_postfix_str(
                    f" {shards_done.value}/{len(shard_tasks)} shards | {cur}")
                pbar.n = windows_done.value
                pbar.refresh(nolock=True)
                time.sleep(0.5)

        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()
        try:
            for future in as_completed(futures):
                shard_key, n_windows, n_skipped, elapsed = future.result()
                kept_windows += n_windows
                total_skipped += n_skipped
        finally:
            stop_monitor.set()
            monitor.join(timeout=2)
        pbar.n = windows_done.value
        pbar.refresh()
        pbar.close()

    elapsed = time.time() - t_start
    logging.info(
        "Preprocessing complete. Shards: %d | Windows: %d | Skipped: %d | Time: %.1fs",
        len(shard_tasks), kept_windows, total_skipped, elapsed,
    )


if __name__ == "__main__":
    main()

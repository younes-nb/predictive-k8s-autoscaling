import argparse
import ctypes
import glob
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


def _decompose_shard(task):
    (shard_x_path, shard_y_path, shard_sid_path,
     shard_out_x_path, shard_out_y_path, shard_out_sid_path, shard_out_last_path,
     cpu_cfg, mem_cfg, feature_idx_cpu, feature_idx_mem, has_mem, shard_idx) = task

    t0 = time.time()

    # Shared progress objects are set as module globals before the pool forks,
    # so workers inherit them (Synchronized objects can't be pickled through
    # the task queue).
    windows_done = _PROGRESS["windows_done"]
    shards_done = _PROGRESS["shards_done"]
    cur_shard_idx = _PROGRESS["cur_shard_idx"]
    with cur_shard_idx.get_lock():
        cur_shard_idx.value = shard_idx

    X = np.load(shard_x_path, mmap_mode="r")
    Y = np.load(shard_y_path)
    S = np.load(shard_sid_path)
    N, input_len, _ = X.shape

    last_cpu = np.asarray(X[:, -1, feature_idx_cpu], dtype=np.float32)
    if has_mem:
        last_mem = np.asarray(X[:, -1, feature_idx_mem], dtype=np.float32)
    else:
        last_mem = None

    n_cpu_channels = cpu_cfg.SWT_LEVEL + 1
    n_mem_channels = (mem_cfg.SWT_LEVEL + 1) if has_mem else 0
    total_channels = n_cpu_channels + n_mem_channels

    out_dir = os.path.dirname(shard_out_x_path)
    os.makedirs(out_dir, exist_ok=True)

    chunk_size, _ = _chunk_and_per_worker_mem(input_len, total_channels)

    # Stream X_dec into a full-size memmap at a running offset, writing only
    # kept windows. RAM stays bounded to one chunk (~1GB), not the whole shard
    # (~4GB for a 12-channel train shard).
    tmp_x = shard_out_x_path + ".tmp"
    out_mmap = np.lib.format.open_memmap(
        tmp_x, mode="w+", dtype="float32",
        shape=(N, input_len, total_channels),
    )
    keep = np.zeros(N, dtype=bool)
    offset = 0
    for a in range(0, N, chunk_size):
        b = min(a + chunk_size, N)
        X_chunk = X[a:b]
        m = b - a
        X_dec_chunk = np.zeros((m, input_len, total_channels), dtype=np.float32)
        keep_chunk = np.ones(m, dtype=bool)
        for i in range(m):
            cpu_ch = decompose_window(X_chunk[i, :, feature_idx_cpu], cpu_cfg)
            if cpu_ch is None:
                keep_chunk[i] = False
                continue
            X_dec_chunk[i, :, :n_cpu_channels] = cpu_ch.T
            if has_mem:
                mem_ch = decompose_window(X_chunk[i, :, feature_idx_mem], mem_cfg)
                if mem_ch is None:
                    keep_chunk[i] = False
                    continue
                X_dec_chunk[i, :, n_cpu_channels:] = mem_ch.T
        keep[a:b] = keep_chunk
        n_kept = int(keep_chunk.sum())
        if n_kept:
            out_mmap[offset:offset + n_kept] = X_dec_chunk[keep_chunk]
            offset += n_kept
        with windows_done.get_lock():
            windows_done.value += m
        del X_dec_chunk, X_chunk, keep_chunk
        gc.collect()

    n_kept_total = int(keep.sum())
    del out_mmap
    if n_kept_total != N:
        # Rare: some windows skipped (constant signal -> std ~ 0). Compact into
        # the final file with the true row count, streaming to stay bounded.
        final = np.lib.format.open_memmap(
            shard_out_x_path, mode="w+", dtype="float32",
            shape=(n_kept_total, input_len, total_channels),
        )
        tmp = np.load(tmp_x, mmap_mode="r")
        for a in range(0, n_kept_total, chunk_size):
            b = min(a + chunk_size, n_kept_total)
            final[a:b] = tmp[a:b]
        del final, tmp
        gc.collect()
        os.remove(tmp_x)
    else:
        os.replace(tmp_x, shard_out_x_path)

    last = np.stack([last_cpu, last_mem], axis=-1) if has_mem else last_cpu
    np.save(shard_out_last_path, last[keep])
    np.save(shard_out_y_path, Y[keep])
    np.save(shard_out_sid_path, S[keep])

    with shards_done.get_lock():
        shards_done.value += 1

    elapsed = time.time() - t0
    return (os.path.basename(shard_x_path), n_kept_total, N - n_kept_total, elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose windowed signals via SWT into channels."
    )
    ap.add_argument("--windows_dir", required=True,
                    help="Path to build_windows output directory")
    ap.add_argument("--out_dir", default="/dataset/swt_preprocess",
                    help="Output directory for decomposed shards")
    ap.add_argument("--feature_set", default="cpu",
                    help="Feature set: 'cpu' for CPU only, 'cpu_mem_both' for CPU + memory")
    ap.add_argument("--swt_level", type=int, default=CFG.SWT_LEVEL,
                    help=f"SWT decomposition level for CPU (default: {CFG.SWT_LEVEL})")
    ap.add_argument("--mem_swt_level", type=int, default=CFG.MEM_SWT_LEVEL,
                    help=f"SWT decomposition level for memory (default: {CFG.MEM_SWT_LEVEL})")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    ap.add_argument("--recompute_preprocessing", action="store_true",
                    help="Recompute the preprocessing approach output, ignoring cached shards")
    args = ap.parse_args()

    has_mem = args.feature_set == "cpu_mem_both"
    cpu_cfg = replace(CFG, SWT_LEVEL=args.swt_level)
    mem_cfg = replace(CFG, SWT_LEVEL=args.mem_swt_level)

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)

    input_len = PREPROCESSING.INPUT_LEN
    total_channels = (cpu_cfg.SWT_LEVEL + 1) + ((mem_cfg.SWT_LEVEL + 1) if has_mem else 0)
    _, per_worker = _chunk_and_per_worker_mem(input_len, total_channels)
    capped = _memory_aware_workers(num_workers, per_worker)
    if capped != num_workers:
        logging.info(
            "Memory-aware worker cap: %d -> %d (per-worker ~%.1fGB)",
            num_workers, capped, per_worker / 1e9,
        )
    num_workers = capped

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    feature_idx_cpu = feature_names.index("cpu_utilization")
    feature_idx_mem = feature_names.index("memory_utilization") if has_mem else -1

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
                cpu_cfg, mem_cfg, feature_idx_cpu, feature_idx_mem, has_mem,
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
    windows_done = mp.Value(ctypes.c_longlong, 0)
    shards_done = mp.Value(ctypes.c_longlong, 0)
    cur_shard_idx = mp.Value(ctypes.c_longlong, -1)
    global _PROGRESS
    _PROGRESS = {
        "windows_done": windows_done,
        "shards_done": shards_done,
        "cur_shard_idx": cur_shard_idx,
    }
    shard_tasks = [t + (i,) for i, t in enumerate(shard_tasks)]

    logging.info("Processing %d shards (%d windows) with %d workers",
                 len(shard_tasks), total_windows, num_workers)

    t_start = time.time()
    kept_windows = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp.get_context("fork")) as executor:
        futures = {executor.submit(_decompose_shard, t): t for t in shard_tasks}
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

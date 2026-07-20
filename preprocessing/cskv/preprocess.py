import argparse
import glob
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from preprocessing.cskv.config import CFG, set_seed
from preprocessing.cskv.decomposition import decompose_service_signal
from shared.config_training_defaults import TRAINING

MAX_IMFS = 3


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
    logging.getLogger("PyEMD").setLevel(logging.WARNING)
    logging.info("Preprocessing log: %s", log_path)


def _decompose_shard(task):
    (shard_x_path, shard_y_path, shard_sid_path,
     shard_out_x_path, shard_out_y_path, shard_out_sid_path, shard_out_last_path,
     cpu_cfg, feature_idx_cpu, no_clustering) = task

    t0 = time.time()

    X = np.load(shard_x_path).astype(np.float32)
    N, input_len, _ = X.shape

    last_cpu = X[:, -1, feature_idx_cpu]

    if no_clustering:
        n_channels = MAX_IMFS
    else:
        n_channels = cpu_cfg.VMD_K + cpu_cfg.N_CLUSTERS - 1

    X_dec = np.zeros((N, input_len, n_channels), dtype=np.float32)
    for i in range(N):
        cpu_signal = X[i, :, feature_idx_cpu].astype(np.float64)
        decomposed = decompose_service_signal(cpu_signal, cpu_cfg, return_raw_imfs=no_clustering)

        if no_clustering:
            raw_imfs = decomposed
            n_raw = len(raw_imfs)
            for k in range(min(n_raw, MAX_IMFS)):
                X_dec[i, :, k] = raw_imfs[k][:input_len]
        else:
            co_imfs, vmd_modes = decomposed
            if vmd_modes is not None:
                n_vmd = vmd_modes.shape[0]
                for k in range(n_vmd):
                    X_dec[i, :, k] = vmd_modes[k][:input_len]
                for k in range(1, cpu_cfg.N_CLUSTERS):
                    X_dec[i, :, n_vmd + k - 1] = co_imfs[k][:input_len]
            else:
                for k in range(cpu_cfg.N_CLUSTERS):
                    X_dec[i, :, k] = co_imfs[k][:input_len]

    np.save(shard_out_x_path, X_dec)
    np.save(shard_out_last_path, last_cpu)
    shutil.copy2(shard_y_path, shard_out_y_path)
    shutil.copy2(shard_sid_path, shard_out_sid_path)

    elapsed = time.time() - t0
    return (os.path.basename(shard_x_path), N, elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose windowed CPU signals via CEEMDAN+Clustering+VMD into channels."
    )
    ap.add_argument("--windows_dir", required=True,
                    help="Path to build_windows output directory")
    ap.add_argument("--out_dir", default="/dataset/cskv_preprocess",
                    help="Output directory for decomposed shards")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    ap.add_argument("--no_clustering", action="store_true",
                    help="Skip Sample Entropy, K-Means, and VMD; use raw IMFs only.")
    args = ap.parse_args()

    set_seed(TRAINING.SEED)

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)

    cpu_cfg = CFG
    spec = get_feature_set("cpu")
    feature_names = list(spec["features"])
    feature_idx_cpu = feature_names.index("cpu_utilization")

    splits = ["train", "val", "test"]
    shard_tasks = []
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

            if os.path.exists(out_x) and os.path.exists(out_last):
                logging.info("Shard %s already done, skipping", base)
                continue

            shard_tasks.append((
                x_path, y_path, sid_path,
                out_x, out_y, out_sid, out_last,
                cpu_cfg, feature_idx_cpu, args.no_clustering,
            ))

    if not shard_tasks:
        logging.info("No shards to process")
        return

    logging.info("Processing %d shards with %d workers", len(shard_tasks), num_workers)

    t_start = time.time()
    total_windows = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_decompose_shard, t): t for t in shard_tasks}
        pbar = tqdm(total=len(shard_tasks), desc="CSKV Decomposition", unit="shard")
        for future in as_completed(futures):
            shard_key, n_windows, elapsed = future.result()
            total_windows += n_windows
            pbar.set_postfix_str(f"{shard_key} ({n_windows}w, {elapsed:.1f}s)")
            pbar.update(1)
        pbar.close()

    elapsed = time.time() - t_start
    logging.info(
        "Preprocessing complete. Shards: %d | Windows: %d | Time: %.1fs",
        len(shard_tasks), total_windows, elapsed,
    )


if __name__ == "__main__":
    main()

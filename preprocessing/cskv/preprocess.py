
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.config_training_defaults import TRAINING
from preprocessing.cskv.config import CFG, set_seed
from preprocessing.cskv._decompose_worker import MAX_IMFS

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

def _load_batch_signals(
    msresource_dir: str, services_batch: list[str],
) -> dict[str, np.ndarray]:
    if not services_batch:
        return {}
    df = (
        pl.scan_parquet(os.path.join(msresource_dir, "*.parquet"))
        .filter(pl.col("msname").is_in(services_batch))
        .group_by("msname", "timestamp")
        .agg(pl.col("cpu_utilization").mean())
        .sort("msname", "timestamp")
        .collect(engine="streaming")
    )
    signals = {}
    for key, group in df.group_by("msname", maintain_order=True):
        ms_name = key[0] if isinstance(key, tuple) else key
        sig = group["cpu_utilization"].to_numpy().astype(np.float32)
        if len(sig) >= PREPROCESSING.INPUT_LEN + PREPROCESSING.PRED_HORIZON:
            signals[ms_name] = sig
    return signals

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose MSResource CPU signals into Co-IMFs for CVCBM training."
    )
    ap.add_argument("--msresource_dir", default=PATHS.PARQUET_MSRESOURCE)
    ap.add_argument("--out_dir", default="/dataset/cskv_preprocess")
    ap.add_argument("--max_services", type=int, default=0, help="Max services (0 = all)")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    ap.add_argument("--batch_size", type=int, default=512,
                    help="Services per batch (default: 512). Load this many from parquet at once.")
    ap.add_argument("--no_clustering", action="store_true",
                    help="Skip Sample Entropy, K-Means, and VMD; use raw IMFs only.")
    args = ap.parse_args()

    set_seed(TRAINING.SEED)

    # Compute worker count as a fraction of available CPU cores (default 90%).
    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)
    if args.no_clustering:
        for k in range(MAX_IMFS):
            os.makedirs(os.path.join(args.out_dir, f"raw_imf_{k}"), exist_ok=True)
    else:
        for k in range(CFG.N_CLUSTERS):
            os.makedirs(os.path.join(args.out_dir, f"co_imf_{k}"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "original"), exist_ok=True)

    # --- Service list ---
    # Reuse a persistent service name index to avoid re-scanning 112 GB of parquet.
    svc_names_path = os.path.join(args.out_dir, "_service_names.npy")
    if os.path.exists(svc_names_path):
        all_services = np.load(svc_names_path, allow_pickle=True).tolist()
        logging.info("Loaded %d service names from index: %s", len(all_services), svc_names_path)
    else:
        logging.info("Scanning parquet for service names (one-time index creation)...")
        t0 = time.time()
        all_services = (
            pl.scan_parquet(os.path.join(args.msresource_dir, "*.parquet"))
            .select("msname").unique()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        all_services.sort()
        np.save(svc_names_path, np.array(all_services, dtype=object))
        logging.info("Found %d unique services and saved index in %.1fs",
                     len(all_services), time.time() - t0)

    # On reruns, load the previous service-to-index mapping so we can
    # look up cached original/ files by service name.
    svc_to_idx_path = os.path.join(args.out_dir, "_svc_to_idx.json")
    prev_svc_to_idx: dict[str, int] = {}
    if os.path.exists(svc_to_idx_path):
        with open(svc_to_idx_path) as f:
            prev_svc_to_idx = json.load(f)
        logging.info("Loaded previous index mapping with %d entries", len(prev_svc_to_idx))
    else:
        # Fallback: reconstruct from .meta files left by a previous run.
        meta_dir = args.out_dir
        meta_files = sorted(
            f for f in os.listdir(meta_dir)
            if f.endswith(".meta.txt") and f.startswith("service_")
        )
        if meta_files:
            for fname in meta_files:
                meta_path = os.path.join(meta_dir, fname)
                try:
                    with open(meta_path) as f:
                        ms_name = f.read().strip()
                    idx_str = fname.replace("service_", "").replace(".meta.txt", "")
                    prev_svc_to_idx[ms_name] = int(idx_str)
                except Exception:
                    pass
            if prev_svc_to_idx:
                logging.info("Reconstructed index mapping from %d meta files",
                             len(prev_svc_to_idx))

    t_load = time.time()
    all_signals: dict[str, np.ndarray] = {}

    # First, try to fill from cached files — avoids any parquet access on reruns.
    need = args.max_services or len(all_services)
    for ms_name in all_services:
        if len(all_signals) >= need:
            break
        idx = prev_svc_to_idx.get(ms_name)
        if idx is None:
            continue
        path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
        if not os.path.exists(path):
            continue
        try:
            sig = np.load(path).astype(np.float32)
            if len(sig) >= PREPROCESSING.INPUT_LEN + PREPROCESSING.PRED_HORIZON:
                all_signals[ms_name] = sig
        except Exception:
            pass

    if len(all_signals) >= need:
        logging.info("Filled %d/%d services from cache (%.1fs) — skipping parquet",
                     len(all_signals), need, time.time() - t_load)
    else:
        logging.info("Cache provided %d/%d services, loading remaining from parquet...",
                     len(all_signals), need)
        # Load remaining signals from parquet in chunks.
        remaining = [ms for ms in all_services if ms not in all_signals]
        batch_size_parquet = 2000
        for chunk_start in range(0, len(remaining), batch_size_parquet):
            chunk = remaining[chunk_start:chunk_start + batch_size_parquet]
            chunk_signals = _load_batch_signals(args.msresource_dir, chunk)
            for ms_name, sig in chunk_signals.items():
                if ms_name not in all_signals:
                    all_signals[ms_name] = sig
                    if len(all_signals) >= need:
                        break
            elapsed = time.time() - t_load
            logging.info("Loaded %d / %d needed signals (%.1fs, chunk %d)",
                         len(all_signals), need, elapsed,
                         chunk_start // batch_size_parquet + 1)
            if len(all_signals) >= need:
                break

    logging.info("Total valid signals: %d (%.1fs)", len(all_signals), time.time() - t_load)

    # Build services list from valid signals; if max_services is set,
    # take the first N valid ones (skips past any that are too short).
    services = sorted(all_signals.keys())
    if args.max_services:
        services = services[:args.max_services]
        all_signals = {ms: all_signals[ms] for ms in services}

    total = len(services)
    batch_size = args.batch_size if args.batch_size > 0 else num_workers
    n_batches = (total + batch_size - 1) // batch_size

    logging.info("Services: %d | Workers: %d | Batch: %d | Batches: %d",
                 total, num_workers, batch_size, n_batches)
    if args.max_services and total < args.max_services:
        logging.warning("Only %d valid services found (requested %d)", total, args.max_services)

    svc_to_idx = {name: i for i, name in enumerate(services)}
    with open(svc_to_idx_path, "w") as f:
        json.dump(svc_to_idx, f)
    worker_script = os.path.join(THIS_DIR, "_decompose_worker.py")
    processed = 0
    skipped = 0
    t_start = time.time()

    pre_done = 0
    for i in range(total):
        dm = os.path.join(args.out_dir, f"service_{i:05d}.done")
        if os.path.exists(dm):
            if args.no_clustering:
                if os.path.exists(os.path.join(args.out_dir, "raw_imf_0", f"service_{i:05d}.npy")):
                    pre_done += 1
            else:
                if all(
                    os.path.exists(os.path.join(args.out_dir, f"co_imf_{k}", f"service_{i:05d}.npy"))
                    for k in range(CFG.N_CLUSTERS)
                ):
                    pre_done += 1
    if pre_done:
        logging.info("Pre-existing completed services: %d", pre_done)

    for batch_idx in range(n_batches):
        batch_services = services[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        signals = {}
        for ms_name in batch_services:
            idx = svc_to_idx[ms_name]
            path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            if os.path.exists(path):
                sig = np.load(path).astype(np.float32)
                if len(sig) >= PREPROCESSING.INPUT_LEN + PREPROCESSING.PRED_HORIZON:
                    signals[ms_name] = sig
                    continue
            if ms_name in all_signals:
                signals[ms_name] = all_signals[ms_name]

        worker_args = []
        for ms_name, sig in signals.items():
            idx = svc_to_idx[ms_name]
            out_path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            if not os.path.exists(out_path):
                np.save(out_path, sig)
            done_marker = os.path.join(args.out_dir, f"service_{idx:05d}.done")
            if os.path.exists(done_marker):
                if args.no_clustering:
                    files_exist = os.path.exists(os.path.join(args.out_dir, "raw_imf_0", f"service_{idx:05d}.npy"))
                else:
                    files_exist = all(
                        os.path.exists(os.path.join(args.out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
                        for k in range(CFG.N_CLUSTERS)
                    )
                if files_exist:
                    continue
                os.remove(done_marker)
            worker_args.append((ms_name, idx))

        for ms_name in batch_services:
            if ms_name not in signals:
                idx = svc_to_idx.get(ms_name, -1)
                if idx >= 0:
                    dm = os.path.join(args.out_dir, f"service_{idx:05d}.done")
                    if not os.path.exists(dm):
                        skipped += 1
                    else:
                        if args.no_clustering:
                            files_exist = os.path.exists(os.path.join(args.out_dir, "raw_imf_0", f"service_{idx:05d}.npy"))
                        else:
                            files_exist = all(
                                os.path.exists(os.path.join(args.out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
                                for k in range(CFG.N_CLUSTERS)
                            )
                        if not files_exist:
                            os.remove(dm)
                            skipped += 1

        if not worker_args:
            logging.info("  Batch %d/%d: no valid services.", batch_idx + 1, n_batches)
            continue

        n_submitted = len(worker_args)
        logging.info("  Batch %d/%d: submitting %d services to %d workers...",
                     batch_idx + 1, n_batches, n_submitted, num_workers)

        def _run_worker(ms_name: str, idx: int) -> tuple[int, str, str, float]:
            worker_env = os.environ.copy()
            worker_env["OMP_NUM_THREADS"] = "1"
            worker_env["MKL_NUM_THREADS"] = "1"
            worker_env["OPENBLAS_NUM_THREADS"] = "1"
            worker_env["VECLIB_MAXIMUM_THREADS"] = "1"
            worker_env["NUMEXPR_NUM_THREADS"] = "1"
            worker_env["LOKY_MAX_CPU_COUNT"] = "1"
            worker_env["JOBLIB_NUM_THREADS"] = "1"
            proc = subprocess.Popen(
                [sys.executable, worker_script, ms_name, str(idx), args.out_dir, "1" if args.no_clustering else "0"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=worker_env,
            )
            t0 = time.time()
            stdout, stderr = proc.communicate()
            duration = time.time() - t0
            return proc.returncode, stdout.decode(), stderr.decode(), duration

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            fut_to_meta = {
                executor.submit(_run_worker, ms_name, idx): (ms_name, idx)
                for ms_name, idx in worker_args
            }

            batch_done = 0
            batch_start_t = time.time()
            log_every = max(1, n_submitted // 10)

            for future in as_completed(fut_to_meta):
                ms_name, idx = fut_to_meta[future]
                returncode, r_msg, err_msg, duration = future.result()
                r_msg_lines = r_msg.strip().split("\n") if r_msg else []

                result_line = ""
                for line in r_msg_lines:
                    if line.startswith("RESULT:"):
                        result_line = line
                        break

                done_count = pre_done + processed + skipped + 1

                if returncode == 0 and result_line:
                    parts = result_line.split(":", 2)
                    success = parts[1] == "True" if len(parts) >= 2 else True
                    message = parts[2] if len(parts) >= 3 else "ok"
                    if "(MAE=" in message:
                        message = "ok"
                    if success:
                        if "already done" in message or "too short" in message:
                            skipped += 1
                        else:
                            processed += 1
                        batch_done += 1
                    else:
                        skipped += 1
                        batch_done += 1
                        logging.error(
                            "  [%s] ERROR — %s",
                            ms_name, message,
                        )
                        continue
                else:
                    skipped += 1
                    batch_done += 1
                    err_msg_short = ""
                    for line in r_msg_lines:
                        if line.startswith("RESULT:False:ERROR:"):
                            err_msg_short = line.split(":", 2)[-1].strip()[:200]
                            break
                    if not err_msg_short:
                        err_msg_short = err_msg.strip()[:200] if err_msg else ""
                    if not err_msg_short:
                        err_msg_short = "unknown"
                    logging.error(
                        "  [%s] CRASHED (exit=%d): %s",
                        ms_name, returncode, err_msg_short,
                    )
                    continue

                # Show per-service completion with name and timing
                pct = f" ({done_count*100//total}%)" if total else ""
                logging.info(
                    "  [%s] done (%d/%d%s, batch %d/%d, %.1fs)",
                    ms_name, done_count, total, pct,
                    batch_idx + 1, n_batches, duration,
                )

                # Periodic batch-level progress with ETA
                if batch_done % log_every == 0 or batch_done == n_submitted:
                    elapsed_batch = time.time() - batch_start_t
                    rate = batch_done / elapsed_batch if elapsed_batch > 0 else 0
                    remaining = n_submitted - batch_done
                    eta = remaining / rate if rate > 0 else 0
                    total_elapsed = time.time() - t_start
                    total_rate = done_count / total_elapsed if total_elapsed > 0 else 0
                    total_remaining = total - done_count
                    total_eta = total_remaining / total_rate if total_rate > 0 else 0
                    logging.info(
                        "  batch %d/%d: %d/%d services, %.1f/s, batch ETA=%.0fs, overall ETA=%.0fs",
                        batch_idx + 1, n_batches, batch_done, n_submitted,
                        rate, eta, total_eta,
                    )

        batch_elapsed = time.time() - batch_start_t
        logging.info("  Batch %d/%d done — %d services (%.1f/s), %.1fs elapsed, %.0fs total.",
                     batch_idx + 1, n_batches, n_submitted,
                     n_submitted / batch_elapsed if batch_elapsed > 0 else 0,
                     batch_elapsed, time.time() - t_start)

    elapsed = time.time() - t_start
    logging.info(
        "Preprocessing complete. Processed: %d | Skipped/Failed: %d | Time: %.1fs",
        processed, skipped, elapsed,
    )

if __name__ == "__main__":
    main()

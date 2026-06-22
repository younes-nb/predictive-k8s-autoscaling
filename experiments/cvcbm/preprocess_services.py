
import argparse
import logging
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS
from experiments.cvcbm.config import CFG

class _ElapsedFormatter(logging.Formatter):

    def __init__(self):
        super().__init__()
        self._start = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = time.time() - self._start
        record.elapsed = f"{elapsed:8.1f}"
        return f"  {record.elapsed}s [{record.levelname}] {record.getMessage()}"

def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "preprocess.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = _ElapsedFormatter()
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
        if len(sig) >= CFG.MIN_SIGNAL_LEN:
            signals[ms_name] = sig
    return signals

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose MSResource CPU signals into Co-IMFs for CVCBM training."
    )
    ap.add_argument("--msresource_dir", default=PATHS.PARQUET_MSRESOURCE)
    ap.add_argument("--out_dir", default="/dataset/cvcbm_preprocess")
    ap.add_argument("--max_services", type=int, default=0, help="Max services (0 = all)")
    ap.add_argument("--num_workers", type=int, default=32,
                    help="Parallel worker count (default: 32)")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Services per batch (default: 256). Load this many from parquet at once.")
    args = ap.parse_args()

    setup_logging(args.out_dir)
    for k in range(CFG.N_CLUSTERS):
        os.makedirs(os.path.join(args.out_dir, f"co_imf_{k}"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "original"), exist_ok=True)

    logging.info("Loading service list from: %s", args.msresource_dir)
    all_services = (
        pl.scan_parquet(os.path.join(args.msresource_dir, "*.parquet"))
        .select("msname").unique()
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )
    all_services.sort()

    services = all_services[:args.max_services] if args.max_services else all_services
    total = len(services)
    batch_size = args.batch_size if args.batch_size > 0 else args.num_workers
    n_batches = (total + batch_size - 1) // batch_size

    logging.info("Services: %d | Workers: %d | Batch: %d | Batches: %d",
                 total, args.num_workers, batch_size, n_batches)

    svc_to_idx = {name: i for i, name in enumerate(services)}
    worker_script = os.path.join(THIS_DIR, "_decompose_worker.py")
    processed = 0
    skipped = 0
    t_start = time.time()

    for batch_idx in range(n_batches):
        batch_services = services[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        logging.info("Batch %d/%d: loading %d services from parquet...",
                     batch_idx + 1, n_batches, len(batch_services))
        signals = _load_batch_signals(args.msresource_dir, batch_services)

        worker_args = []
        for ms_name, sig in signals.items():
            idx = svc_to_idx[ms_name]
            np.save(os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy"), sig)
            worker_args.append((ms_name, idx))

        for ms_name in batch_services:
            if ms_name not in signals:
                idx = svc_to_idx.get(ms_name, -1)
                if idx >= 0 and not os.path.exists(
                    os.path.join(args.out_dir, f"service_{idx:05d}.done")
                ):
                    skipped += 1

        if not worker_args:
            logging.info("  No valid services in this batch.")
            continue

        n_submitted = len(worker_args)
        logging.info("  Submitting %d services to %d workers...", n_submitted, args.num_workers)

        idx_in_batch = 0
        while idx_in_batch < n_submitted:
            chunk = worker_args[idx_in_batch: idx_in_batch + args.num_workers]
            procs = []
            for ms_name, idx in chunk:
                worker_env = os.environ.copy()
                worker_env["OMP_NUM_THREADS"] = "1"
                worker_env["MKL_NUM_THREADS"] = "1"
                worker_env["OPENBLAS_NUM_THREADS"] = "1"
                worker_env["VECLIB_MAXIMUM_THREADS"] = "1"
                worker_env["NUMEXPR_NUM_THREADS"] = "1"
                worker_env["LOKY_MAX_CPU_COUNT"] = "1"
                worker_env["JOBLIB_NUM_THREADS"] = "1"
                procs.append(subprocess.Popen(
                    [sys.executable, worker_script, ms_name, str(idx), args.out_dir],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=worker_env,
                ))

            for p in procs:
                stdout, stderr = p.communicate()
                r_msg = stdout.decode().strip() if stdout else ""
                r_msg_lines = r_msg.split("\n") if r_msg else []

                result_line = ""
                for line in r_msg_lines:
                    if line.startswith("RESULT:"):
                        result_line = line
                        break

                elapsed = time.time() - t_start
                done_count = processed + skipped + 1

                if p.returncode == 0 and result_line:
                    parts = result_line.split(":", 2)
                    success = parts[1] == "True" if len(parts) >= 2 else True
                    message = parts[2] if len(parts) >= 3 else "ok"
                    if success:
                        if "already done" in message or "too short" in message:
                            skipped += 1
                        else:
                            processed += 1
                        logging.info(
                            "[%d/%d] %s  (%d done, batch %d/%d, %.1fs)",
                            done_count, total, message,
                            done_count, batch_idx + 1, n_batches, elapsed,
                        )
                    else:
                        skipped += 1
                        logging.error(
                            "[%d/%d] ERROR — %s",
                            done_count, total, message,
                        )
                else:
                    skipped += 1
                    err_msg = stderr.decode().strip()[:200] if stderr else "unknown"
                    logging.error(
                        "[%d/%d] Worker crashed (exit=%d): %s",
                        done_count, total, p.returncode, err_msg,
                    )

            idx_in_batch += len(chunk)

        logging.info("  Batch %d/%d done — %d services, %.1fs elapsed.",
                     batch_idx + 1, n_batches, n_submitted,
                     time.time() - t_start)

    elapsed = time.time() - t_start
    logging.info(
        "Preprocessing complete. Processed: %d | Skipped/Failed: %d | Time: %.1fs",
        processed, skipped, elapsed,
    )

if __name__ == "__main__":
    main()

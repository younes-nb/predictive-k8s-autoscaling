
import argparse
import logging
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ap.add_argument("--num_workers", type=int, default=80,
                    help="Parallel worker count (default: 80)")
    ap.add_argument("--batch_size", type=int, default=512,
                    help="Services per batch (default: 512). Load this many from parquet at once.")
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

        signals = {}
        for ms_name in batch_services:
            idx = svc_to_idx[ms_name]
            path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            if os.path.exists(path):
                sig = np.load(path).astype(np.float32)
                if len(sig) >= CFG.MIN_SIGNAL_LEN:
                    signals[ms_name] = sig

        worker_args = []
        for ms_name, sig in signals.items():
            idx = svc_to_idx[ms_name]
            out_path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            if not os.path.exists(out_path):
                np.save(out_path, sig)
            done_marker = os.path.join(args.out_dir, f"service_{idx:05d}.done")
            if os.path.exists(done_marker):
                co_imf_files_exist = all(
                    os.path.exists(os.path.join(args.out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
                    for k in range(CFG.N_CLUSTERS)
                )
                if co_imf_files_exist:
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
                    elif not all(
                        os.path.exists(os.path.join(args.out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
                        for k in range(CFG.N_CLUSTERS)
                    ):
                        os.remove(dm)
                        skipped += 1

        if not worker_args:
            logging.info("  Batch %d/%d: no valid services.", batch_idx + 1, n_batches)
            continue

        n_submitted = len(worker_args)
        logging.info("  Batch %d/%d: submitting %d services to %d workers...",
                     batch_idx + 1, n_batches, n_submitted, args.num_workers)

        def _run_worker(ms_name: str, idx: int) -> tuple[int, str, str]:
            worker_env = os.environ.copy()
            worker_env["OMP_NUM_THREADS"] = "1"
            worker_env["MKL_NUM_THREADS"] = "1"
            worker_env["OPENBLAS_NUM_THREADS"] = "1"
            worker_env["VECLIB_MAXIMUM_THREADS"] = "1"
            worker_env["NUMEXPR_NUM_THREADS"] = "1"
            worker_env["LOKY_MAX_CPU_COUNT"] = "1"
            worker_env["JOBLIB_NUM_THREADS"] = "1"
            proc = subprocess.Popen(
                [sys.executable, worker_script, ms_name, str(idx), args.out_dir],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=worker_env,
            )
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode()

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            fut_to_meta = {
                executor.submit(_run_worker, ms_name, idx): (ms_name, idx)
                for ms_name, idx in worker_args
            }

            for future in as_completed(fut_to_meta):
                ms_name, idx = fut_to_meta[future]
                returncode, r_msg, err_msg = future.result()
                r_msg_lines = r_msg.strip().split("\n") if r_msg else []

                result_line = ""
                for line in r_msg_lines:
                    if line.startswith("RESULT:"):
                        result_line = line
                        break

                elapsed = time.time() - t_start
                done_count = processed + skipped + 1

                if returncode == 0 and result_line:
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
                    err_msg_short = err_msg.strip()[:200] if err_msg else ""
                    if not err_msg_short:
                        for line in r_msg_lines:
                            if line.startswith("RESULT:False:ERROR:"):
                                err_msg_short = line.split(":", 2)[-1].strip()[:200]
                                break
                    if not err_msg_short:
                        err_msg_short = "unknown"
                    logging.error(
                        "[%d/%d] Worker crashed (exit=%d): %s",
                        done_count, total, returncode, err_msg_short,
                    )

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

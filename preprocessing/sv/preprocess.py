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
from preprocessing.sv.config import CFG

CHANNEL_DIRS = [f"vmd_mode_{k}" for k in range(10)] + ["D2", "A2"]
MEM_CHANNEL_DIRS = [f"mem_vmd_mode_{k}" for k in range(10)] + ["mem_D2", "mem_A2"]


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
    logging.getLogger("preprocessing.sv.decomposition").setLevel(logging.WARNING)
    logging.info("Preprocessing log: %s", log_path)


def _load_batch_signals(
    msresource_dir: str, services_batch: list[str], load_memory: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    if not services_batch:
        return {}, None
    agg_cols = [pl.col("cpu_utilization").mean()]
    if load_memory:
        agg_cols.append(pl.col("memory_utilization").mean())
    df = (
        pl.scan_parquet(os.path.join(msresource_dir, "*.parquet"))
        .filter(pl.col("msname").is_in(services_batch))
        .group_by("msname", "timestamp")
        .agg(agg_cols)
        .sort("msname", "timestamp")
        .collect(engine="streaming")
    )
    min_len = PREPROCESSING.INPUT_LEN + PREPROCESSING.PRED_HORIZON
    cpu_signals: dict[str, np.ndarray] = {}
    mem_signals: dict[str, np.ndarray] = {} if load_memory else None
    for key, group in df.group_by("msname", maintain_order=True):
        ms_name = key[0] if isinstance(key, tuple) else key
        sig = group["cpu_utilization"].to_numpy().astype(np.float32)
        if len(sig) >= min_len:
            cpu_signals[ms_name] = sig
        if load_memory:
            mem_sig = group["memory_utilization"].to_numpy().astype(np.float32)
            if len(mem_sig) >= min_len:
                mem_signals[ms_name] = mem_sig
    return cpu_signals, mem_signals


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Decompose MSResource signals via SWT + VMD into channels."
    )
    ap.add_argument("--msresource_dir", default=PATHS.PARQUET_MSRESOURCE)
    ap.add_argument("--out_dir", default="/dataset/sv_preprocess")
    ap.add_argument("--max_services", type=int, default=0, help="Max services (0 = all)")
    ap.add_argument("--feature_set", default="cpu",
                    help="Feature set: 'cpu' for CPU only, 'cpu_mem_both' for CPU + memory")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    ap.add_argument("--batch_size", type=int, default=512,
                    help="Services per batch (default: 512).")
    args = ap.parse_args()

    has_mem = args.feature_set == "cpu_mem_both"
    all_channel_dirs = CHANNEL_DIRS + (MEM_CHANNEL_DIRS if has_mem else [])

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)

    for d in all_channel_dirs:
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "original"), exist_ok=True)
    if has_mem:
        os.makedirs(os.path.join(args.out_dir, "mem_original"), exist_ok=True)

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

    svc_to_idx_path = os.path.join(args.out_dir, "_svc_to_idx.json")
    prev_svc_to_idx: dict[str, int] = {}
    if os.path.exists(svc_to_idx_path):
        with open(svc_to_idx_path) as f:
            prev_svc_to_idx = json.load(f)
        logging.info("Loaded previous index mapping with %d entries", len(prev_svc_to_idx))
    else:
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
    all_cpu_signals: dict[str, np.ndarray] = {}
    all_mem_signals: dict[str, np.ndarray] = {} if has_mem else None

    need = args.max_services or len(all_services)
    min_len = PREPROCESSING.INPUT_LEN + PREPROCESSING.PRED_HORIZON
    for ms_name in all_services:
        if len(all_cpu_signals) >= need:
            break
        idx = prev_svc_to_idx.get(ms_name)
        if idx is None:
            continue
        path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
        if not os.path.exists(path):
            continue
        try:
            sig = np.load(path).astype(np.float32)
            if len(sig) >= min_len:
                all_cpu_signals[ms_name] = sig
        except Exception:
            pass
        if has_mem:
            mem_path = os.path.join(args.out_dir, "mem_original", f"service_{idx:05d}.npy")
            if os.path.exists(mem_path):
                try:
                    mem_sig = np.load(mem_path).astype(np.float32)
                    if len(mem_sig) >= min_len:
                        all_mem_signals[ms_name] = mem_sig
                except Exception:
                    pass

    if len(all_cpu_signals) >= need:
        logging.info("Filled %d/%d services from cache (%.1fs) — skipping parquet",
                     len(all_cpu_signals), need, time.time() - t_load)
    else:
        logging.info("Cache provided %d/%d services, loading remaining from parquet...",
                     len(all_cpu_signals), need)
        remaining = [ms for ms in all_services if ms not in all_cpu_signals]
        batch_size_parquet = 2000
        for chunk_start in range(0, len(remaining), batch_size_parquet):
            chunk = remaining[chunk_start:chunk_start + batch_size_parquet]
            cpu_chunk, mem_chunk = _load_batch_signals(
                args.msresource_dir, chunk, load_memory=has_mem,
            )
            for ms_name, sig in cpu_chunk.items():
                if ms_name not in all_cpu_signals:
                    all_cpu_signals[ms_name] = sig
                    if len(all_cpu_signals) >= need:
                        break
            if has_mem and mem_chunk:
                for ms_name, sig in mem_chunk.items():
                    if ms_name not in all_mem_signals:
                        all_mem_signals[ms_name] = sig
            elapsed = time.time() - t_load
            logging.info("Loaded %d / %d needed signals (%.1fs, chunk %d)",
                         len(all_cpu_signals), need, elapsed,
                         chunk_start // batch_size_parquet + 1)
            if len(all_cpu_signals) >= need:
                break

    if has_mem:
        valid = set(all_cpu_signals.keys()) & set(all_mem_signals.keys())
        dropped = len(all_cpu_signals) - len(valid)
        if dropped:
            logging.info("Dropped %d services without memory signal", dropped)
        all_cpu_signals = {ms: all_cpu_signals[ms] for ms in valid}
        all_mem_signals = {ms: all_mem_signals[ms] for ms in valid}

    logging.info("Total valid signals: %d (%.1fs)%s",
                 len(all_cpu_signals), time.time() - t_load,
                 " (CPU+mem)" if has_mem else " (CPU)")

    services = sorted(all_cpu_signals.keys())
    if args.max_services:
        services = services[:args.max_services]
        all_cpu_signals = {ms: all_cpu_signals[ms] for ms in services}
        if has_mem:
            all_mem_signals = {ms: all_mem_signals[ms] for ms in services}

    total = len(services)
    batch_size = args.batch_size if args.batch_size > 0 else num_workers
    n_batches = (total + batch_size - 1) // batch_size

    logging.info("Services: %d | Workers: %d | Batch: %d | Batches: %d | Feature set: %s",
                 total, num_workers, batch_size, n_batches, args.feature_set)
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
        dm = os.path.join(args.out_dir, CHANNEL_DIRS[0], f"service_{i:05d}.done")
        if os.path.exists(dm):
            if all(
                os.path.exists(os.path.join(args.out_dir, d, f"service_{i:05d}.npy"))
                for d in all_channel_dirs
            ):
                pre_done += 1
    if pre_done:
        logging.info("Pre-existing completed services: %d", pre_done)

    for batch_idx in range(n_batches):
        batch_services = services[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        cpu_batch: dict[str, np.ndarray] = {}
        mem_batch: dict[str, np.ndarray] = {} if has_mem else None
        for ms_name in batch_services:
            idx = svc_to_idx[ms_name]
            path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            cpu_sig = None
            if os.path.exists(path):
                try:
                    cpu_sig = np.load(path).astype(np.float32)
                    if len(cpu_sig) < min_len:
                        cpu_sig = None
                except Exception:
                    cpu_sig = None
            if cpu_sig is None and ms_name in all_cpu_signals:
                cpu_sig = all_cpu_signals[ms_name]
            if cpu_sig is None:
                continue

            mem_sig = None
            if has_mem:
                mem_path = os.path.join(args.out_dir, "mem_original", f"service_{idx:05d}.npy")
                if os.path.exists(mem_path):
                    try:
                        mem_sig = np.load(mem_path).astype(np.float32)
                        if len(mem_sig) < min_len:
                            mem_sig = None
                    except Exception:
                        mem_sig = None
                if mem_sig is None and ms_name in all_mem_signals:
                    mem_sig = all_mem_signals[ms_name]
                if mem_sig is None:
                    continue

            cpu_batch[ms_name] = cpu_sig
            if has_mem:
                mem_batch[ms_name] = mem_sig

        worker_args = []
        for ms_name in cpu_batch:
            idx = svc_to_idx[ms_name]
            out_path = os.path.join(args.out_dir, "original", f"service_{idx:05d}.npy")
            if not os.path.exists(out_path):
                np.save(out_path, cpu_batch[ms_name])
            if has_mem:
                mem_out = os.path.join(args.out_dir, "mem_original", f"service_{idx:05d}.npy")
                if not os.path.exists(mem_out):
                    np.save(mem_out, mem_batch[ms_name])
            done_marker = os.path.join(args.out_dir, CHANNEL_DIRS[0], f"service_{idx:05d}.done")
            if os.path.exists(done_marker):
                files_exist = all(
                    os.path.exists(os.path.join(args.out_dir, d, f"service_{idx:05d}.npy"))
                    for d in all_channel_dirs
                )
                if files_exist:
                    continue
                os.remove(done_marker)
            worker_args.append((ms_name, idx))

        for ms_name in batch_services:
            if ms_name not in cpu_batch:
                idx = svc_to_idx.get(ms_name, -1)
                if idx >= 0:
                    dm = os.path.join(args.out_dir, CHANNEL_DIRS[0], f"service_{idx:05d}.done")
                    if not os.path.exists(dm):
                        skipped += 1
                    else:
                        files_exist = all(
                            os.path.exists(os.path.join(args.out_dir, d, f"service_{idx:05d}.npy"))
                            for d in all_channel_dirs
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
                [sys.executable, worker_script, ms_name, str(idx), args.out_dir,
                 args.feature_set],
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

                pct = f" ({done_count*100//total}%)" if total else ""
                logging.info(
                    "  [%s] done (%d/%d%s, batch %d/%d, %.1fs)",
                    ms_name, done_count, total, pct,
                    batch_idx + 1, n_batches, duration,
                )

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

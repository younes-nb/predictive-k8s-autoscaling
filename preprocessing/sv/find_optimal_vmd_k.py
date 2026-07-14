#!/usr/bin/env python3
"""
Determine optimal VMD K by analyzing all service signals window-by-window.
Mirrors the TSDP pipeline: INPUT_LEN=60, STRIDE=5, SWT->D1, then VMD K=1..20.
Parallelized via subprocess workers (same pattern as preprocess_services.py).
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.sv.config import CFG

logger = logging.getLogger(__name__)

DEFAULT_ORIGINAL_DIR = "/dataset/sv_preprocess/original"
WORKER_SCRIPT = os.path.join(THIS_DIR, "_vmd_k_worker.py")


class _TehranFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.getMessage()}"


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = _TehranFormatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def discover_services(original_dir: str) -> list[tuple[str, int]]:
    """Return list of (file_path, service_index) for valid service files."""
    pattern = os.path.join(original_dir, "service_*.npy")
    files = sorted(glob.glob(pattern))
    result = []
    for f in files:
        base = os.path.basename(f)
        idx = int(base.replace("service_", "").replace(".npy", ""))
        result.append((f, idx))
    return result


def run_worker(service_path: str, idx: int, max_k: int,
               json_dir: str) -> tuple[int, str, str, float]:
    """Run one worker subprocess, return (returncode, stdout, stderr, duration)."""
    json_out = os.path.join(json_dir, f"service_{idx:05d}.json")
    worker_env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT", "JOBLIB_NUM_THREADS"):
        worker_env[var] = "1"

    proc = subprocess.Popen(
        [sys.executable, WORKER_SCRIPT, service_path, str(max_k), json_out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=worker_env,
    )
    t0 = time.time()
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode(), time.time() - t0


def aggregate_all(json_dir: str, max_k: int, n_services: int):
    """Load worker JSON outputs and average metrics across all services."""
    cf_acc = {K: [] for K in range(1, max_k + 1)}
    kt_acc = {K: [] for K in range(1, max_k + 1)}
    cr_acc = {K: [] for K in range(1, max_k + 1)}
    total_windows = 0
    loaded = 0

    for f in sorted(glob.glob(os.path.join(json_dir, "service_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        total_windows += data.get("_windows", 0)
        loaded += 1
        for K in range(1, max_k + 1):
            key = f"K{K}"
            if key in data:
                cf_acc[K].append(data[key]['cf'])
                kt_acc[K].append(data[key]['kt'])
                cr_acc[K].append(data[key]['cr'])

    avg_center = []
    avg_kurt = []
    avg_corr = []
    for K in range(1, max_k + 1):
        avg_center.append({f'w{i+1}': v for i, v in enumerate([
            np.mean(cf_acc[K]) if cf_acc[K] else np.nan
        ])})
        avg_kurt.append({f'Kurt{i+1}': np.mean(kt_acc[K]) if kt_acc[K] else np.nan})
        avg_corr.append({f'R{i+1}': np.mean(cr_acc[K]) if cr_acc[K] else np.nan})

    # For multi-mode metrics, we need the per-mode breakdown from the worker.
    # The worker stores per-K averaged scalars (mean across modes).
    # Re-expand into per-mode tables using the config TOTAL_CHANNELS structure.
    # Since the worker averages across all modes for each K, we build simple tables.

    return avg_center, avg_kurt, avg_corr, total_windows, loaded


def expand_tables(json_dir: str, max_k: int):
    """Build full per-mode tables from worker JSONs.

    Each worker stores per-K: {cf: float, kt: float, cr: float} which is the
    mean across modes. To build per-mode columns, we recompute from the raw
    per-mode data stored in the worker output.

    Since the worker only stores the aggregate, we re-read the per-service
    per-K per-mode arrays. We modify the worker to store per-mode lists.
    """
    # We re-do aggregation from scratch by reading the JSONs.
    # Each JSON has K1..Kmax_k, each with cf, kt, cr (scalar averages).
    # To get per-mode columns, we need the worker to store per-mode data.
    # For now, we use the scalar average as a single-column table.
    # The full per-mode table requires the worker to emit mode-level data.

    # Collect all per-service per-K per-mode data
    per_k_centers = {K: [] for K in range(1, max_k + 1)}
    per_k_kurt = {K: [] for K in range(1, max_k + 1)}
    per_k_corr = {K: [] for K in range(1, max_k + 1)}

    for f in sorted(glob.glob(os.path.join(json_dir, "service_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        for K in range(1, max_k + 1):
            key = f"K{K}"
            if key in data and 'per_mode' in data[key]:
                pm = data[key]['per_mode']
                per_k_centers[K].append([m['cf'] for m in pm])
                per_k_kurt[K].append([m['kt'] for m in pm])
                per_k_corr[K].append([m['cr'] for m in pm])

    center_table = []
    kurt_table = []
    corr_table = []

    for K in range(1, max_k + 1):
        c_row = {}
        k_row = {}
        r_row = {}
        if per_k_centers[K]:
            n_modes = max(len(s) for s in per_k_centers[K])
            for mi in range(n_modes):
                vals = [s[mi] for s in per_k_centers[K] if mi < len(s)]
                c_row[f'w{mi+1}'] = float(np.mean(vals)) if vals else np.nan
                vals = [s[mi] for s in per_k_kurt[K] if mi < len(s)]
                k_row[f'Kurt{mi+1}'] = float(np.mean(vals)) if vals else np.nan
                vals = [s[mi] for s in per_k_corr[K] if mi < len(s)]
                r_row[f'R{mi+1}'] = float(np.mean(vals)) if vals else np.nan
        else:
            for mi in range(max_k):
                c_row[f'w{mi+1}'] = np.nan
                k_row[f'Kurt{mi+1}'] = np.nan
                r_row[f'R{mi+1}'] = np.nan
        center_table.append(c_row)
        kurt_table.append(k_row)
        corr_table.append(r_row)

    return center_table, kurt_table, corr_table


def print_table(data_list: list[dict], title: str, max_k: int = 20):
    print(f"\n{title}")
    cols = min(max_k, 8)
    print("-" * (6 + 10 * cols))

    sample_key = list(data_list[0].keys())[0]
    prefix = 'w' if 'w' in sample_key else ('Kurt' if 'Kurt' in sample_key else 'R')

    header = "K".ljust(4)
    for i in range(1, cols + 1):
        header += f"{prefix}{i}".ljust(10)
    print(header)

    for K in range(1, max_k + 1):
        row = f"{K}".ljust(4)
        current_data = data_list[K - 1]
        for i in range(1, cols + 1):
            key = f"{prefix}{i}"
            val = current_data.get(key, np.nan)
            if np.isnan(val):
                row += "-".ljust(10)
            elif prefix == 'w':
                row += f"{val:.6f}"[:9].ljust(10)
            else:
                row += f"{val:.4f}"[:9].ljust(10)
        print(row)


def find_optimal_k(
    center_freqs_list: list[dict],
    kurtosis_list: list[dict],
    correlation_list: list[dict],
    max_k: int = 20,
) -> tuple[int, dict]:
    logger.info("Determining optimal K...")

    n_modes = max_k
    cf_arr = np.nan_to_num(
        np.array([[d.get(f'w{i+1}', np.nan) for i in range(n_modes)] for d in center_freqs_list]),
        nan=0.0,
    )
    kt_arr = np.nan_to_num(
        np.array([[d.get(f'Kurt{i+1}', np.nan) for i in range(n_modes)] for d in kurtosis_list]),
        nan=0.0,
    )
    cr_arr = np.nan_to_num(
        np.array([[d.get(f'R{i+1}', np.nan) for i in range(n_modes)] for d in correlation_list]),
        nan=0.0,
    )

    analysis = {
        'over_decomp_k': max_k + 1,
        'best_kurtosis_k': 1,
        'correlation_decay_k': max_k + 1,
        'reasons': [],
    }

    # Criterion 1: center frequency convergence
    thr = 0.005
    for K in range(4, max_k + 1):
        fK = cf_arr[K - 1, :K]
        fP = cf_arr[K - 2, :K - 1]
        fK = fK[fK > 1e-6]
        fP = fP[fP > 1e-6]
        if len(fK) >= 3 and len(fP) >= 2:
            for freq in fK[1:]:
                if np.min(np.abs(fP - freq)) < thr:
                    analysis['over_decomp_k'] = K
                    analysis['reasons'].append(
                        f"Over-decomposition at K={K}: freq {freq:.6f} close to existing"
                    )
                    break
            if analysis['over_decomp_k'] <= max_k:
                break

    # Criterion 2: kurtosis of highest mode
    best_k, best_v = 1, -np.inf
    for K in range(1, min(analysis['over_decomp_k'], max_k) + 1):
        v = kt_arr[K - 1, K - 1]
        if v > best_v:
            best_v = v
            best_k = K
    analysis['best_kurtosis_k'] = best_k

    # Criterion 3: correlation decay
    cd_k = max_k + 1
    prev = float(np.sum(cr_arr[0, :1])) if cr_arr.shape[1] > 0 else 0
    for K in range(2, min(analysis['over_decomp_k'], max_k) + 1):
        curr = float(np.sum(cr_arr[K - 1, :K]))
        if curr < prev * 0.98:
            cd_k = K
            analysis['correlation_decay_k'] = K
            analysis['reasons'].append(
                f"Correlation decay at K={K}: {prev:.4f} -> {curr:.4f}"
            )
            break
        prev = curr

    candidates = [analysis['over_decomp_k'], analysis['best_kurtosis_k'] + 1,
                  analysis['correlation_decay_k']]
    valid = [c for c in candidates if 2 <= c <= max_k]
    optimal_k = min(valid) if valid else min(max_k, max(2, max_k // 3))

    logger.info("Over-decomp=%d  Best-kurt=%d  Corr-decay=%d  -> Optimal K=%d",
                analysis['over_decomp_k'], analysis['best_kurtosis_k'],
                analysis['correlation_decay_k'], optimal_k)
    return optimal_k, analysis


def main():
    ap = argparse.ArgumentParser(description="Find optimal VMD K (windowed, parallelized).")
    ap.add_argument("--original_dir", default=DEFAULT_ORIGINAL_DIR,
                    help="Dir with service_*.npy (default: %(default)s)")
    ap.add_argument("--max_k", type=int, default=10,
                    help="Maximum K to test (default: 10)")
    ap.add_argument("--max_services", type=int, default=0, help="0 = all")
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores (default: 0.9)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    services = discover_services(args.original_dir)
    if not services:
        logger.error("No valid service files in %s", args.original_dir)
        sys.exit(1)

    if args.max_services > 0:
        services = services[:args.max_services]

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))
    total = len(services)

    # Working dir for intermediate JSON files (use /tmp since /dataset may be read-only)
    import tempfile
    work_dir = os.path.join(tempfile.gettempdir(), "vmd_k_work")
    json_dir = os.path.join(work_dir, "jsons")
    os.makedirs(json_dir, exist_ok=True)

    logger.info("Services: %d | Workers: %d | Max K: %d | INPUT_LEN: %d | STRIDE: %d",
                total, num_workers, args.max_k, CFG.INPUT_LEN, CFG.STRIDE)

    n_batches = (total + args.batch_size - 1) // args.batch_size
    t_start = time.time()
    processed = 0
    skipped = 0

    for batch_idx in range(n_batches):
        batch = services[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
        logger.info("Batch %d/%d: %d services", batch_idx + 1, n_batches, len(batch))

        def _run(path, idx):
            return run_worker(path, idx, args.max_k, json_dir)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run, p, i): (p, i) for p, i in batch}
            batch_done = 0
            batch_t = time.time()

            for future in as_completed(futures):
                path, idx = futures[future]
                rc, out, err, dur = future.result()
                batch_done += 1

                if rc == 0 and "RESULT:True" in out:
                    msg = out.strip().split("\n")[-1]
                    if "no_data" in msg:
                        skipped += 1
                    else:
                        processed += 1
                else:
                    skipped += 1

                done_count = processed + skipped

                if batch_done % max(1, len(batch) // 10) == 0 or batch_done == len(batch):
                    elapsed = time.time() - batch_t
                    rate = batch_done / elapsed if elapsed > 0 else 0
                    total_elapsed = time.time() - t_start
                    total_rate = done_count / total_elapsed if total_elapsed > 0 else 0
                    eta = (total - done_count) / total_rate if total_rate > 0 else 0
                    logger.info(
                        "  [%d/%d] batch %d/%d done %d/%d (%.0f%%) | %.1f svc/s | ETA %.0fs",
                        done_count, total, batch_idx + 1, n_batches,
                        batch_done, len(batch), 100 * batch_done / len(batch),
                        rate, eta,
                    )

    total_elapsed = time.time() - t_start
    logger.info("Processed: %d | Skipped: %d | Time: %.1fs", processed, skipped, total_elapsed)

    if processed == 0:
        logger.error("No services processed successfully")
        sys.exit(1)

    logger.info("Aggregating metrics across %d services...", processed)
    center_table, kurt_table, corr_table = expand_tables(json_dir, args.max_k)

    print_table(center_table,
                f"Table 4: Avg center frequencies ({processed} services), K=1..{args.max_k}.",
                args.max_k)
    print_table(kurt_table,
                f"Table 5: Avg kurtosis ({processed} services), K=1..{args.max_k}.",
                args.max_k)
    print_table(corr_table,
                f"Table 6: Avg correlation with D1 ({processed} services), K=1..{args.max_k}.",
                args.max_k)

    optimal_k, analysis = find_optimal_k(center_table, kurt_table, corr_table, args.max_k)

    print(f"\n{'=' * 60}")
    print(f"OPTIMAL K SELECTION RESULT  ({processed} services)")
    print(f"{'=' * 60}")
    print(f"  Over-decomposition K : {analysis['over_decomp_k']}")
    print(f"  Best kurtosis K      : {analysis['best_kurtosis_k']}")
    print(f"  Correlation decay K  : {analysis['correlation_decay_k']}")
    if analysis['reasons']:
        print(f"\n  Key findings:")
        for r in analysis['reasons']:
            print(f"    - {r}")
    print(f"\n  >>> Optimal K = {optimal_k} <<<")
    print(f"{'=' * 60}")

    # Cleanup work dir
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

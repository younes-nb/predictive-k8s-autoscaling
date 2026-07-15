"""
Decomposition Analysis: SWT Energy Distribution & SVMD Mode Count

Phase 0: Create windows for each input size (16, 32, 64, 128) and cache to disk.
         Also cache D1 (SWT level=1) for use in Phase 2.
Phase 1: For 16 cases (4 input sizes x 4 SWT levels), compute energy of every
         component and average across all samples.
Phase 2: For each input size, run SWT level=1 to get D1, then run successive VMD
         (SVMD) on D1. Report mode count statistics (AVG, P50, P95, Min, Max).

Both CPU and Memory utilisation are processed independently.

Parallelisation follows the same subprocess-worker pattern as
``preprocessing/sv/preprocess.py``: ``ThreadPoolExecutor`` + ``subprocess.Popen``
with 90 % of CPU cores by default.

SVMD implementation based on:
  Nazari & Sakhaei, "Successive variational mode decomposition",
  Signal Processing 174 (2020) 107610.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS
from shared.config_preprocessing_defaults import PREPROCESSING

METRIC_COLUMNS = {
    "cpu": "cpu_utilization",
    "memory": "memory_utilization",
}

_WINDOW_WORKER = os.path.join(THIS_DIR, "_window_worker.py")
_ENERGY_WORKER = os.path.join(THIS_DIR, "_energy_worker.py")
_SVMD_WORKER = os.path.join(THIS_DIR, "_svmd_worker.py")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _TehranFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.getMessage()}"


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "analysis.log")
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
    logging.getLogger("pywt").setLevel(logging.WARNING)
    logging.info("Log file: %s", log_path)


def _worker_env() -> dict[str, str]:
    """Environment that pins each subprocess to a single thread."""
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT", "JOBLIB_NUM_THREADS"):
        env[var] = "1"
    return env


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_signals(
    msresource_dir: str, max_services: int = 0,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load signals from parquet for requested metrics.

    Returns
    -------
    dict mapping metric_key ("cpu" / "memory") -> {service_name: ndarray}.
    """
    if metrics is None:
        metrics = ["cpu"]

    all_columns = [METRIC_COLUMNS[m] for m in metrics]

    parts = sorted(
        p for p in os.listdir(msresource_dir)
        if p.endswith(".parquet") and p.startswith("part-")
    )
    parts = [os.path.join(msresource_dir, p) for p in parts]
    logging.info("Found %d parquet parts in %s", len(parts), msresource_dir)

    result: dict[str, dict[str, np.ndarray]] = {m: {} for m in metrics}

    t0 = time.time()
    for part_idx, part_path in enumerate(parts):
        if max_services and all(len(result[m]) >= max_services for m in metrics):
            break

        logging.info("  Reading part %d/%d ...", part_idx + 1, len(parts))

        svc_names = (
            pl.scan_parquet(part_path)
            .select("msname").unique()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        svc_names.sort()

        if not max_services:
            for pp in parts[part_idx + 1:]:
                more = (
                    pl.scan_parquet(pp)
                    .select("msname").unique()
                    .collect(engine="streaming")
                    .to_series()
                    .to_list()
                )
                svc_names = sorted(set(svc_names) | set(more))
            logging.info("Total unique services across all parts: %d", len(svc_names))

        needed = [s for s in svc_names
                  if not max_services or any(len(result[m]) < max_services for m in metrics)]
        if max_services:
            needed = needed[:max_services * 2]

        if not needed:
            continue

        df = (
            pl.scan_parquet(part_path)
            .filter(pl.col("msname").is_in(needed))
            .group_by("msname", "timestamp")
            .agg([pl.col(col).mean() for col in all_columns])
            .sort("msname", "timestamp")
            .collect(engine="streaming")
        )

        for key, group in df.group_by("msname", maintain_order=True):
            ms_name = key[0] if isinstance(key, tuple) else key
            all_ok = True
            for m in metrics:
                col = METRIC_COLUMNS[m]
                sig = group[col].to_numpy().astype(np.float32)
                if len(sig) < 16:
                    all_ok = False
                    break
                result[m][ms_name] = sig
            if not all_ok:
                for m in metrics:
                    result[m].pop(ms_name, None)

        counts = ", ".join(f"{m}={len(result[m])}" for m in metrics)
        logging.info("  Loaded %s (%.1fs)", counts, time.time() - t0)

        if max_services and all(len(result[m]) >= max_services for m in metrics):
            break

    if max_services:
        for m in metrics:
            result[m] = dict(sorted(result[m].items())[:max_services])

    for m in metrics:
        logging.info("Total %s signals: %d", m, len(result[m]))

    return result


# ---------------------------------------------------------------------------
# Phase 0 - Window preparation + D1 caching (parallelised)
# ---------------------------------------------------------------------------

def phase0_prepare_windows(
    all_signals: dict[str, np.ndarray],
    out_dir: str,
    input_sizes: list[int],
    stride: int = PREPROCESSING.STRIDE,
    num_workers: int = 1,
) -> dict[int, dict[str, np.ndarray]]:
    """Create windows for each input size using parallel subprocess workers.

    1. Save raw signals to ``{out_dir}/original/``.
    2. Dispatch ``_window_worker.py`` per service via ThreadPoolExecutor.
    3. Wait for completion, verify outputs.

    Returns
    -------
    dict mapping input_size -> {service_name: ndarray(n_windows, input_size)}
    """
    services = sorted(all_signals.keys())
    svc_to_idx = {name: i for i, name in enumerate(services)}

    idx_path = os.path.join(out_dir, "_svc_to_idx.json")
    with open(idx_path, "w") as f:
        json.dump(svc_to_idx, f)
    names_path = os.path.join(out_dir, "_service_names.npy")
    np.save(names_path, np.array(services, dtype=object))

    # --- save signals to disk ---
    orig_dir = os.path.join(out_dir, "original")
    os.makedirs(orig_dir, exist_ok=True)
    for svc_name in services:
        idx = svc_to_idx[svc_name]
        npy_path = os.path.join(orig_dir, f"service_{idx:05d}.npy")
        if not os.path.exists(npy_path):
            np.save(npy_path, all_signals[svc_name])

    # --- identify which services still need processing ---
    input_sizes_json = json.dumps(input_sizes)
    worker_args: list[tuple[str, int]] = []
    for svc_name in services:
        idx = svc_to_idx[svc_name]
        # check if ANY input size is missing windows
        needs_work = False
        for sz in input_sizes:
            win_path = os.path.join(out_dir, f"windows_{sz}", f"service_{idx:05d}.npy")
            if not os.path.exists(win_path):
                sig = all_signals[svc_name]
                if len(sig) >= sz:
                    needs_work = True
                    break
        if needs_work:
            worker_args.append((svc_name, idx))

    logging.info("  Phase 0: %d services to process, %d workers",
                 len(worker_args), num_workers)

    if not worker_args:
        logging.info("  All windows already cached.")
    else:
        t0 = time.time()
        done_count = 0
        failed = 0

        def _run(svc_name: str, idx: int) -> tuple[int, str, str, float]:
            proc = subprocess.Popen(
                [sys.executable, _WINDOW_WORKER, svc_name, str(idx),
                 out_dir, input_sizes_json, str(stride)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_worker_env(),
            )
            t_start = time.time()
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode(), time.time() - t_start

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            fut_map = {
                pool.submit(_run, sn, idx): (sn, idx)
                for sn, idx in worker_args
            }
            for future in as_completed(fut_map):
                svc_name, idx = fut_map[future]
                rc, out, err, dur = future.result()
                done_count += 1
                if rc == 0 and "RESULT:True" in out:
                    status = "ok"
                else:
                    failed += 1
                    # extract error message
                    status = "FAILED"
                    for line in out.strip().split("\n"):
                        if line.startswith("RESULT:False:ERROR:"):
                            status = line.split(":", 2)[-1].strip()[:120]
                            break
                logging.info("  [%s] %d/%d  %s  (%.1fs)",
                             svc_name, done_count, len(worker_args), status, dur)

        elapsed = time.time() - t0
        logging.info("  Phase 0 workers done: %d ok, %d failed (%.1fs)",
                     done_count - failed, failed, elapsed)

    # --- load results into memory ---
    result: dict[int, dict[str, np.ndarray]] = {sz: {} for sz in input_sizes}
    for svc_name in services:
        idx = svc_to_idx[svc_name]
        for sz in input_sizes:
            wpath = os.path.join(out_dir, f"windows_{sz}", f"service_{idx:05d}.npy")
            if os.path.exists(wpath):
                result[sz][svc_name] = np.load(wpath)

    for sz in input_sizes:
        logging.info("  Loaded %d services for input_size=%d", len(result[sz]), sz)

    return result


# ---------------------------------------------------------------------------
# Phase 1 - SWT Energy Analysis
# ---------------------------------------------------------------------------

def phase1_swt_energy(
    out_dir: str,
    swt_levels: list[int],
    input_sizes: list[int],
    metric_label: str,
    num_workers: int = 1,
) -> None:
    """Compute energy of every SWT component using parallel subprocess workers."""
    svc_idx_path = os.path.join(out_dir, "_svc_to_idx.json")
    with open(svc_idx_path) as f:
        svc_to_idx = json.load(f)
    services = sorted(svc_to_idx.keys())

    input_sizes_json = json.dumps(input_sizes)
    swt_levels_json = json.dumps(swt_levels)

    # accumulators keyed by (input_size, level)
    acc: dict[tuple[int, int], dict] = {}
    for sz in input_sizes:
        for lv in swt_levels:
            acc[(sz, lv)] = {
                "energies": np.zeros(lv + 1, dtype=np.float64),
                "total": 0.0,
                "count": 0,
            }

    logging.info("  Phase 1: processing %d services with %d workers...", len(services), num_workers)
    t0 = time.time()

    def _run(svc_name: str, idx: int) -> tuple[int, str, str, float]:
        proc = subprocess.Popen(
            [sys.executable, _ENERGY_WORKER, svc_name, str(idx),
             out_dir, input_sizes_json, swt_levels_json],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_worker_env(),
        )
        t_start = time.time()
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode(), time.time() - t_start

    worker_args = [(svc_name, svc_to_idx[svc_name]) for svc_name in services]
    done_count = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        fut_map = {
            pool.submit(_run, sn, idx): (sn, idx)
            for sn, idx in worker_args
        }
        for future in as_completed(fut_map):
            svc_name, idx = fut_map[future]
            rc, out, err, dur = future.result()
            done_count += 1

            if rc == 0 and "RESULT:True" in out:
                for line in out.strip().split("\n"):
                    if line.startswith("RESULT:True:"):
                        payload = line.split(":", 2)[-1]
                        data = json.loads(payload)
                        for key_str, vals in data.items():
                            parts = key_str.split("_")
                            inp_sz, lvl = int(parts[0]), int(parts[1])
                            k = (inp_sz, lvl)
                            n_comp = lvl + 1
                            acc[k]["energies"] += np.array(vals["energies"][:n_comp], dtype=np.float64)
                            acc[k]["total"] += vals["total"]
                            acc[k]["count"] += vals["count"]
                        break
            else:
                failed += 1

            if done_count % max(1, len(worker_args) // 10) == 0 or done_count == len(worker_args):
                logging.info("    %d/%d services done (%.1fs)",
                             done_count, len(worker_args), time.time() - t0)

    elapsed = time.time() - t0
    logging.info("  Phase 1 workers done: %d ok, %d failed (%.1fs)",
                 done_count - failed, failed, elapsed)

    # --- aggregate and print ---
    results: list[dict] = []
    for input_size in input_sizes:
        for level in swt_levels:
            k = (input_size, level)
            data = acc[k]
            n_windows = data["count"]
            if n_windows == 0:
                continue

            comp_names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
            n_comp = level + 1
            avg_energies = data["energies"] / n_windows
            avg_total = data["total"] / n_windows

            logging.info("    size=%d  level=%d  windows=%d  E_total=%.4f",
                         input_size, level, n_windows, avg_total)

            for ci, cname in enumerate(comp_names):
                pct = avg_energies[ci] / avg_total * 100 if avg_total > 0 else 0
                results.append({
                    "input_size": input_size,
                    "level": level,
                    "component": cname,
                    "avg_energy": float(avg_energies[ci]),
                    "avg_energy_pct": float(pct),
                })

    print("\n" + "=" * 80)
    print(f"PHASE 1: SWT Energy Distribution [{metric_label.upper()}]")
    print("=" * 80)
    header = f"{'input_size':>10}  {'level':>5}  {'component':>10}  {'avg_energy':>14}  {'avg_energy_pct':>15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['input_size']:>10}  {r['level']:>5}  {r['component']:>10}"
              f"  {r['avg_energy']:>14.6f}  {r['avg_energy_pct']:>14.2f}%")

    csv_path = os.path.join(out_dir, f"phase1_swt_energy_{metric_label}.csv")
    with open(csv_path, "w") as f:
        f.write("input_size,level,component,avg_energy,avg_energy_pct\n")
        for r in results:
            f.write(f"{r['input_size']},{r['level']},{r['component']},"
                    f"{r['avg_energy']:.8f},{r['avg_energy_pct']:.4f}\n")
    logging.info("  Phase 1 results saved to %s", csv_path)


# ---------------------------------------------------------------------------
# Phase 2 - SWT(1) + SVMD Mode Count (parallelised)
# ---------------------------------------------------------------------------

def phase2_svmd_mode_count(
    out_dir: str,
    input_sizes: list[int],
    svmd_alpha: float,
    svmd_tau: float,
    svmd_tol: float,
    svmd_max_modes: int,
    svmd_stop_power_ratio: float,
    metric_label: str,
    num_workers: int = 1,
) -> None:
    """Load cached D1, run SVMD via parallel workers, report mode count statistics."""
    results: list[dict] = []

    for input_size in input_sizes:
        d1_dir = os.path.join(out_dir, f"d1_{input_size}")
        if not os.path.isdir(d1_dir):
            logging.warning("  D1 directory not found for input_size=%d — skipping", input_size)
            continue

        npy_files = sorted(f for f in os.listdir(d1_dir) if f.endswith(".npy"))
        if not npy_files:
            logging.warning("  No D1 files for input_size=%d — skipping", input_size)
            continue

        logging.info("  SVMD for input_size=%d (%d services, %d workers)...",
                     input_size, len(npy_files), num_workers)
        t0 = time.time()

        # --- dispatch SVMD workers ---
        mode_counts: list[int] = []

        def _run_svmd(fname: str) -> tuple[int, str, str, float]:
            svc_idx = int(fname.replace("service_", "").replace(".npy", ""))
            proc = subprocess.Popen(
                [sys.executable, _SVMD_WORKER, str(svc_idx), d1_dir,
                 str(input_size), str(svmd_alpha), str(svmd_tau),
                 str(svmd_tol), str(svmd_max_modes), str(svmd_stop_power_ratio)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_worker_env(),
            )
            t_start = time.time()
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode(), time.time() - t_start

        done_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            fut_map = {
                pool.submit(_run_svmd, fname): fname
                for fname in npy_files
            }
            for future in as_completed(fut_map):
                fname = fut_map[future]
                rc, out, err, dur = future.result()
                done_count += 1
                if rc == 0 and "RESULT:True" in out:
                    for line in out.strip().split("\n"):
                        if line.startswith("RESULT:True:"):
                            payload = line.split(":", 2)[-1]
                            data = json.loads(payload)
                            mode_counts.extend(data["counts"])
                            break
                else:
                    logging.warning("  [%s] worker failed (exit=%d)", fname, rc)

                if done_count % max(1, len(npy_files) // 10) == 0 or done_count == len(npy_files):
                    logging.info("    %d/%d services done (%.1fs)",
                                 done_count, len(npy_files), time.time() - t0)

        if not mode_counts:
            logging.warning("  No valid D1 windows for input_size=%d", input_size)
            continue

        mc = np.array(mode_counts, dtype=np.int32)
        stats = {
            "input_size": input_size,
            "n_samples": len(mc),
            "avg": float(np.mean(mc)),
            "p50": float(np.percentile(mc, 50)),
            "p95": float(np.percentile(mc, 95)),
            "min": int(np.min(mc)),
            "max": int(np.max(mc)),
        }
        results.append(stats)

        logging.info("    size=%d: n=%d avg=%.2f p50=%.0f p95=%.0f min=%d max=%d (%.1fs)",
                     input_size, stats["n_samples"], stats["avg"],
                     stats["p50"], stats["p95"], stats["min"], stats["max"],
                     time.time() - t0)

    print("\n" + "=" * 80)
    print(f"PHASE 2: SVMD Mode Count [{metric_label.upper()}] (SWT level=1 -> D1 -> SVMD)")
    print("=" * 80)
    header = f"{'input_size':>10}  {'n_samples':>10}  {'avg':>8}  {'p50':>6}  {'p95':>6}  {'min':>4}  {'max':>4}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['input_size']:>10}  {r['n_samples']:>10}  {r['avg']:>8.2f}"
              f"  {r['p50']:>6.0f}  {r['p95']:>6.0f}  {r['min']:>4}  {r['max']:>4}")

    csv_path = os.path.join(out_dir, f"phase2_svmd_mode_count_{metric_label}.csv")
    with open(csv_path, "w") as f:
        f.write("input_size,n_samples,avg,p50,p95,min,max\n")
        for r in results:
            f.write(f"{r['input_size']},{r['n_samples']},{r['avg']:.4f},"
                    f"{r['p50']:.1f},{r['p95']:.1f},{r['min']},{r['max']}\n")
    logging.info("  Phase 2 results saved to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SWT energy analysis & SVMD mode count across input sizes.",
    )

    # data
    ap.add_argument("--msresource_dir", default=PATHS.PARQUET_MSRESOURCE)
    ap.add_argument("--out_dir", default="/dataset/decomp_analysis")
    ap.add_argument("--max_services", type=int, default=0,
                    help="Max services to use (0 = all).")
    ap.add_argument("--metrics", nargs="+", default=["cpu", "memory"],
                    choices=list(METRIC_COLUMNS.keys()),
                    help="Metrics to analyse (default: cpu memory).")

    # analysis grid
    ap.add_argument("--input_sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--swt_levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--stride", type=int, default=PREPROCESSING.STRIDE,
                    help=f"Window stride (default: {PREPROCESSING.STRIDE}).")

    # SVMD parameters
    ap.add_argument("--svmd_alpha", type=float, default=2000.0)
    ap.add_argument("--svmd_tau", type=float, default=0.0)
    ap.add_argument("--svmd_tol", type=float, default=1e-7)
    ap.add_argument("--svmd_max_modes", type=int, default=20)
    ap.add_argument("--svmd_stop_power_ratio", type=float, default=1e-6)

    # parallelism
    ap.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores for workers (default: 0.9).")

    # control
    ap.add_argument("--skip_window_prep", action="store_true",
                    help="Skip Phase 0 — reuse cached windows and D1.")
    ap.add_argument("--skip_phase1", action="store_true")
    ap.add_argument("--skip_phase2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    np.random.default_rng(args.seed)

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))

    setup_logging(args.out_dir)
    logging.info("Args: %s", vars(args))
    logging.info("CPUs: %d | Workers: %d", n_cpus, num_workers)

    # ---- Phase 0: load signals for all requested metrics ----
    all_signals_by_metric: dict[str, dict[str, np.ndarray]] = {}
    if not args.skip_window_prep:
        logging.info("=" * 60)
        logging.info("PHASE 0: Loading signals")
        logging.info("=" * 60)
        all_signals_by_metric = load_all_signals(
            args.msresource_dir, args.max_services, metrics=args.metrics,
        )
        for m in args.metrics:
            if not all_signals_by_metric.get(m):
                logging.error("No %s signals loaded.", m)

    # ---- process each metric ----
    for metric in args.metrics:
        metric_out = os.path.join(args.out_dir, metric)
        os.makedirs(metric_out, exist_ok=True)

        logging.info("")
        logging.info("#" * 60)
        logging.info("# METRIC: %s", metric.upper())
        logging.info("#" * 60)

        # Phase 0
        if not args.skip_window_prep:
            logging.info("PHASE 0: Window preparation [%s]", metric)
            sigs = all_signals_by_metric.get(metric, {})
            if not sigs:
                logging.warning("  No %s signals — skipping metric.", metric)
                continue
            window_data = phase0_prepare_windows(
                sigs, metric_out, args.input_sizes,
                stride=args.stride, num_workers=num_workers,
            )
        else:
            logging.info("PHASE 0: loading cached windows [%s]", metric)
            window_data = {}
            svc_idx_path = os.path.join(metric_out, "_svc_to_idx.json")
            if not os.path.exists(svc_idx_path):
                logging.warning("  No cached index for %s — skipping.", metric)
                continue
            with open(svc_idx_path) as f:
                svc_to_idx = json.load(f)
            services = sorted(svc_to_idx.keys())
            for sz in args.input_sizes:
                win_dir = os.path.join(metric_out, f"windows_{sz}")
                svc_map: dict[str, np.ndarray] = {}
                if os.path.isdir(win_dir):
                    for svc_name in services:
                        idx = svc_to_idx[svc_name]
                        wpath = os.path.join(win_dir, f"service_{idx:05d}.npy")
                        if os.path.exists(wpath):
                            svc_map[svc_name] = np.load(wpath)
                window_data[sz] = svc_map
                logging.info("  Loaded %d services for input_size=%d", len(svc_map), sz)

        # Phase 1
        if not args.skip_phase1:
            logging.info("PHASE 1: SWT Energy [%s]", metric)
            phase1_swt_energy(
                metric_out, args.swt_levels, args.input_sizes, metric,
                num_workers=num_workers,
            )
        else:
            logging.info("PHASE 1: skipped [%s]", metric)

        # Phase 2
        if not args.skip_phase2:
            logging.info("PHASE 2: SVMD Mode Count [%s]", metric)
            phase2_svmd_mode_count(
                metric_out, args.input_sizes,
                svmd_alpha=args.svmd_alpha,
                svmd_tau=args.svmd_tau,
                svmd_tol=args.svmd_tol,
                svmd_max_modes=args.svmd_max_modes,
                svmd_stop_power_ratio=args.svmd_stop_power_ratio,
                metric_label=metric,
                num_workers=num_workers,
            )
        else:
            logging.info("PHASE 2: skipped [%s]", metric)

    logging.info("Done.")


if __name__ == "__main__":
    main()

"""
Decomposition Analysis: SWT Energy Distribution & SVMD Mode Count

Phase 0: Create windows for each input size (16, 32, 64, 128) and cache to disk.
         Also cache D1 (SWT level=1) for use in Phase 2.
Phase 1: For 16 cases (4 input sizes x 4 SWT levels), compute energy of every
         component and average across all samples.
Phase 2: For each input size, run SWT level=1 to get D1, then run successive VMD
         (SVMD) on D1. Report mode count statistics (AVG, P50, P95, Min, Max).

Both CPU and Memory utilisation are processed independently.

SVMD implementation based on:
  Nazari & Sakhaei, "Successive variational mode decomposition",
  Signal Processing 174 (2020) 107610.
"""

import argparse
import json
import logging
import os
import sys
import time
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


# ---------------------------------------------------------------------------
# SVMD - Successive Variational Mode Decomposition
# (Nazari & Sakhaei, Signal Processing 174, 2020)
# ---------------------------------------------------------------------------

def svmd(
    signal: np.ndarray,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_modes: int = 20,
    stop_power_ratio: float = 1e-6,
    dc: int = 0,
    init: int = 1,
) -> np.ndarray:
    """Successive VMD - extract modes one at a time from the residual.

    Uses ``vmdpy.VMD`` with K=1 per iteration.  After each mode is extracted
    it is subtracted from the signal and the next mode is extracted from the
    residual.

    Returns
    -------
    modes : ndarray, shape (n_modes, T)
    """
    from vmdpy import VMD
    import warnings as _warnings

    signal = np.asarray(signal, dtype=np.float64)
    orig_len = len(signal)

    if orig_len < 4:
        return signal[np.newaxis, :]

    pad = orig_len % 2
    if pad:
        signal = np.append(signal, signal[-1])

    eps = np.finfo(np.float64).eps
    signal_power = float(np.sum(signal ** 2))
    residual = signal.copy()
    modes_list: list[np.ndarray] = []

    for k in range(max_modes):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                u, _, _ = VMD(residual, alpha, tau, K=1, DC=dc, init=init, tol=tol)
            except Exception:
                break

        mode = u[0]

        if pad:
            mode = mode[:-1]

        mode_power = float(np.sum(mode ** 2))
        if k > 0 and mode_power / (signal_power + eps) < stop_power_ratio:
            break

        modes_list.append(mode)
        residual = residual - u[0]

    if not modes_list:
        return signal[:orig_len][np.newaxis, :]

    return np.stack(modes_list, axis=0)


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

    # list parquet parts
    parts = sorted(
        p for p in os.listdir(msresource_dir)
        if p.endswith(".parquet") and p.startswith("part-")
    )
    parts = [os.path.join(msresource_dir, p) for p in parts]
    logging.info("Found %d parquet parts in %s", len(parts), msresource_dir)

    result: dict[str, dict[str, np.ndarray]] = {m: {} for m in metrics}

    # Scan parts one at a time until we have enough services
    t0 = time.time()
    for part_idx, part_path in enumerate(parts):
        if max_services and all(len(result[m]) >= max_services for m in metrics):
            break

        logging.info("  Reading part %d/%d ...", part_idx + 1, len(parts))

        # Get service names from this part
        svc_names = (
            pl.scan_parquet(part_path)
            .select("msname").unique()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        svc_names.sort()

        # If not limiting, collect all names first across all parts
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

        # Filter to services we still need
        needed = [s for s in svc_names
                  if not max_services or any(len(result[m]) < max_services for m in metrics)]
        if max_services:
            needed = needed[:max_services * 2]  # read a bit extra in case some are too short

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
# Phase 0 - Window preparation + D1 caching
# ---------------------------------------------------------------------------

def phase0_prepare_windows(
    all_signals: dict[str, np.ndarray],
    out_dir: str,
    input_sizes: list[int],
    stride: int = PREPROCESSING.STRIDE,
) -> dict[int, dict[str, np.ndarray]]:
    """Create windows for each input size, save to disk, and cache D1.

    Returns
    -------
    dict mapping input_size -> {service_name: ndarray(n_windows, input_size)}
    """
    import pywt

    services = sorted(all_signals.keys())
    svc_to_idx = {name: i for i, name in enumerate(services)}

    idx_path = os.path.join(out_dir, "_svc_to_idx.json")
    with open(idx_path, "w") as f:
        json.dump(svc_to_idx, f)
    names_path = os.path.join(out_dir, "_service_names.npy")
    np.save(names_path, np.array(services, dtype=object))

    result: dict[int, dict[str, np.ndarray]] = {sz: {} for sz in input_sizes}

    for input_size in input_sizes:
        win_dir = os.path.join(out_dir, f"windows_{input_size}")
        d1_dir = os.path.join(out_dir, f"d1_{input_size}")
        os.makedirs(win_dir, exist_ok=True)
        os.makedirs(d1_dir, exist_ok=True)

        done_marker = os.path.join(win_dir, ".done")
        if os.path.exists(done_marker):
            logging.info("  windows_%d cached — loading from disk", input_size)
            for svc_name in services:
                idx = svc_to_idx[svc_name]
                wpath = os.path.join(win_dir, f"service_{idx:05d}.npy")
                if os.path.exists(wpath):
                    result[input_size][svc_name] = np.load(wpath)
            continue

        logging.info("  Creating windows for input_size=%d (stride=%d)...", input_size, stride)
        t0 = time.time()
        count = 0

        for svc_name in services:
            sig = all_signals[svc_name]
            if len(sig) < input_size:
                continue

            T = len(sig) - input_size + 1
            windows = []
            d1s = []
            for i in range(0, T, stride):
                w = sig[i: i + input_size].astype(np.float64)
                windows.append(w.astype(np.float32))

                coeffs = pywt.swt(w, "sym4", level=1, norm=True, trim_approx=True)
                _, d1 = coeffs
                d1s.append(d1.astype(np.float32))

            if not windows:
                continue

            idx = svc_to_idx[svc_name]
            np.save(os.path.join(win_dir, f"service_{idx:05d}.npy"),
                    np.stack(windows, axis=0))
            np.save(os.path.join(d1_dir, f"service_{idx:05d}.npy"),
                    np.stack(d1s, axis=0))
            result[input_size][svc_name] = np.stack(windows, axis=0)
            count += 1

        open(done_marker, "a").close()
        logging.info("  input_size=%d: %d services (%.1fs)", input_size, count, time.time() - t0)

    return result


# ---------------------------------------------------------------------------
# Phase 1 - SWT Energy Analysis
# ---------------------------------------------------------------------------

def phase1_swt_energy(
    window_data: dict[int, dict[str, np.ndarray]],
    out_dir: str,
    swt_levels: list[int],
    input_sizes: list[int],
    metric_label: str,
) -> None:
    """Compute energy of every SWT component for all (input_size, level) cases."""
    import pywt

    results: list[dict] = []

    for input_size in input_sizes:
        svc_windows = window_data[input_size]
        if not svc_windows:
            logging.warning("  No windows for input_size=%d — skipping", input_size)
            continue

        for level in swt_levels:
            comp_names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
            n_comp = level + 1

            energy_sums = np.zeros(n_comp, dtype=np.float64)
            total_energy_sum = 0.0
            n_windows = 0

            for svc_name, win_arr in svc_windows.items():
                for wi in range(win_arr.shape[0]):
                    window = win_arr[wi].astype(np.float64)
                    if np.std(window) < 1e-12:
                        continue

                    coeffs = pywt.swt(window, "sym4", level=level, norm=True, trim_approx=True)
                    assert len(coeffs) == n_comp

                    energies = np.array([float(np.sum(c ** 2)) for c in coeffs])
                    e_total = float(np.sum(energies))

                    energy_sums += energies
                    total_energy_sum += e_total
                    n_windows += 1

            if n_windows == 0:
                continue

            avg_energies = energy_sums / n_windows
            avg_total = total_energy_sum / n_windows

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

    # print
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
# Phase 2 - SWT(1) + SVMD Mode Count
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
) -> None:
    """Load cached D1, run SVMD on each, report mode count statistics."""
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

        logging.info("  SVMD for input_size=%d (%d services)...", input_size, len(npy_files))
        t0 = time.time()

        mode_counts: list[int] = []

        for fname in npy_files:
            d1_arr = np.load(os.path.join(d1_dir, fname)).astype(np.float64)
            for wi in range(d1_arr.shape[0]):
                d1 = d1_arr[wi]
                if np.std(d1) < 1e-12:
                    continue
                modes = svmd(
                    d1,
                    alpha=svmd_alpha,
                    tau=svmd_tau,
                    tol=svmd_tol,
                    max_modes=svmd_max_modes,
                    stop_power_ratio=svmd_stop_power_ratio,
                )
                mode_counts.append(modes.shape[0])

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

    # control
    ap.add_argument("--skip_window_prep", action="store_true",
                    help="Skip Phase 0 — reuse cached windows and D1.")
    ap.add_argument("--skip_phase1", action="store_true")
    ap.add_argument("--skip_phase2", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    np.random.default_rng(args.seed)

    setup_logging(args.out_dir)
    logging.info("Args: %s", vars(args))

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
                sigs, metric_out, args.input_sizes, stride=args.stride,
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
                window_data, metric_out, args.swt_levels, args.input_sizes, metric,
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
            )
        else:
            logging.info("PHASE 2: skipped [%s]", metric)

    logging.info("Done.")


if __name__ == "__main__":
    main()

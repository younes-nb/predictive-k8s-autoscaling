import gc
import os
import sys
import argparse
from datetime import datetime

import duckdb
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths, PREPROCESSING

DEFAULT_BINS = 128
CPU_RANGE = (0.0, 1.0)
MEMORY_RANGE = (0.0, 1.0)
CORR_RANGE = (-1.0, 1.0)

INPUT_LEN = PREPROCESSING.INPUT_LEN
PRED_HORIZON = 5
STRIDE = PREPROCESSING.STRIDE
BATCH_SIZE = 256


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CPU and memory utilization from parquet files."
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=Paths.PARQUET_MSRESOURCE,
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=Paths.LOGS_DIR,
        help="Directory to save plots.",
    )
    return parser.parse_args()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    xs = x - x.mean()
    ys = y - y.mean()
    denom = np.sqrt((xs**2).sum()) * np.sqrt((ys**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def _bin_index(value: float, lo: float, hi: float, n_bins: int) -> int:
    v = max(lo, min(hi, float(value)))
    idx = int((v - lo) / (hi - lo) * n_bins)
    if idx == n_bins:
        idx -= 1
    return idx


def plot_histogram(bin_edges, counts, title, xlabel, out_path, color="steelblue"):
    if np.sum(counts) == 0:
        log(f"No data for {title}, skipping plot.")
        return
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]
    plt.figure(figsize=(10, 6))
    plt.bar(centers, counts, width=width, color=color, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Sample Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Saved plot: {out_path}")
    plt.close()


def main():
    args = parse_args()
    parquet_pattern = os.path.join(args.parquet_dir, "*.parquet")
    n_bins = args.bins
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '20GB'")

    overlap = min(INPUT_LEN, PRED_HORIZON)

    # ---- Phase 1: Max/Min per MS (average per timestamp) ----
    log("Phase 1: Computing max/min per microservice (averaged per timestamp)...")
    max_min_df = con.execute(f"""
        WITH per_timestamp AS (
            SELECT
                msname,
                timestamp,
                AVG(cpu_utilization) AS cpu,
                AVG(memory_utilization) AS mem
            FROM read_parquet('{parquet_pattern}')
            WHERE cpu_utilization BETWEEN 0.0 AND 1.0
              AND memory_utilization BETWEEN 0.0 AND 1.0
            GROUP BY msname, timestamp
        )
        SELECT
            msname,
            MAX(cpu) AS max_cpu,
            MIN(cpu) AS min_cpu,
            MAX(mem) AS max_mem,
            MIN(mem) AS min_mem
        FROM per_timestamp
        GROUP BY msname
        ORDER BY msname
    """).df()
    n_ms = len(max_min_df)
    log(f"Found {n_ms} unique microservices.")

    # ---- Phase 2: Window-based statistics ----
    log("Phase 2: Computing window-based statistics...")
    msnames = max_min_df["msname"].tolist()

    cpu_hist = np.zeros(n_bins, dtype=np.int64)
    mem_hist = np.zeros(n_bins, dtype=np.int64)
    cpu_corr_hist = np.zeros(n_bins, dtype=np.int64)
    mem_corr_hist = np.zeros(n_bins, dtype=np.int64)

    total_cpu_sum = 0.0
    total_cpu_n = 0
    total_mem_sum = 0.0
    total_mem_n = 0
    cpu_corr_sum = 0.0
    cpu_corr_count = 0
    mem_corr_sum = 0.0
    mem_corr_count = 0
    total_windows = 0

    n_batches = (len(msnames) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(msnames), BATCH_SIZE):
        batch = msnames[i:i + BATCH_SIZE]
        in_clause = ", ".join([f"'{m}'" for m in batch])
        batch_num = i // BATCH_SIZE + 1
        log(f"Batch {batch_num}/{n_batches}: {len(batch)} msnames...")

        instance_data = con.execute(f"""
            SELECT msname, msinstanceid,
                   LIST(cpu_utilization ORDER BY timestamp_dt) AS cpu_vals,
                   LIST(memory_utilization ORDER BY timestamp_dt) AS mem_vals
            FROM read_parquet('{parquet_pattern}')
            WHERE msname IN ({in_clause})
              AND cpu_utilization BETWEEN 0.0 AND 1.0
              AND memory_utilization BETWEEN 0.0 AND 1.0
            GROUP BY msname, msinstanceid
            HAVING COUNT(*) >= {INPUT_LEN + PRED_HORIZON}
        """).fetchall()

        for msname, msinstanceid, cpu_list, mem_list in instance_data:
            cpu_arr = np.array(cpu_list, dtype=np.float64)
            mem_arr = np.array(mem_list, dtype=np.float64)
            n = len(cpu_arr)
            max_start = n - INPUT_LEN - PRED_HORIZON

            for s in range(0, max_start + 1, STRIDE):
                input_cpu = cpu_arr[s:s + INPUT_LEN]
                input_mem = mem_arr[s:s + INPUT_LEN]
                horizon_cpu = cpu_arr[s + INPUT_LEN:s + INPUT_LEN + PRED_HORIZON]
                horizon_mem = mem_arr[s + INPUT_LEN:s + INPUT_LEN + PRED_HORIZON]

                if np.any(~np.isfinite(input_cpu)) or np.any(~np.isfinite(horizon_cpu)):
                    continue

                total_cpu_sum += float(input_cpu.sum())
                total_cpu_n += INPUT_LEN
                sample_cpu = float(np.mean(input_cpu))
                cpu_hist[_bin_index(sample_cpu, *CPU_RANGE, n_bins)] += 1

                if np.all(np.isfinite(input_mem)) and np.all(np.isfinite(horizon_mem)):
                    total_mem_sum += float(input_mem.sum())
                    total_mem_n += INPUT_LEN
                    sample_mem = float(np.mean(input_mem))
                    mem_hist[_bin_index(sample_mem, *MEMORY_RANGE, n_bins)] += 1

                x_tail = input_cpu[-overlap:]
                y_tail = horizon_cpu[-overlap:]
                r_cpu = _pearson(x_tail, y_tail)
                if np.isfinite(r_cpu):
                    cpu_corr_sum += r_cpu
                    cpu_corr_hist[_bin_index(r_cpu, *CORR_RANGE, n_bins)] += 1
                    cpu_corr_count += 1

                if np.all(np.isfinite(input_mem)) and np.all(np.isfinite(horizon_mem)):
                    x_tail_m = input_mem[-overlap:]
                    y_tail_m = horizon_mem[-overlap:]
                    r_mem = _pearson(x_tail_m, y_tail_m)
                    if np.isfinite(r_mem):
                        mem_corr_sum += r_mem
                        mem_corr_hist[_bin_index(r_mem, *CORR_RANGE, n_bins)] += 1
                        mem_corr_count += 1

                total_windows += 1

            del cpu_arr, mem_arr, cpu_list, mem_list
            gc.collect()

    con.close()

    # ---- Compute summary statistics ----
    global_avg_cpu = total_cpu_sum / max(total_cpu_n, 1)
    global_avg_mem = total_mem_sum / max(total_mem_n, 1)
    avg_cpu_corr = cpu_corr_sum / max(cpu_corr_count, 1) if cpu_corr_count > 0 else float("nan")
    avg_mem_corr = mem_corr_sum / max(mem_corr_count, 1) if mem_corr_count > 0 else float("nan")
    avg_max_cpu = float(max_min_df["max_cpu"].mean())
    avg_min_cpu = float(max_min_df["min_cpu"].mean())
    avg_max_mem = float(max_min_df["max_mem"].mean())
    avg_min_mem = float(max_min_df["min_mem"].mean())

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("  CPU/MEMORY UTILIZATION SUMMARY")
    print("=" * 60)
    print(f"Parquet files:                  {parquet_pattern}")
    print(f"Unique microservices:           {n_ms}")
    print()
    print("  PER-SAMPLE STATISTICS (windowed):")
    print(f"  Total windows created:        {total_windows}")
    print(f"  Global avg CPU (all steps):   {global_avg_cpu:.6f}")
    print(f"  Global avg memory (all steps):{global_avg_mem:.6f}")
    print(f"  Samples with CPU correlation: {cpu_corr_count}")
    if cpu_corr_count > 0:
        print(f"  Avg CPU correlation (tail vs horizon): {avg_cpu_corr:.6f}")
    else:
        print("  Avg CPU correlation:              n/a")
    print(f"  Samples with mem correlation: {mem_corr_count}")
    if mem_corr_count > 0:
        print(f"  Avg mem correlation (tail vs horizon): {avg_mem_corr:.6f}")
    else:
        print("  Avg mem correlation:              n/a")
    print()
    print("  MAX VALUE ANALYSIS (per MS):")
    print(f"  Average max CPU:              {avg_max_cpu:.6f}")
    print(f"  Average max memory:           {avg_max_mem:.6f}")
    print()
    print("  MIN VALUE ANALYSIS (per MS):")
    print(f"  Average min CPU:              {avg_min_cpu:.6f}")
    print(f"  Average min memory:           {avg_min_mem:.6f}")
    print("=" * 60 + "\n")

    # ---- Save all plots ----
    cpu_bin_edges = np.linspace(*CPU_RANGE, n_bins + 1)
    mem_bin_edges = np.linspace(*MEMORY_RANGE, n_bins + 1)
    corr_bin_edges = np.linspace(*CORR_RANGE, n_bins + 1)

    plot_histogram(
        cpu_bin_edges, cpu_hist,
        "Per-Sample CPU Utilization Distribution",
        "Average CPU Utilization (across input window steps)",
        os.path.join(args.out_dir, f"cpu_utilization_hist_{ts}.png"),
        color="steelblue",
    )

    plot_histogram(
        mem_bin_edges, mem_hist,
        "Per-Sample Memory Utilization Distribution",
        "Average Memory Utilization (across input window steps)",
        os.path.join(args.out_dir, f"memory_utilization_hist_{ts}.png"),
        color="forestgreen",
    )

    if cpu_corr_count > 0:
        plot_histogram(
            corr_bin_edges, cpu_corr_hist,
            "Per-Sample CPU Correlation: Input Tail vs Horizon",
            "Correlation",
            os.path.join(args.out_dir, f"cpu_correlation_hist_{ts}.png"),
            color="orchid",
        )

    if mem_corr_count > 0:
        plot_histogram(
            corr_bin_edges, mem_corr_hist,
            "Per-Sample Memory Correlation: Input Tail vs Horizon",
            "Correlation",
            os.path.join(args.out_dir, f"memory_correlation_hist_{ts}.png"),
            color="orchid",
        )

    mm_bins = np.linspace(0.0, 1.0, n_bins + 1)
    plot_histogram(
        mm_bins,
        np.histogram(max_min_df["max_cpu"].to_numpy(), bins=n_bins, range=(0, 1))[0],
        "Max CPU per Microservice Distribution",
        "Max CPU Utilization",
        os.path.join(args.out_dir, f"max_cpu_hist_{ts}.png"),
        color="coral",
    )

    plot_histogram(
        mm_bins,
        np.histogram(max_min_df["max_mem"].to_numpy(), bins=n_bins, range=(0, 1))[0],
        "Max Memory per Microservice Distribution",
        "Max Memory Utilization",
        os.path.join(args.out_dir, f"max_memory_hist_{ts}.png"),
        color="coral",
    )

    plot_histogram(
        mm_bins,
        np.histogram(max_min_df["min_cpu"].to_numpy(), bins=n_bins, range=(0, 1))[0],
        "Min CPU per Microservice Distribution",
        "Min CPU Utilization",
        os.path.join(args.out_dir, f"min_cpu_hist_{ts}.png"),
        color="lightseagreen",
    )

    plot_histogram(
        mm_bins,
        np.histogram(max_min_df["min_mem"].to_numpy(), bins=n_bins, range=(0, 1))[0],
        "Min Memory per Microservice Distribution",
        "Min Memory Utilization",
        os.path.join(args.out_dir, f"min_memory_hist_{ts}.png"),
        color="lightseagreen",
    )

    log("Done.")


if __name__ == "__main__":
    main()

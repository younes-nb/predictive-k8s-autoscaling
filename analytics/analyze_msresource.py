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

from config.defaults import Paths

DEFAULT_BINS = 128


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
    con.execute("SET temp_directory = '/tmp'")

    log("Computing per-timestamp averages (across instances) for each microservice...")
    con.execute(f"""
        CREATE TEMP TABLE per_ts AS
        SELECT
            msname,
            timestamp,
            AVG(cpu_utilization) AS cpu,
            AVG(memory_utilization) AS mem,
            COUNT(DISTINCT msinstanceid) AS num_instances
        FROM read_parquet('{parquet_pattern}')
        WHERE cpu_utilization BETWEEN 0.0 AND 1.0
          AND memory_utilization BETWEEN 0.0 AND 1.0
        GROUP BY msname, timestamp
    """)

    cpu_hist_rows = con.execute(f"""
        SELECT CAST(LEAST(FLOOR(cpu * {n_bins}), {n_bins - 1}) AS INTEGER) AS bin_idx,
               COUNT(*) AS cnt
        FROM per_ts
        GROUP BY bin_idx
        ORDER BY bin_idx
    """).fetchall()

    mem_hist_rows = con.execute(f"""
        SELECT CAST(LEAST(FLOOR(mem * {n_bins}), {n_bins - 1}) AS INTEGER) AS bin_idx,
               COUNT(*) AS cnt
        FROM per_ts
        GROUP BY bin_idx
        ORDER BY bin_idx
    """).fetchall()

    global_avg_cpu, global_avg_mem = con.execute("""
        SELECT AVG(cpu), AVG(mem) FROM per_ts
    """).fetchone()

    max_min_df = con.execute("""
        SELECT msname,
               MAX(cpu) AS max_cpu,
               MIN(cpu) AS min_cpu,
               MAX(mem) AS max_mem,
               MIN(mem) AS min_mem
        FROM per_ts
        GROUP BY msname
        ORDER BY msname
    """).df()

    avg_inst_df = con.execute("""
        SELECT msname, AVG(num_instances) AS avg_instances_per_ts
        FROM per_ts
        GROUP BY msname
    """).df()
    n_ms = len(max_min_df)
    log(f"Found {n_ms} unique microservices.")

    log("Computing total unique instances per microservice across entire timeline...")
    total_inst_df = con.execute(f"""
        SELECT msname, COUNT(DISTINCT msinstanceid) AS total_instances
        FROM read_parquet('{parquet_pattern}')
        WHERE cpu_utilization BETWEEN 0.0 AND 1.0
          AND memory_utilization BETWEEN 0.0 AND 1.0
        GROUP BY msname
    """).df()
    con.close()

    cpu_counts = np.zeros(n_bins, dtype=np.int64)
    for bin_idx, cnt in cpu_hist_rows:
        cpu_counts[bin_idx] = cnt

    mem_counts = np.zeros(n_bins, dtype=np.int64)
    for bin_idx, cnt in mem_hist_rows:
        mem_counts[bin_idx] = cnt

    avg_max_cpu = float(max_min_df["max_cpu"].mean())
    avg_min_cpu = float(max_min_df["min_cpu"].mean())
    avg_max_mem = float(max_min_df["max_mem"].mean())
    avg_min_mem = float(max_min_df["min_mem"].mean())

    avg_inst_per_ms = float(avg_inst_df["avg_instances_per_ts"].mean())
    max_avg_inst = float(avg_inst_df["avg_instances_per_ts"].max())
    min_avg_inst = float(avg_inst_df["avg_instances_per_ts"].min())

    avg_total_inst = float(total_inst_df["total_instances"].mean())
    max_total_inst = int(total_inst_df["total_instances"].max())
    min_total_inst = int(total_inst_df["total_instances"].min())

    print("\n" + "=" * 60)
    print("  CPU/MEMORY UTILIZATION SUMMARY")
    print("=" * 60)
    print(f"Parquet files:                  {parquet_pattern}")
    print(f"Unique microservices:           {n_ms}")
    print()
    print("  PER-TIMESTAMP UTILIZATION (averaged across instances):")
    print(f"  Global avg CPU:               {global_avg_cpu:.6f}")
    print(f"  Global avg memory:            {global_avg_mem:.6f}")
    print()
    print("  INSTANCE COUNT ANALYSIS:")
    print("  Per-MS avg instances per timestamp:")
    print(f"    Average across MSs:         {avg_inst_per_ms:.3f}")
    print(f"    Max across MSs:             {max_avg_inst:.3f}")
    print(f"    Min across MSs:             {min_avg_inst:.3f}")
    print("  Per-MS total unique instances (entire timeline):")
    print(f"    Average across MSs:         {avg_total_inst:.3f}")
    print(f"    Max across MSs:             {max_total_inst}")
    print(f"    Min across MSs:             {min_total_inst}")
    print()
    print("  MAX VALUE ANALYSIS (per MS):")
    print(f"  Average max CPU:              {avg_max_cpu:.6f}")
    print(f"  Average max memory:           {avg_max_mem:.6f}")
    print()
    print("  MIN VALUE ANALYSIS (per MS):")
    print(f"  Average min CPU:              {avg_min_cpu:.6f}")
    print(f"  Average min memory:           {avg_min_mem:.6f}")
    print("=" * 60 + "\n")

    mm_bins = np.linspace(0.0, 1.0, n_bins + 1)
    plot_histogram(
        mm_bins, cpu_counts,
        "Per-Timestamp CPU Utilization Distribution",
        "Average CPU Utilization (across instances per timestamp)",
        os.path.join(args.out_dir, f"cpu_hist_{ts}.png"),
        color="steelblue",
    )

    plot_histogram(
        mm_bins, mem_counts,
        "Per-Timestamp Memory Utilization Distribution",
        "Average Memory Utilization (across instances per timestamp)",
        os.path.join(args.out_dir, f"memory_hist_{ts}.png"),
        color="forestgreen",
    )

    inst_counts = avg_inst_df["avg_instances_per_ts"].to_numpy()
    inst_hist, inst_edges = np.histogram(inst_counts, bins=n_bins)
    plot_histogram(
        inst_edges, inst_hist,
        "Per-MS Average Instances per Timestamp Distribution",
        "Average Number of Instances per Timestamp",
        os.path.join(args.out_dir, f"avg_instances_per_ts_hist_{ts}.png"),
        color="goldenrod",
    )

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

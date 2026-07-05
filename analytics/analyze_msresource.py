import os
import sys
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any

import duckdb
import numpy as np
import pandas as pd
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


def _compute_avg_instances_chunk(chunk_msnames: list, parquet_pattern: str) -> Dict[str, float]:
    """Worker: compute avg instances per timestamp for a chunk of msnames."""
    import time as _time

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET preserve_insertion_order = false")

    result = {}
    BATCH_SIZE = 256
    n_batches = (len(chunk_msnames) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunk_msnames), BATCH_SIZE):
        batch = chunk_msnames[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        t_start = _time.time()

        in_clause = ", ".join([f"'{m}'" for m in batch])
        rows = con.execute(f"""
            WITH per_timestamp AS (
                SELECT msname, timestamp,
                       COUNT(DISTINCT msinstanceid) AS num_instances
                FROM read_parquet('{parquet_pattern}')
                WHERE msname IN ({in_clause})
                  AND cpu_utilization BETWEEN 0.0 AND 1.0
                  AND memory_utilization BETWEEN 0.0 AND 1.0
                GROUP BY msname, timestamp
            )
            SELECT msname, AVG(num_instances) AS avg_instances_per_ts
            FROM per_timestamp
            GROUP BY msname
        """).fetchall()

        elapsed = _time.time() - t_start
        log(f"  Inst batch {batch_num}/{n_batches}: {len(batch)} msnames, {elapsed:.1f}s elapsed")

        for msname, avg_val in rows:
            result[msname] = avg_val

    con.close()
    return result


def main():
    args = parse_args()
    parquet_pattern = os.path.join(args.parquet_dir, "*.parquet")
    n_bins = args.bins
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    con = duckdb.connect()
    con.execute("SET threads TO 2")
    con.execute("SET memory_limit = '12GB'")
    con.execute("SET preserve_insertion_order = false")
    os.makedirs("/dataset/temp", exist_ok=True)
    con.execute("SET temp_directory = '/dataset/temp'")

    log("Scan 1/4: Computing CPU/memory histograms and global averages...")
    cpu_bins, cpu_cnts, mem_bins, mem_cnts, global_avg_cpu, global_avg_mem = \
        con.execute(f"""
            WITH per_timestamp AS (
                SELECT msname, timestamp,
                       AVG(cpu_utilization) AS cpu,
                       AVG(memory_utilization) AS mem
                FROM read_parquet('{parquet_pattern}')
                WHERE cpu_utilization BETWEEN 0.0 AND 1.0
                  AND memory_utilization BETWEEN 0.0 AND 1.0
                GROUP BY msname, timestamp
            ),
            cpu_hist AS (
                SELECT CAST(LEAST(FLOOR(cpu * {n_bins}), {n_bins - 1}) AS INTEGER) AS bin_idx,
                       COUNT(*) AS cnt
                FROM per_timestamp
                GROUP BY bin_idx
            ),
            mem_hist AS (
                SELECT CAST(LEAST(FLOOR(mem * {n_bins}), {n_bins - 1}) AS INTEGER) AS bin_idx,
                       COUNT(*) AS cnt
                FROM per_timestamp
                GROUP BY bin_idx
            ),
            globals AS (
                SELECT AVG(cpu) AS avg_cpu, AVG(mem) AS avg_mem
                FROM per_timestamp
            )
            SELECT (SELECT LIST(bin_idx ORDER BY bin_idx) FROM cpu_hist),
                   (SELECT LIST(cnt  ORDER BY bin_idx) FROM cpu_hist),
                   (SELECT LIST(bin_idx ORDER BY bin_idx) FROM mem_hist),
                   (SELECT LIST(cnt  ORDER BY bin_idx) FROM mem_hist),
                   avg_cpu, avg_mem
            FROM globals
        """).fetchone()
    cpu_hist_rows = list(zip(cpu_bins, cpu_cnts))
    mem_hist_rows = list(zip(mem_bins, mem_cnts))

    log("Scan 2/4: Computing per-microservice max/min...")
    max_min_df = con.execute(f"""
        WITH per_timestamp AS (
            SELECT msname, timestamp,
                   AVG(cpu_utilization) AS cpu,
                   AVG(memory_utilization) AS mem
            FROM read_parquet('{parquet_pattern}')
            WHERE cpu_utilization BETWEEN 0.0 AND 1.0
              AND memory_utilization BETWEEN 0.0 AND 1.0
            GROUP BY msname, timestamp
        )
        SELECT msname,
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

    log("Scan 3/4: Computing avg instances per timestamp per microservice (parallel)...")
    msnames = max_min_df["msname"].tolist()
    BATCH_INST = 256
    nw = min(8, len(msnames))
    chunks = np.array_split(msnames, nw)
    log(f"Spawning {nw} workers...")

    import time as _time
    t_scan3_start = _time.time()

    with mp.Pool(nw) as pool:
        results = pool.starmap(
            _compute_avg_instances_chunk,
            [(chunk.tolist(), parquet_pattern) for chunk in chunks],
        )

    avg_inst_dict = {}
    for r in results:
        avg_inst_dict.update(r)
    t_scan3_end = _time.time()
    log(f"Scan 3/4 done: {len(avg_inst_dict)} msnames processed in {t_scan3_end - t_scan3_start:.1f}s")
    avg_inst_df = pd.DataFrame(
        list(avg_inst_dict.items()),
        columns=["msname", "avg_instances_per_ts"],
    )
    max_min_df = max_min_df.merge(avg_inst_df, on="msname", how="left")

    log("Scan 4/4: Computing total unique instances per microservice...")
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

    avg_inst_per_ms = float(max_min_df["avg_instances_per_ts"].mean())
    max_avg_inst = float(max_min_df["avg_instances_per_ts"].max())
    min_avg_inst = float(max_min_df["avg_instances_per_ts"].min())

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

    inst_counts = max_min_df["avg_instances_per_ts"].to_numpy()
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

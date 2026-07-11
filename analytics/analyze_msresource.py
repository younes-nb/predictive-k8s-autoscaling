import os
import sys
import time
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
CACHE_DIR = "/dataset/analysis_cache"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def _cache_path(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}.parquet")


def _cache_path_np(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}.npz")


def _load_cache(tag: str):
    pc = _cache_path(tag)
    nc = _cache_path_np(tag)
    if os.path.exists(pc):
        log(f"  Cache hit for '{tag}' ({pc})")
        return ("parquet", pd.read_parquet(pc))
    if os.path.exists(nc):
        log(f"  Cache hit for '{tag}' ({nc})")
        data = np.load(nc, allow_pickle=True)
        return ("npz", data)
    return None


def _save_cache(tag: str, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        path = _cache_path(tag)
        obj.to_parquet(path, index=False)
        log(f"  Cached '{tag}' -> {path}")
    elif isinstance(obj, dict):
        path = _cache_path_np(tag)
        np.savez(path, **{k: v for k, v in obj.items()})
        log(f"  Cached '{tag}' -> {path}")
    elif isinstance(obj, tuple):
        path = _cache_path_np(tag)
        np.savez(path, *obj)
        log(f"  Cached '{tag}' -> {path}")


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
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use cached scan results if available (default: True).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable cache and re-compute all scans.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for scan 3 (default: 8).",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=False,
        help="Delete all cached scan results before running.",
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


def _compute_avg_instances_chunk(
    worker_id: int,
    chunk_msnames: list,
    parquet_pattern: str,
) -> Dict[str, float]:
    """Worker: compute avg instances per timestamp for a chunk of msnames."""
    t_worker_start = time.time()
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET preserve_insertion_order = false")

    result = {}
    BATCH_SIZE = 256
    total = len(chunk_msnames)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"  [W{worker_id}] Starting: {total} msnames in {n_batches} batches")

    for i in range(0, total, BATCH_SIZE):
        batch = chunk_msnames[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        t_batch = time.time()

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

        elapsed = time.time() - t_batch
        pct = (batch_num / n_batches) * 100
        log(
            f"  [W{worker_id}] Batch {batch_num}/{n_batches} "
            f"({pct:.0f}%): {len(batch)} msnames, {len(rows)} results, {_fmt_duration(elapsed)}"
        )

        for msname, avg_val in rows:
            result[msname] = avg_val

    con.close()
    total_elapsed = time.time() - t_worker_start
    log(f"  [W{worker_id}] Finished: {len(result)}/{total} msnames, {_fmt_duration(total_elapsed)}")
    return result


def main():
    t_total_start = time.time()
    args = parse_args()
    parquet_pattern = os.path.join(args.parquet_dir, "*.parquet")
    n_bins = args.bins
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.clear_cache:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            log(f"Cleared cache: {CACHE_DIR}")
        else:
            log("No cache to clear.")

    use_cache = args.use_cache and not args.no_cache
    log(f"Parquet pattern: {parquet_pattern}")
    log(f"Bins: {n_bins}, Workers: {args.num_workers}, Cache: {'ON' if use_cache else 'OFF'}")

    # ---- Scan 1/4: Histograms + globals ----
    log("=" * 60)
    log("Scan 1/4: CPU/memory histograms and global averages...")
    t_scan = time.time()

    cached = _load_cache("scan1") if use_cache else None
    if cached is not None:
        kind, data = cached
        if kind == "npz":
            cpu_bins = data["cpu_bins"].tolist()
            cpu_cnts = data["cpu_cnts"].tolist()
            mem_bins = data["mem_bins"].tolist()
            mem_cnts = data["mem_cnts"].tolist()
            global_avg_cpu = float(data["global_avg_cpu"])
            global_avg_mem = float(data["global_avg_mem"])
        else:
            row = data.iloc[0]
            cpu_bins = row["cpu_bins"]
            cpu_cnts = row["cpu_cnts"]
            mem_bins = row["mem_bins"]
            mem_cnts = row["mem_cnts"]
            global_avg_cpu = float(row["global_avg_cpu"])
            global_avg_mem = float(row["global_avg_mem"])
    else:
        con = duckdb.connect()
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit = '12GB'")
        con.execute("SET preserve_insertion_order = false")
        os.makedirs("/dataset/temp", exist_ok=True)
        con.execute("SET temp_directory = '/dataset/temp'")

        log("  Querying histograms + globals...")
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
        con.close()

        _save_cache("scan1", {
            "cpu_bins": cpu_bins,
            "cpu_cnts": cpu_cnts,
            "mem_bins": mem_bins,
            "mem_cnts": mem_cnts,
            "global_avg_cpu": global_avg_cpu,
            "global_avg_mem": global_avg_mem,
        })

    cpu_hist_rows = list(zip(cpu_bins, cpu_cnts))
    mem_hist_rows = list(zip(mem_bins, mem_cnts))
    t_elapsed = time.time() - t_scan
    log(f"Scan 1/4 done in {_fmt_duration(t_elapsed)}")

    # ---- Scan 2/4: Per-MS max/min ----
    log("=" * 60)
    log("Scan 2/4: Per-microservice max/min...")
    t_scan = time.time()

    cached = _load_cache("scan2") if use_cache else None
    if cached is not None:
        kind, data = cached
        max_min_df = data
    else:
        con = duckdb.connect()
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit = '12GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET temp_directory = '/dataset/temp'")

        log("  Querying per-MS max/min...")
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
        con.close()
        _save_cache("scan2", max_min_df)

    n_ms = len(max_min_df)
    t_elapsed = time.time() - t_scan
    log(f"Scan 2/4 done in {_fmt_duration(t_elapsed)}: {n_ms} unique microservices")

    # ---- Scan 3/4: Avg instances per timestamp (parallel) ----
    log("=" * 60)
    log("Scan 3/4: Avg instances per timestamp per microservice (parallel)...")
    t_scan = time.time()

    cached = _load_cache("scan3") if use_cache else None
    if cached is not None:
        kind, data = cached
        avg_inst_dict = {k: float(v) for k, v in data.items()}
    else:
        msnames = max_min_df["msname"].tolist()
        nw = min(args.num_workers, len(msnames))
        chunks = np.array_split(msnames, nw)
        log(f"  {len(msnames)} msnames -> {nw} workers, {len(chunks[0])} msnames/worker avg")

        t_workers_start = time.time()
        with mp.Pool(nw) as pool:
            results = pool.starmap(
                _compute_avg_instances_chunk,
                [(i, chunk.tolist(), parquet_pattern) for i, chunk in enumerate(chunks)],
            )

        avg_inst_dict = {}
        for r in results:
            avg_inst_dict.update(r)
        t_workers_elapsed = time.time() - t_workers_start
        log(f"  All workers combined: {len(avg_inst_dict)} msnames, {_fmt_duration(t_workers_elapsed)}")

        _save_cache("scan3", avg_inst_dict)

    avg_inst_df = pd.DataFrame(
        list(avg_inst_dict.items()),
        columns=["msname", "avg_instances_per_ts"],
    )
    max_min_df = max_min_df.merge(avg_inst_df, on="msname", how="left")
    t_elapsed = time.time() - t_scan
    log(f"Scan 3/4 done in {_fmt_duration(t_elapsed)}")

    # ---- Scan 4/4: Total unique instances ----
    log("=" * 60)
    log("Scan 4/4: Total unique instances per microservice...")
    t_scan = time.time()

    cached = _load_cache("scan4") if use_cache else None
    if cached is not None:
        kind, data = cached
        total_inst_df = data
    else:
        con = duckdb.connect()
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit = '12GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET temp_directory = '/dataset/temp'")

        log("  Querying total unique instances...")
        total_inst_df = con.execute(f"""
            SELECT msname, COUNT(DISTINCT msinstanceid) AS total_instances
            FROM read_parquet('{parquet_pattern}')
            WHERE cpu_utilization BETWEEN 0.0 AND 1.0
              AND memory_utilization BETWEEN 0.0 AND 1.0
            GROUP BY msname
        """).df()
        con.close()
        _save_cache("scan4", total_inst_df)

    t_elapsed = time.time() - t_scan
    log(f"Scan 4/4 done in {_fmt_duration(t_elapsed)}")

    # ---- Aggregation + Summary ----
    log("=" * 60)
    log("Aggregating results...")
    t_agg = time.time()

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

    t_agg_elapsed = time.time() - t_agg
    log(f"Aggregation done in {_fmt_duration(t_agg_elapsed)}")

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

    # ---- Plots ----
    log("Generating plots...")
    t_plots = time.time()

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

    t_plots_elapsed = time.time() - t_plots
    log(f"Plots done in {_fmt_duration(t_plots_elapsed)}")

    t_total = time.time() - t_total_start
    log(f"All done. Total wall-clock: {_fmt_duration(t_total)}")


if __name__ == "__main__":
    main()

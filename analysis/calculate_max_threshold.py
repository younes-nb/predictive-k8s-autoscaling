import os
import random
import argparse
import sys
import subprocess
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def run_ingestion():
    ingest_script = os.path.join(REPO_ROOT, "preprocessing", "ingest_traces_parquet.py")

    tasks = [
        {
            "table": "msresource",
            "raw": Paths.RAW_MSRESOURCE,
            "out": Paths.PARQUET_THRESHOLD_MSRESOURCE,
        },
        {"table": "msrtmcre", "raw": Paths.RAW_MSRTMCRE, "out": Paths.PARQUET_THRESHOLD_MSRTMCRE},
    ]

    for task in tasks:
        print(f"🛠  Running Ingestion for {task['table']}...")
        cmd = [
            sys.executable,
            ingest_script,
            "--table",
            task["table"],
            "--feature_set",
            "threshold_analysis",
            "--raw_dir",
            task["raw"],
            "--out_dir",
            task["out"],
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"❌ Ingestion failed for {task['table']}. Exiting.")
            sys.exit(1)
    print("✅ Ingestion complete.\n")


def get_base_threshold_for_ms(df_ms):
    if df_ms["cpu_utilization"].max() > 1.0:
        df_ms = df_ms.with_columns(pl.col("cpu_utilization") / 100.0)

    df_ms = df_ms.with_columns(
        [
            (
                (
                    pl.col("providerrpc_mcr")
                    + pl.col("http_mcr")
                    + pl.col("providermq_mcr")
                ).alias("total_mcr")
            ),
            (
                (pl.col("providerrpc_rt") * pl.col("providerrpc_mcr"))
                + (pl.col("http_rt") * pl.col("http_mcr"))
                + (pl.col("providermq_rt") * pl.col("providermq_mcr"))
            ).alias("weighted_rt_sum"),
        ]
    )

    df_ms = df_ms.filter(pl.col("total_mcr") > 0)
    if df_ms.height == 0:
        return None

    df_ms = df_ms.with_columns(
        (pl.col("weighted_rt_sum") / pl.col("total_mcr")).alias("agg_rt")
    )
    df_ms = df_ms.with_columns((pl.col("cpu_utilization") * 100).round(0) / 100).rename(
        {"cpu_utilization": "cpu_bin"}
    )

    analysis_data = (
        df_ms.group_by("cpu_bin")
        .agg(pl.col("agg_rt").quantile(0.95).alias("p95_rt"))
        .sort("cpu_bin")
    )
    x, y = analysis_data["cpu_bin"].to_numpy(), analysis_data["p95_rt"].to_numpy()

    if len(x) < 5:
        return None

    try:
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing")
        if kneedle.knee:
            return round(kneedle.knee - 0.05, 2)
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Empirical Saturation Analysis with Auto-Ingest"
    )
    parser.add_argument("--rt_mcr", type=str, default=Paths.PARQUET_THRESHOLD_MSRTMCRE)
    parser.add_argument("--cpu", type=str, default=Paths.PARQUET_THRESHOLD_MSRESOURCE)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--out", type=str, default="threshold_distribution.png")
    parser.add_argument(
        "--skip_ingest", action="store_true", help="Skip the preprocessing/ingest step"
    )

    args = parser.parse_args()

    if not args.skip_ingest:
        run_ingestion()
    else:
        print("⏭  Skipping ingest step as requested.")

    print("🚀 Initializing Lazy DataFrames...")
    try:
        q_rt = (
            pl.scan_parquet(args.rt_mcr)
            if not args.rt_mcr.endswith(".csv")
            else pl.scan_csv(args.rt_mcr)
        )
        q_cpu = (
            pl.scan_parquet(args.cpu)
            if not args.cpu.endswith(".csv")
            else pl.scan_csv(args.cpu)
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    ms_names = q_rt.select("msname").unique().collect().get_column("msname").to_list()
    selected_ms = (
        random.sample(ms_names, min(args.count, len(ms_names)))
        if args.count
        else ms_names
    )

    results = []
    for i, name in enumerate(selected_ms):
        print(f"[{i+1}/{len(selected_ms)}] Processing {name}...", end="\r")
        ms_cpu = q_cpu.filter(pl.col("msname") == name)
        ms_rt = q_rt.filter(pl.col("msname") == name)
        combined = ms_rt.join(
            ms_cpu, on=["timestamp", "msinstanceid", "msname"]
        ).collect()
        if not combined.is_empty():
            threshold = get_base_threshold_for_ms(combined)
            if threshold:
                results.append(threshold)

    if not results:
        print("\n❌ No valid saturation points found.")
        return

    avg_val = np.mean(results)
    print(
        "\n"
        + "=" * 45
        + f"\nRecommended BASE_THRESHOLD: {avg_val:.2f}\n"
        + "=" * 45
    )

    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(avg_val, color="red", linestyle="dashed", label=f"Mean ({avg_val:.2f})")
    plt.title("Distribution of Optimal CPU Thresholds")
    plt.savefig(args.out)


if __name__ == "__main__":
    main()

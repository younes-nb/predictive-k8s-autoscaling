import os
import random
import argparse
import sys
import subprocess
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

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
            "out": Paths.PARQUET_MSRESOURCE,
        },
        {"table": "msrtmcre", "raw": Paths.RAW_MSRTMCRE, "out": Paths.PARQUET_MSRTMCRE},
    ]

    for task in tasks:
        print(f"🛠  Syncing {task['table']}...")
        subprocess.run(
            [
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
            ],
            check=True,
        )


def get_noise_floor_for_ms(df_ms, idle_percentile=0.10):
    if df_ms["cpu_utilization"].max() > 1.0:
        df_ms = df_ms.with_columns(pl.col("cpu_utilization") / 100.0)

    df_ms = df_ms.with_columns(
        (
            pl.col("providerrpc_mcr") + pl.col("http_mcr") + pl.col("providermq_mcr")
        ).alias("total_mcr")
    )

    active_df = df_ms.filter(pl.col("total_mcr") > 0)
    if active_df.height < 10:
        return None

    traffic_threshold = active_df["total_mcr"].quantile(idle_percentile)
    idle_periods = active_df.filter(pl.col("total_mcr") <= traffic_threshold)

    if idle_periods.height < 5:
        return None

    noise_peak = idle_periods["cpu_utilization"].quantile(0.95)

    recommended_min = noise_peak + 0.05

    return round(min(recommended_min, 1.0), 2)


def main():
    parser = argparse.ArgumentParser(
        description="Noise Floor Analysis for MIN_THRESHOLD Calibration"
    )
    parser.add_argument(
        "--count", type=int, default=None, help="Number of MS to sample"
    )
    parser.add_argument(
        "--idle_p", type=float, default=0.10, help="Percentile to define 'idle' traffic"
    )
    parser.add_argument("--out", type=str, default="min_threshold_distribution.png")
    parser.add_argument("--skip_ingest", action="store_true")

    args = parser.parse_args()

    if not args.skip_ingest:
        run_ingestion()

    q_rt = pl.scan_parquet(Paths.PARQUET_MSRTMCRE)
    q_cpu = pl.scan_parquet(Paths.PARQUET_MSRESOURCE)

    ms_names = q_rt.select("msname").unique().collect().get_column("msname").to_list()
    selected_ms = (
        random.sample(ms_names, min(args.count, len(ms_names)))
        if args.count
        else ms_names
    )

    results = []
    for i, name in enumerate(selected_ms):
        print(f"[{i+1}/{len(selected_ms)}] Analyzing noise for {name}...", end="\r")

        combined = (
            q_rt.filter(pl.col("msname") == name)
            .join(
                q_cpu.filter(pl.col("msname") == name),
                on=["timestamp", "msinstanceid", "msname"],
            )
            .collect()
        )

        if not combined.is_empty():
            val = get_noise_floor_for_ms(combined, args.idle_p)
            if val:
                results.append(val)

    if not results:
        print("\n❌ Could not find sufficient idle periods for analysis.")
        return

    final_min_threshold = np.percentile(results, 95)

    print("\n" + "=" * 45)
    print("📉 THESIS JUSTIFICATION: MIN_THRESHOLD (NOISE FLOOR)")
    print("=" * 45)
    print(f"Services Analyzed:       {len(results)}")
    print(f"Mean Idle CPU:           {np.mean(results):.2f}")
    print(f"Cluster Noise Ceiling:   {final_min_threshold:.2f}")
    print("-" * 45)
    print(f"Thesis Recommendation: Set MIN_THRESHOLD to {final_min_threshold:.2f}")
    print(f"This clears background noise for 95% of analyzed services.")
    print("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=20, color="lightcoral", edgecolor="black", alpha=0.7)
    plt.axvline(
        final_min_threshold,
        color="blue",
        linestyle="dashed",
        label=f"Rec. MIN ({final_min_threshold:.2f})",
    )
    plt.title("Distribution of Service Noise Floors (Idle CPU + Buffer)")
    plt.xlabel("Required Min CPU Threshold")
    plt.ylabel("Service Count")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(args.out)


if __name__ == "__main__":
    main()

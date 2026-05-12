import os
import sys
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def main():
    parser = argparse.ArgumentParser(description="Empirical Startup Latency Analysis")
    parser.add_argument("--cpu", type=str, default=Paths.PARQUET_MSRESOURCE)
    parser.add_argument("--rt_mcr", type=str, default=Paths.PARQUET_MSRTMCRE)
    parser.add_argument("--out", type=str, default="startup_latency.png")
    args = parser.parse_args()

    print("🚀 Loading data...")
    q_cpu = pl.scan_parquet(args.cpu)

    q_rt = pl.scan_parquet(args.rt_mcr)

    print("running allocation query...")
    alloc_times = q_cpu.group_by("msinstanceid").agg(
        pl.col("timestamp").min().alias("t_alloc")
    )

    print("running readiness query...")
    ready_times = (
        q_rt.with_columns(
            (
                pl.col("providerrpc_mcr")
                + pl.col("http_mcr")
                + pl.col("providermq_mcr")
            ).alias("total_mcr")
        )
        .filter(pl.col("total_mcr") > 0)
        .group_by("msinstanceid")
        .agg(pl.col("timestamp").min().alias("t_ready"))
    )

    print("joining and calculating lag...")
    df_lag = (
        alloc_times.join(ready_times, on="msinstanceid", how="inner")
        .with_columns((pl.col("t_ready") - pl.col("t_alloc")).alias("lag_ms"))
        .filter(pl.col("lag_ms") >= 0)
        .collect()
    )

    lags_seconds = df_lag["lag_ms"].to_numpy() / 1000.0

    if len(lags_seconds) == 0:
        print("❌ No valid startup sequences found.")
        return

    avg_lag = np.mean(lags_seconds)
    p50 = np.percentile(lags_seconds, 50)
    p90 = np.percentile(lags_seconds, 90)
    p99 = np.percentile(lags_seconds, 99)

    recommended_horizon = int(np.ceil(p90 / 60.0))

    print("\n" + "=" * 45)
    print("⏱️  STARTUP LATENCY ANALYSIS")
    print("=" * 45)
    print(f"Pods Analyzed:      {len(lags_seconds)}")
    print(f"Average Lag:        {avg_lag:.1f}s")
    print(f"Median Lag (P50):   {p50:.1f}s")
    print(f"Worst-case (P90):   {p90:.1f}s")
    print(f"Extreme (P99):      {p99:.1f}s")
    print("-" * 45)
    print(f"Since the data resolution is 60s:")
    print(f"- {np.mean(lags_seconds <= 60):.1%} of pods are ready within 1 interval.")
    print(f"- {np.mean(lags_seconds <= 120):.1%} of pods are ready within 2 intervals.")
    print("-" * 45)
    print(
        f"Thesis Recommendation: Set PREDICTION_HORIZON to {max(2, recommended_horizon)} intervals"
    )
    print(
        f"({max(2, recommended_horizon) * 60} seconds) to cover the P90 startup latency."
    )
    print("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.hist(
        lags_seconds,
        bins=range(0, 660, 60),
        color="mediumpurple",
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(p90, color="red", linestyle="dashed", label=f"P90 ({p90:.0f}s)")
    plt.title("Distribution of Pod Startup Latencies (Time to Readiness)")
    plt.xlabel("Latency (Seconds)")
    plt.ylabel("Pod Count")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(range(0, 660, 60))
    plt.savefig(args.out)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()

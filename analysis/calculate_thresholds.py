import os
import random
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator


def get_base_threshold_for_ms(df_ms):
    if df_ms["cpu_utilization"].max() > 1.0:
        df_ms = df_ms.with_columns(pl.col("cpu_utilization") / 100.0)

    df_ms = df_ms.with_columns(
        [
            (
                pl.col("providerrpc_mcr")
                + pl.col("http_mcr")
                + pl.col("providermq_mcr")
            ).alias("total_mcr"),
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

    x = analysis_data["cpu_bin"].to_numpy()
    y = analysis_data["p95_rt"].to_numpy()

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
        description="Large-scale Knee Analysis for Base Threshold Justification"
    )
    parser.add_argument(
        "--rt_mcr", type=str, required=True, help="Path to MSRTMCR CSV/Parquet"
    )
    parser.add_argument(
        "--cpu", type=str, required=True, help="Path to MSResource/CPU CSV/Parquet"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of random MS to analyze (None = All)",
    )
    args = parser.parse_args()

    print("🚀 Initializing Lazy DataFrames...")
    q_rt = (
        pl.scan_csv(args.rt_mcr)
        if args.rt_mcr.endswith(".csv")
        else pl.scan_parquet(args.rt_mcr)
    )
    q_cpu = (
        pl.scan_csv(args.cpu)
        if args.cpu.endswith(".csv")
        else pl.scan_parquet(args.cpu)
    )

    ms_names = q_rt.select("msname").unique().collect().get_column("msname").to_list()

    if args.count:
        selected_ms = random.sample(ms_names, min(args.count, len(ms_names)))
        print(f"🎲 Selected {len(selected_ms)} random microservices for analysis.")
    else:
        selected_ms = ms_names
        print(f"📊 Analyzing all {len(selected_ms)} microservices.")

    results = []

    for i, name in enumerate(selected_ms):
        print(f"[{i+1}/{len(selected_ms)}] Analyzing {name}...", end="\r")

        ms_cpu = q_cpu.filter(pl.col("msname") == name)
        ms_rt = q_rt.filter(pl.col("msname") == name)

        combined = ms_rt.join(
            ms_cpu, on=["timestamp", "msinstanceid", "msname"]
        ).collect()

        threshold = get_base_threshold_for_ms(combined)
        if threshold:
            results.append(threshold)

    if not results:
        print("\n❌ No valid saturation points found in the selected sample.")
        return

    avg_val = np.mean(results)
    p95_val = np.percentile(results, 95)
    p05_val = np.percentile(results, 5)

    print("\n" + "=" * 40)
    print("📈 CLUSTER-WIDE BASE THRESHOLD STATS")
    print("=" * 40)
    print(f"Microservices Analyzed: {len(results)}")
    print(f"Average Base Threshold:  {avg_val:.2f}")
    print(f"P95 Base Threshold:      {p95_val:.2f}")
    print(f"P05 (Conservative) Thr:  {p05_val:.2f}")
    print("-" * 40)
    print(f"Thesis Recommendation: Set BASE_THRESHOLD to {avg_val:.2f}")
    print(f"and MIN_THRESHOLD to {p05_val:.2f}")
    print("=" * 40)

    plt.figure(figsize=(10, 5))
    plt.hist(results, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(avg_val, color="red", linestyle="dashed", label=f"Mean ({avg_val:.2f})")
    plt.axvline(
        p95_val, color="green", linestyle="dotted", label=f"P95 ({p95_val:.2f})"
    )
    plt.title("Distribution of Empirically Derived Base Thresholds")
    plt.xlabel("CPU Utilization Threshold")
    plt.ylabel("Service Count")
    plt.legend()
    plt.savefig("threshold_distribution.png")

if __name__ == "__main__":
    main()

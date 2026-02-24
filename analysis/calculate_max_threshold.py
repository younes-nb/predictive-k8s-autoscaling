import os
import random
import argparse
import sys
import subprocess
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import medfilt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


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
        return None, None

    df_ms = df_ms.with_columns(
        (pl.col("weighted_rt_sum") / pl.col("total_mcr")).alias("agg_rt")
    )
    df_ms = df_ms.with_columns((pl.col("cpu_utilization") * 100).round(0) / 100).rename(
        {"cpu_utilization": "cpu_bin"}
    )

    analysis_data = (
        df_ms.group_by("cpu_bin")
        .agg(
            [
                pl.col("agg_rt").quantile(0.95).alias("p95_rt"),
                pl.len().alias("sample_count"),
            ]
        )
        .filter(pl.col("sample_count") >= 10)
        .sort("cpu_bin")
    )

    x, y = analysis_data["cpu_bin"].to_numpy(), analysis_data["p95_rt"].to_numpy()
    if len(x) < 7:
        return None, None

    baseline_rt = np.mean(y[:3])
    if np.max(y) < (baseline_rt * 1.5):
        return None, None

    y_filtered = medfilt(y, kernel_size=3)

    y_monotonic = np.maximum.accumulate(y_filtered)

    y_min, y_max = np.min(y_monotonic), np.max(y_monotonic)
    y_norm = (y_monotonic - y_min) / (y_max - y_min) if y_max > y_min else y_monotonic

    window = 5
    if len(y_norm) >= window:
        y_smoothed = np.convolve(y_norm, np.ones(window) / window, mode="valid")
        x_smoothed = x[window - 1 :]
    else:
        y_smoothed, x_smoothed = y_norm, x

    try:
        kneedle = KneeLocator(
            x_smoothed, y_smoothed, S=1.0, curve="convex", direction="increasing"
        )

        if kneedle.knee:
            val = round(kneedle.knee, 2)
            if 0.40 <= val <= 0.95:
                return val, {"x": x, "y": y, "knee": val}
    except:
        pass

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--skip_ingest", action="store_true")
    args = parser.parse_args()

    print("🚀 Initializing Lazy DataFrames...")
    q_rt = pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRTMCRE)
    q_cpu = pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRESOURCE)

    ms_names = q_rt.select("msname").unique().collect().get_column("msname").to_list()
    selected_ms = (
        random.sample(ms_names, min(args.count, len(ms_names)))
        if args.count
        else ms_names
    )

    results_data = []
    skipped_count = 0

    for i, name in enumerate(selected_ms):
        print(f"[{i+1}/{len(selected_ms)}] Processing {name}... ", end="")
        combined = (
            q_rt.filter(pl.col("msname") == name)
            .join(
                q_cpu.filter(pl.col("msname") == name),
                on=["timestamp", "msinstanceid", "msname"],
            )
            .collect()
        )

        if not combined.is_empty():
            threshold, plot_info = get_base_threshold_for_ms(combined)
            if threshold:
                results_data.append(
                    {"name": name, "threshold": threshold, "plot": plot_info}
                )
                print(f"✅ Knee at: {threshold}")
            else:
                skipped_count += 1
                print("Skipped (Invalid Curve)")
        else:
            skipped_count += 1
            print("Empty Data")

    if not results_data:
        print("\n❌ No valid knees found.")
        return

    thresholds = [r["threshold"] for r in results_data]
    avg_val = np.mean(thresholds)

    print("\n" + "=" * 45)
    print("📊 REFINED SATURATION RESULTS")
    print("=" * 45)
    print(f"Total Processed: {len(selected_ms)}")
    print(f"Successfully Calibrated: {len(results_data)}")
    print(f"Skipped/Filtered Out: {skipped_count}")
    print("-" * 45)
    print(f"Recommended MAX_THRESHOLD: {avg_val:.2f}")
    print("=" * 45)

    min_case = min(results_data, key=lambda x: x["threshold"])
    max_case = max(results_data, key=lambda x: x["threshold"])
    avg_case = min(results_data, key=lambda x: abs(x["threshold"] - avg_val))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (case, title) in zip(
        axes, [(min_case, "Low"), (avg_case, "Avg"), (max_case, "High")]
    ):
        d = case["plot"]
        ax.plot(d["x"], d["y"], "o-", alpha=0.4, label="Raw P95 RT")
        ax.axvline(
            d["knee"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Knee: {d['knee']}",
        )
        ax.set_title(f"{title} Threshold: {case['name']}")
        ax.set_xlabel("CPU Utilization")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("saturation_examples.png")

    plt.figure(figsize=(10, 6))
    plt.hist(thresholds, bins=20, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(
        avg_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg_val:.2f}"
    )
    plt.title("Distribution of Detected CPU Saturation Knees")
    plt.xlabel("CPU Threshold")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("threshold_distribution.png")


if __name__ == "__main__":
    main()

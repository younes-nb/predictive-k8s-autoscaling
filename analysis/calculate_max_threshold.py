import os
import random
import argparse
import sys
import subprocess
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import gc

pl.Config.set_streaming_chunk_size(10000)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def analyze_microservice_arrays(x, y):
    if len(x) < 5:
        return None, None

    low_cpu_mask = x <= max(0.30, x[min(4, len(x) - 1)])
    baseline_rt = np.median(y[low_cpu_mask])

    if baseline_rt == 0 or np.isnan(baseline_rt):
        return None, None

    degradation_threshold = max(baseline_rt * 1.75, baseline_rt + 2.0)

    if np.max(y) < degradation_threshold:
        max_cpu_observed = np.max(x)
        if max_cpu_observed >= 0.60:
            val = min(0.95, round(max_cpu_observed, 2))
            return val, {"status": "Safe"}
        return None, None

    y_smoothed = medfilt(y, kernel_size=3)
    breach_count = 0
    for i in range(len(x)):
        if x[i] < 0.40:
            continue
        if y_smoothed[i] >= degradation_threshold:
            breach_count += 1
            if breach_count >= 2:
                knee_idx = max(0, i - 1)
                val = max(0.40, min(0.95, round(x[knee_idx], 2)))
                return val, {"status": "Saturated"}
        else:
            breach_count = 0
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    print("🔍 Phase 1: Discovering Services...")
    with pl.StringCache():
        all_ms_names = (
            pl.scan_parquet(
                os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
            )
            .select("msname")
            .unique()
            .collect(engine="streaming")
            .get_column("msname")
            .to_list()
        )

    if args.count:
        all_ms_names = random.sample(all_ms_names, min(args.count, len(all_ms_names)))

    total_ms = len(all_ms_names)
    print(f"🎯 Target: {total_ms} services. Starting Processing...")

    results_data = []

    with pl.StringCache():
        for i in range(0, total_ms, args.batch_size):
            batch = all_ms_names[i : i + args.batch_size]
            print(
                f"📦 Batch {i//args.batch_size + 1}: Analyzing {len(batch)} services..."
            )

            q_rt = (
                pl.scan_parquet(
                    os.path.join(Paths.PARQUET_THRESHOLD_MSRTMCRE, "*.parquet")
                )
                .filter(pl.col("msname").is_in(batch))
                .with_columns(
                    total_mcr=(
                        pl.col("providerrpc_mcr")
                        + pl.col("http_mcr")
                        + pl.col("providermq_mcr")
                    )
                )
                .filter(pl.col("total_mcr") > 0)
                .with_columns(
                    agg_rt=(
                        (
                            pl.col("providerrpc_rt") * pl.col("providerrpc_mcr")
                            + pl.col("http_rt") * pl.col("http_mcr")
                            + pl.col("providermq_rt") * pl.col("providermq_mcr")
                        )
                        / pl.col("total_mcr")
                    )
                )
                .select(["timestamp", "msinstanceid", "msname", "agg_rt"])
                .cast({"msname": pl.Categorical, "msinstanceid": pl.Categorical})
            )

            q_cpu = (
                pl.scan_parquet(
                    os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
                )
                .filter(pl.col("msname").is_in(batch))
                .with_columns(
                    cpu_utilization=pl.when(pl.col("cpu_utilization") > 1.0)
                    .then(pl.col("cpu_utilization") / 100.0)
                    .otherwise(pl.col("cpu_utilization"))
                )
                .with_columns(cpu_bin=(pl.col("cpu_utilization") * 100).round(0) / 100)
                .select(["timestamp", "msinstanceid", "msname", "cpu_bin"])
                .cast({"msname": pl.Categorical, "msinstanceid": pl.Categorical})
            )

            try:
                df_batch = (
                    q_rt.join(q_cpu, on=["timestamp", "msinstanceid", "msname"])
                    .group_by(["msname", "cpu_bin"])
                    .agg(p95_rt=pl.col("agg_rt").quantile(0.95), n=pl.len())
                    .filter(pl.col("n") >= 3)
                    .sort(["msname", "cpu_bin"])
                    .collect(engine="streaming")
                )

                if not df_batch.is_empty():
                    for msname, partition in df_batch.partition_by(
                        "msname", as_dict=True
                    ).items():
                        threshold, _ = analyze_microservice_arrays(
                            partition["cpu_bin"].to_numpy(),
                            partition["p95_rt"].to_numpy(),
                        )
                        if threshold:
                            results_data.append(
                                {"msname": msname, "threshold": threshold}
                            )

                del df_batch
                gc.collect()

            except Exception as e:
                print(f"⚠️ Batch failed: {e}")
                continue

    if results_data:
        thresholds = [r["threshold"] for r in results_data]
        print(f"\n✅ Analysis Complete.")
        print(f"Average CPU Saturation Threshold: {np.mean(thresholds):.2f}")

        plt.figure(figsize=(10, 6))
        plt.hist(thresholds, bins=20, color="skyblue", edgecolor="black")
        plt.title("Distribution of CPU Saturation Thresholds")
        plt.savefig("threshold_distribution.png")
    else:
        print("❌ No saturation data found.")


if __name__ == "__main__":
    main()

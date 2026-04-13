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
        {
            "table": "msrtmcre",
            "raw": Paths.RAW_MSRTMCRE,
            "out": Paths.PARQUET_THRESHOLD_MSRTMCRE,
        },
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
        ]
        subprocess.run(cmd, check=True)
    print("✅ Ingestion complete.\n")


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
            return val, {
                "x": x,
                "y": y,
                "knee": val,
                "baseline": baseline_rt,
                "threshold_rt": degradation_threshold,
                "status": "Safe",
            }
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
                return val, {
                    "x": x,
                    "y": y,
                    "knee": val,
                    "baseline": baseline_rt,
                    "threshold_rt": degradation_threshold,
                    "status": "Saturated",
                }
        else:
            breach_count = 0
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=None, help="Limit number of services"
    )
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=150,
        help="Smaller batch = lower RAM footprint",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        run_ingestion()

    print("🔍 Phase 1: Streaming Microservice Discovery...")
    try:
        all_ms_names = (
            pl.scan_parquet(
                os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
            )
            .select("msname")
            .unique()
            .collect(streaming=True)
            .get_column("msname")
            .to_list()
        )
    except Exception as e:
        print(f"❌ Streaming discovery failed: {e}. Attempting standard discovery...")
        all_ms_names = (
            pl.scan_parquet(
                os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
            )
            .select("msname")
            .unique()
            .collect()
            .get_column("msname")
            .to_list()
        )

    if args.count:
        all_ms_names = random.sample(all_ms_names, min(args.count, len(all_ms_names)))

    total_ms = len(all_ms_names)
    print(f"🎯 Total microservices to analyze: {total_ms}")

    results_data = []
    plot_samples = {}

    print(f"🚀 Starting Batched Analysis (Batch Size: {args.batch_size})...")

    for i in range(0, total_ms, args.batch_size):
        batch = all_ms_names[i : i + args.batch_size]
        print(f"📦 Batch {i//args.batch_size + 1}: Processing {len(batch)} services...")

        q_rt = (
            pl.scan_parquet(os.path.join(Paths.PARQUET_THRESHOLD_MSRTMCRE, "*.parquet"))
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
        )

        try:
            df_batch = (
                q_rt.join(q_cpu, on=["timestamp", "msinstanceid", "msname"])
                .group_by(["msname", "cpu_bin"])
                .agg(p95_rt=pl.col("agg_rt").quantile(0.95), n_samples=pl.len())
                .filter(pl.col("n_samples") >= 3)
                .sort(["msname", "cpu_bin"])
                .collect(streaming=True)
            )

            if not df_batch.is_empty():
                for msname, partition in df_batch.partition_by(
                    "msname", as_dict=True
                ).items():
                    threshold, info = analyze_microservice_arrays(
                        partition["cpu_bin"].to_numpy(), partition["p95_rt"].to_numpy()
                    )
                    if threshold:
                        results_data.append({"msname": msname, "threshold": threshold})
                        if len(plot_samples) < 20:
                            plot_samples[msname] = {
                                "name": msname,
                                "threshold": threshold,
                                "plot": info,
                            }

        except Exception as e:
            print(f"⚠️ Batch {i//args.batch_size + 1} crashed: {e}")
            continue

        finally:
            gc.collect()

    if not results_data:
        print("❌ No saturation knees detected. Check data ingestion.")
        return

    thresholds = [r["threshold"] for r in results_data]
    avg_threshold = np.mean(thresholds)

    print("\n" + "=" * 40)
    print(f"📊 FINAL SATURATION RESULTS")
    print("=" * 40)
    print(f"Services Calibrated:  {len(results_data)}")
    print(f"Recommended MAX_CPU:  {avg_threshold:.2f}")
    print("=" * 40)

    plt.figure(figsize=(10, 6))
    plt.hist(thresholds, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(
        avg_threshold, color="red", linestyle="--", label=f"Mean: {avg_threshold:.2f}"
    )
    plt.title("Distribution of CPU Saturation Thresholds")
    plt.xlabel("CPU Utilization")
    plt.ylabel("Microservice Count")
    plt.legend()
    plt.savefig("threshold_distribution.png")

    print("📈 Saved distribution to threshold_distribution.png")


if __name__ == "__main__":
    main()

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
                "status": "Safe at High CPU",
            }
        else:
            return None, None

    y_smoothed = medfilt(y, kernel_size=3)
    breach_count = 0
    required_consecutive_breaches = 2

    for i in range(len(x)):
        if x[i] < 0.50:
            continue

        if y_smoothed[i] >= degradation_threshold:
            breach_count += 1
            if breach_count >= required_consecutive_breaches:
                knee_idx = i - required_consecutive_breaches + 1
                val = round(x[knee_idx], 2)
                val = max(0.50, min(0.95, val))

                return val, {
                    "x": x,
                    "y": y,
                    "knee": val,
                    "baseline": baseline_rt,
                    "threshold_rt": degradation_threshold,
                    "status": "Degradation Found",
                }
        else:
            breach_count = 0

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    if not args.skip_ingest:
        run_ingestion()
    else:
        print("⏭  Skipping ingest step as requested.")

    print("🚀 Initializing Lazy DataFrames...")

    q_rt_skinny = (
        pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRTMCRE)
        .with_columns(
            total_mcr=(
                pl.col("providerrpc_mcr")
                + pl.col("http_mcr")
                + pl.col("providermq_mcr")
            ),
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

    q_cpu_skinny = (
        pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRESOURCE)
        .with_columns(
            cpu_utilization=pl.when(pl.col("cpu_utilization") > 1.0)
            .then(pl.col("cpu_utilization") / 100.0)
            .otherwise(pl.col("cpu_utilization"))
        )
        .with_columns(cpu_bin=((pl.col("cpu_utilization") * 100).round(0) / 100))
        .select(["timestamp", "msinstanceid", "msname", "cpu_bin"])
    )

    print("🔍 Identifying unique microservices...")
    all_ms_names = (
        q_rt_skinny.select("msname")
        .unique()
        .collect(engine="streaming")
        .get_column("msname")
        .to_list()
    )

    selected_ms = (
        random.sample(all_ms_names, min(args.count, len(all_ms_names)))
        if args.count
        else all_ms_names
    )

    total_to_process = len(selected_ms)
    print(f"🎯 Total microservices to analyze: {total_to_process}")

    results_data = []
    plot_samples = {}
    skipped_count = 0
    batch_size = args.batch_size

    for i in range(0, total_to_process, batch_size):
        batch_names = selected_ms[i : i + batch_size]
        print(
            f"📦 Processing batch {i//batch_size + 1}: [{i} to {min(i+batch_size, total_to_process)}]"
        )

        q_batch = (
            q_rt_skinny.filter(pl.col("msname").is_in(batch_names))
            .join(
                q_cpu_skinny.filter(pl.col("msname").is_in(batch_names)),
                on=["timestamp", "msinstanceid", "msname"],
            )
            .group_by(["msname", "cpu_bin"])
            .agg(p95_rt=pl.col("agg_rt").quantile(0.95), sample_count=pl.len())
            .filter(pl.col("sample_count") >= 3)
        )

        try:
            df_batch = q_batch.collect(engine="streaming").sort(["msname", "cpu_bin"])

            for partition_df in df_batch.partition_by("msname"):
                name = partition_df["msname"][0]
                x = partition_df["cpu_bin"].to_numpy()
                y = partition_df["p95_rt"].to_numpy()

                threshold, plot_info = analyze_microservice_arrays(x, y)
                if threshold:
                    results_data.append({"name": name, "threshold": threshold})

                    if len(plot_samples) < 50:
                        plot_samples[name] = {
                            "name": name,
                            "threshold": threshold,
                            "plot": plot_info,
                        }
                else:
                    skipped_count += 1

            del df_batch
            gc.collect()

        except Exception as e:
            print(f"⚠️ Error processing batch {i//batch_size + 1}: {e}")
            continue

    if not results_data:
        print("\n❌ No valid knees found.")
        return

    thresholds = [r["threshold"] for r in results_data]
    avg_val = np.mean(thresholds)

    print("\n" + "=" * 45)
    print("📊 REFINED SATURATION RESULTS")
    print("=" * 45)
    print(f"Total Processed: {total_to_process}")
    print(f"Successfully Calibrated: {len(results_data)}")
    print(f"Skipped/Filtered Out: {skipped_count}")
    print("-" * 45)
    print(f"Recommended MAX_THRESHOLD: {avg_val:.2f}")
    print("=" * 45)

    sample_cases = list(plot_samples.values())
    if sample_cases:
        min_case = min(sample_cases, key=lambda x: x["threshold"])
        max_case = max(sample_cases, key=lambda x: x["threshold"])
        avg_case = min(sample_cases, key=lambda x: abs(x["threshold"] - avg_val))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (case, title) in zip(
            axes, [(min_case, "Low"), (avg_case, "Avg"), (max_case, "High")]
        ):
            d = case["plot"]
            ax.plot(d["x"], d["y"], "o-", alpha=0.6, label="Raw P95 RT")
            ax.axhline(
                d["baseline"],
                color="green",
                linestyle=":",
                label=f"Baseline: {d['baseline']:.1f}ms",
            )
            ax.axhline(
                d["threshold_rt"],
                color="orange",
                linestyle=":",
                label=f"Degraded: {d['threshold_rt']:.1f}ms",
            )

            line_label = (
                f"Max Safe CPU: {d['knee']}"
                if d.get("status") == "Safe at High CPU"
                else f"Saturation CPU: {d['knee']}"
            )
            ax.axvline(
                d["knee"], color="red", linestyle="--", linewidth=2, label=line_label
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

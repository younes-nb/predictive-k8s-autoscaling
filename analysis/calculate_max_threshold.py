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
        return None, None

    y_smoothed = medfilt(y, kernel_size=3)
    breach_count = 0
    for i in range(len(x)):
        if x[i] < 0.50:
            continue
        if y_smoothed[i] >= degradation_threshold:
            breach_count += 1
            if breach_count >= 2:
                knee_idx = i - 1
                val = max(0.50, min(0.95, round(x[knee_idx], 2)))
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
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit number of microservices to analyze",
    )
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=300,
        help="Number of microservices per memory-safe batch",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        run_ingestion()
    else:
        print("⏭  Skipping ingest step.")

    print("🔍 Phase 1: Discovering unique microservices via Lazy Scan...")
    all_ms_names = (
        pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRESOURCE)
        .select("msname")
        .unique()
        .collect()
        .get_column("msname")
        .to_list()
    )

    if args.count:
        all_ms_names = random.sample(all_ms_names, min(args.count, len(all_ms_names)))

    total_ms = len(all_ms_names)
    print(f"🎯 Total microservices to process: {total_ms}")

    results_data = []
    plot_samples = {}
    skipped_count = 0

    print(f"🚀 Starting Batch Processing (Size: {args.batch_size})...")

    for i in range(0, total_ms, args.batch_size):
        batch = all_ms_names[i : i + args.batch_size]
        curr_batch_num = (i // args.batch_size) + 1
        print(f"📦 Batch {curr_batch_num}: Processing {len(batch)} services...")

        with pl.StringCache():
            q_rt = (
                pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRTMCRE)
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
                .cast({"msinstanceid": pl.Categorical, "msname": pl.Categorical})
            )

            q_cpu = (
                pl.scan_parquet(Paths.PARQUET_THRESHOLD_MSRESOURCE)
                .filter(pl.col("msname").is_in(batch))
                .with_columns(
                    cpu_utilization=pl.when(pl.col("cpu_utilization") > 1.0)
                    .then(pl.col("cpu_utilization") / 100.0)
                    .otherwise(pl.col("cpu_utilization"))
                )
                .with_columns(
                    cpu_bin=((pl.col("cpu_utilization") * 100).round(0) / 100)
                )
                .select(["timestamp", "msinstanceid", "msname", "cpu_bin"])
                .cast({"msinstanceid": pl.Categorical, "msname": pl.Categorical})
            )

            try:
                df_batch = (
                    q_rt.join(q_cpu, on=["timestamp", "msinstanceid", "msname"])
                    .group_by(["msname", "cpu_bin"])
                    .agg(p95_rt=pl.col("agg_rt").quantile(0.95), sample_count=pl.len())
                    .filter(pl.col("sample_count") >= 3)
                    .sort(["msname", "cpu_bin"])
                    .collect()
                )
            except Exception as e:
                print(f"⚠️ Batch {curr_batch_num} failed: {e}")
                continue

        if not df_batch.is_empty():
            batch_partitions = df_batch.with_columns(
                pl.col("msname").cast(pl.String)
            ).partition_by("msname", as_dict=True)

            for msname, partition_df in batch_partitions.items():
                x = partition_df["cpu_bin"].to_numpy()
                y = partition_df["p95_rt"].to_numpy()

                threshold, plot_info = analyze_microservice_arrays(x, y)
                if threshold:
                    results_data.append({"name": msname, "threshold": threshold})
                    if len(plot_samples) < 50:
                        plot_samples[msname] = {
                            "name": msname,
                            "threshold": threshold,
                            "plot": plot_info,
                        }
                else:
                    skipped_count += 1

        del df_batch
        gc.collect()

    if not results_data:
        print("\n❌ No valid saturation points found in any batch.")
        return

    thresholds = [r["threshold"] for r in results_data]
    avg_val = np.mean(thresholds)

    print("\n" + "=" * 45)
    print("📊 BATCHED SATURATION RESULTS")
    print("=" * 45)
    print(f"Total Processed:        {total_ms}")
    print(f"Calibrated:             {len(results_data)}")
    print(f"Skipped:                {skipped_count}")
    print("-" * 45)
    print(f"Recommended MAX_CPU:    {avg_val:.2f}")
    print("=" * 45)

    print("📈 Saving visualization results...")
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
                label=f"Limit: {d['threshold_rt']:.1f}ms",
            )
            line_label = (
                f"Max Safe: {d['knee']}"
                if d.get("status") == "Safe at High CPU"
                else f"Saturation: {d['knee']}"
            )
            ax.axvline(
                d["knee"], color="red", linestyle="--", linewidth=2, label=line_label
            )
            ax.set_title(f"{title}: {case['name']}")
            ax.set_xlabel("CPU")
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
    print(
        "✅ Done. Results saved to saturation_examples.png and threshold_distribution.png"
    )


if __name__ == "__main__":
    main()

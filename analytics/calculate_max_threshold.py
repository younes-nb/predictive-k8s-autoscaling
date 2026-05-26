import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import duckdb
import datetime
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


class LoggerWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_ingestion():
    ingest_script = os.path.join(
        REPO_ROOT,
        "preprocessing",
        "ingest_traces_parquet.py",
    )

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
        print(f"🛠 Running ingestion for {task['table']}...")

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
            print(f"❌ Ingestion failed for {task['table']}")
            sys.exit(1)

    print("✅ Ingestion complete.\n")


def find_last_stable_cpu(
    x,
    y,
    degradation_threshold,
    consecutive_bad_bins=3,
):
    bad = y > degradation_threshold

    if len(bad) < consecutive_bad_bins:
        return np.max(x), "Safe"

    kernel = np.ones(consecutive_bad_bins, dtype=int)

    consecutive = np.convolve(
        bad.astype(int),
        kernel,
        mode="valid",
    )

    degradation_points = np.where(consecutive >= consecutive_bad_bins)[0]

    if len(degradation_points) == 0:
        return np.max(x), "Safe"

    first_bad_window = degradation_points[0]

    last_safe_idx = max(0, first_bad_window - 1)

    return x[last_safe_idx], "Degraded"


def analyze_microservice_arrays(x, y):
    MIN_REQUIRED_BINS = 10
    MIN_CPU_SPAN = 0.20
    MIN_SAMPLES_PER_REGION = 3

    if len(x) < MIN_REQUIRED_BINS:
        return None

    sort_idx = np.argsort(x)

    x = x[sort_idx]
    y = y[sort_idx]

    valid_mask = ~(np.isnan(x) | np.isnan(y))

    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < MIN_REQUIRED_BINS:
        return None

    cpu_span = np.max(x) - np.min(x)

    if cpu_span < MIN_CPU_SPAN:
        return None

    kernel_size = 3 if len(y) < 15 else 5
    y_smoothed = medfilt(y, kernel_size=kernel_size)

    baseline_region = x <= np.percentile(x, 30)

    if np.sum(baseline_region) < MIN_SAMPLES_PER_REGION:
        return None

    baseline_rt = np.median(y_smoothed[baseline_region])

    if baseline_rt <= 0 or np.isnan(baseline_rt):
        return None

    degradation_threshold = max(
        baseline_rt * 2.0,
        baseline_rt + np.std(y_smoothed),
    )

    threshold_cpu, status = find_last_stable_cpu(
        x,
        y_smoothed,
        degradation_threshold,
        consecutive_bad_bins=3,
    )

    confidence = "Low"

    if len(x) >= 20 and cpu_span >= 0.40:
        confidence = "High"
    elif len(x) >= 15 and cpu_span >= 0.30:
        confidence = "Medium"

    return {
        "threshold": round(float(threshold_cpu), 2),
        "status": status,
        "confidence": confidence,
        "x": x,
        "y": y,
        "y_smoothed": y_smoothed,
        "baseline_rt": baseline_rt,
        "degradation_threshold": degradation_threshold,
        "cpu_span": cpu_span,
        "n_bins": len(x),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit applied to total services",
    )

    parser.add_argument(
        "--skip_ingest",
        action="store_true",
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/dataset/duckdb_temp",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of microservices to join at once",
    )

    args = parser.parse_args()

    os.makedirs(Paths.LOGS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file_path = os.path.join(
        Paths.LOGS_DIR,
        f"max_threshold_analysis_{timestamp}.log",
    )

    dist_img_path = os.path.join(
        Paths.LOGS_DIR,
        f"threshold_distribution_{timestamp}.png",
    )

    exam_img_path = os.path.join(
        Paths.LOGS_DIR,
        f"saturation_examples_{timestamp}.png",
    )

    sys.stdout = LoggerWriter(log_file_path)

    print(f"🕒 Starting analysis run at {timestamp}")
    print(f"📂 Logs directory: {Paths.LOGS_DIR}\n")

    if not args.skip_ingest:
        run_ingestion()
    else:
        print("⏭ Skipping ingest step.")

    if not os.path.exists(args.temp_dir):
        print(f"📁 Creating directory: {args.temp_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)

    db_path = os.path.join(
        args.temp_dir,
        "alibaba_processing.db",
    )

    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)

    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='20GB'")

    cpu_parquet_path = os.path.join(
        Paths.PARQUET_THRESHOLD_MSRESOURCE,
        "*.parquet",
    )

    rt_parquet_path = os.path.join(
        Paths.PARQUET_THRESHOLD_MSRTMCRE,
        "*.parquet",
    )

    print("🔍 Phase 1a: identifying unique microservices...")

    unique_ms_query = f"""
    SELECT DISTINCT msname
    FROM read_parquet('{cpu_parquet_path}')
    """

    unique_msnames = con.execute(unique_ms_query).df()["msname"].tolist()

    unique_msnames = [m for m in unique_msnames if pd.notna(m)]

    if args.count:
        unique_msnames = unique_msnames[: args.count]

    total_services = len(unique_msnames)

    print(f"🎯 Found {total_services} unique microservices.")

    con.execute("""
        CREATE TABLE ms_aggregated (
            msname VARCHAR,
            cpu_bin DOUBLE,
            p95_rt DOUBLE,
            n_samples BIGINT
        )
    """)

    total_batches = (total_services + args.batch_size - 1) // args.batch_size

    print(f"🔍 Phase 1b: processing in " f"{total_batches} batches...")

    for i in range(0, total_services, args.batch_size):

        batch = unique_msnames[i : i + args.batch_size]

        msnames_sql_list = ", ".join([f"'{m}'" for m in batch])

        batch_num = (i // args.batch_size) + 1

        print(
            f"📦 Batch {batch_num}/{total_batches} " f"({len(batch)} services)...",
            flush=True,
        )

        query = f"""
        INSERT INTO ms_aggregated

        WITH cpu_data AS (
            SELECT
                timestamp,
                msinstanceid,
                msname,

                ROUND(
                    CASE
                        WHEN cpu_utilization > 1.0
                        THEN cpu_utilization / 100.0
                        ELSE cpu_utilization
                    END,
                    2
                ) AS cpu_bin

            FROM read_parquet('{cpu_parquet_path}')

            WHERE msname IN ({msnames_sql_list})
        ),

        rt_data AS (
            SELECT
                timestamp,
                msinstanceid,
                msname,

                (
                    providerrpc_mcr +
                    http_mcr +
                    providermq_mcr
                ) AS total_mcr,

                (
                    (providerrpc_rt * providerrpc_mcr) +
                    (http_rt * http_mcr) +
                    (providermq_rt * providermq_mcr)
                ) AS total_rt_sum

            FROM read_parquet('{rt_parquet_path}')

            WHERE msname IN ({msnames_sql_list})
              AND (
                    providerrpc_mcr +
                    http_mcr +
                    providermq_mcr
                  ) > 0
        )

        SELECT
            c.msname,
            c.cpu_bin,

            approx_quantile(
                r.total_rt_sum / r.total_mcr,
                0.95
            ) AS p95_rt,

            COUNT(*) AS n_samples

        FROM cpu_data c

        JOIN rt_data r
            ON c.timestamp = r.timestamp
            AND c.msinstanceid = r.msinstanceid
            AND c.msname = r.msname

        GROUP BY c.msname, c.cpu_bin

        HAVING COUNT(*) >= 5
        """

        con.execute(query)

    print("✅ Aggregation complete.")

    df = con.execute("""
        SELECT
            msname,
            cpu_bin,
            p95_rt,
            n_samples
        FROM ms_aggregated
        ORDER BY msname, cpu_bin
    """).df()

    con.close()

    if os.path.exists(db_path):
        os.remove(db_path)

    print("🔍 Phase 2: operational safe CPU analysis...")

    results_data = []

    grouped = df.groupby("msname")

    for msname, group in grouped:

        result = analyze_microservice_arrays(
            group["cpu_bin"].to_numpy(),
            group["p95_rt"].to_numpy(),
        )

        if result is None:
            continue

        result["msname"] = msname

        results_data.append(result)

    if not results_data:
        print("❌ No valid services found.")
        return

    thresholds = [r["threshold"] for r in results_data]

    mean_thresh = np.mean(thresholds)
    median_thresh = np.median(thresholds)

    safe_count = sum(1 for r in results_data if r["status"] == "Safe")

    degraded_count = sum(1 for r in results_data if r["status"] == "Degraded")

    high_conf = sum(1 for r in results_data if r["confidence"] == "High")

    medium_conf = sum(1 for r in results_data if r["confidence"] == "Medium")

    low_conf = sum(1 for r in results_data if r["confidence"] == "Low")

    print("\n" + "=" * 60)
    print("📊 OPERATIONAL SAFE CPU RESULTS")
    print("=" * 60)

    print(f"Valid Services:           {len(results_data)}")
    print(f"Mean Safe CPU:            {mean_thresh:.2f}")
    print(f"Median Safe CPU:          {median_thresh:.2f}")

    print()

    print(f"Services Still Safe:      {safe_count}")
    print(f"Services Degraded:        {degraded_count}")

    print()

    print(f"High Confidence:          {high_conf}")
    print(f"Medium Confidence:        {medium_conf}")
    print(f"Low Confidence:           {low_conf}")

    print("=" * 60)

    plt.figure(figsize=(10, 6))

    plt.hist(
        thresholds,
        bins=20,
        color="coral",
        edgecolor="black",
    )

    plt.axvline(
        mean_thresh,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_thresh:.2f}",
    )

    plt.axvline(
        median_thresh,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_thresh:.2f}",
    )

    plt.title("Distribution of Operational Safe CPU")
    plt.xlabel("CPU Utilization")
    plt.ylabel("Microservice Count")

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(dist_img_path)

    print(f"📈 Saved distribution plot: {dist_img_path}")

    results_data.sort(key=lambda r: r["threshold"])

    min_ex = results_data[0]
    max_ex = results_data[-1]

    avg_ex = min(
        results_data,
        key=lambda r: abs(r["threshold"] - mean_thresh),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    examples = [
        ("Lowest Safe CPU", min_ex),
        ("Average Safe CPU", avg_ex),
        ("Highest Safe CPU", max_ex),
    ]

    for ax, (title_prefix, ex) in zip(axes, examples):

        ax.plot(
            ex["x"],
            ex["y"],
            marker="o",
            linestyle="-",
            alpha=0.5,
            label="Raw P95 RT",
        )

        ax.plot(
            ex["x"],
            ex["y_smoothed"],
            linewidth=2,
            label="Smoothed P95 RT",
        )

        ax.axhline(
            ex["baseline_rt"],
            color="green",
            linestyle=":",
            label=f"Baseline: {ex['baseline_rt']:.1f}ms",
        )

        ax.axhline(
            ex["degradation_threshold"],
            color="orange",
            linestyle=":",
            label=f"Degradation: {ex['degradation_threshold']:.1f}ms",
        )

        ax.axvline(
            ex["threshold"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Safe CPU: {ex['threshold']}",
        )

        ax.set_title(
            f"{title_prefix}\n"
            f"{ex['msname']}\n"
            f"({ex['status']} | {ex['confidence']})"
        )

        ax.set_xlabel("CPU Utilization")
        ax.set_ylabel("Latency (ms)")

        ax.grid(True, alpha=0.3)

        ax.legend()

    plt.tight_layout()

    plt.savefig(exam_img_path)

    print(f"📈 Saved example plots: {exam_img_path}")


if __name__ == "__main__":
    main()

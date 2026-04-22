from asyncio import subprocess
import datetime
import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import duckdb
import pwlf
import warnings

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
    if len(x) < 8:
        return None, None, None, None, None, None

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    y_smoothed = medfilt(y, kernel_size=3)
    y_rcmin = np.minimum.accumulate(y_smoothed[::-1])[::-1]

    baseline_idx = max(2, len(x) // 3)
    baseline_rt = np.median(y_rcmin[:baseline_idx])

    if baseline_rt == 0 or np.isnan(baseline_rt):
        return None, None, None, None, None, None

    degradation_threshold = max(baseline_rt * 1.5, baseline_rt + 2.0)
    max_rt = np.max(y_rcmin)

    if max_rt < degradation_threshold:
        max_cpu_observed = np.max(x)
        if max_cpu_observed >= 0.50:
            val = min(0.95, round(max_cpu_observed, 2))
            return val, {"status": "Safe"}, x, y, baseline_rt, degradation_threshold
        return None, None, None, None, None, None

    breach_idx = np.where(y_rcmin >= degradation_threshold)[0]
    if len(breach_idx) == 0:
        return None, None, None, None, None, None
    first_breach_cpu = x[breach_idx[0]]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            my_pwlf = pwlf.PiecewiseLinFit(x, y_rcmin)
            breakpoints = my_pwlf.fit(2)
            knee = breakpoints[1]
            slopes = my_pwlf.calc_slopes()

            slope1, slope2 = slopes[0], slopes[1]

            if slope2 > 0 and slope2 > slope1:
                if knee < 0.30 and first_breach_cpu > 0.40:
                    final_knee = max(0.30, first_breach_cpu - 0.10)
                else:
                    final_knee = max(0.30, knee)

                if final_knee <= 0.95:
                    return (
                        round(final_knee, 2),
                        {"status": "Saturated"},
                        x,
                        y,
                        baseline_rt,
                        degradation_threshold,
                    )

    except Exception:
        pass

    return None, None, None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=None, help="Limit applied to total services"
    )
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument("--temp_dir", type=str, default="/dataset/duckdb_temp")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of microservices to join at once",
    )
    args = parser.parse_args()

    os.makedirs(Paths.LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file_path = os.path.join(Paths.LOGS_DIR, f"max_threshold_analysis_{timestamp}.log")
    dist_img_path = os.path.join(
        Paths.LOGS_DIR, f"threshold_distribution_{timestamp}.png"
    )
    exam_img_path = os.path.join(Paths.LOGS_DIR, f"saturation_examples_{timestamp}.png")

    sys.stdout = LoggerWriter(log_file_path)
    
    print(f"🕒 Starting Analysis Run at {timestamp}")
    print(f"📂 Logs and Output will be saved to: {Paths.LOGS_DIR}\n")

    if not args.skip_ingest:
        run_ingestion()
    else:
        print("⏭  Skipping ingest step as requested.")

    if not os.path.exists(args.temp_dir):
        print(f"📁 Creating directory: {args.temp_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)

    db_path = os.path.join(args.temp_dir, "alibaba_processing.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='20GB'")

    cpu_parquet_path = os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
    rt_parquet_path = os.path.join(Paths.PARQUET_THRESHOLD_MSRTMCRE, "*.parquet")

    print("🔍 Phase 1a: Identifying unique microservices...")
    unique_ms_query = f"SELECT DISTINCT msname FROM read_parquet('{cpu_parquet_path}')"
    unique_msnames = con.execute(unique_ms_query).df()["msname"].tolist()
    unique_msnames = [m for m in unique_msnames if pd.notna(m)]

    if args.count:
        unique_msnames = unique_msnames[: args.count]

    total_services = len(unique_msnames)
    print(f"🎯 Found {total_services} unique microservices to process.")

    con.execute(
        """
        CREATE TABLE ms_aggregated (
            msname VARCHAR,
            cpu_bin DOUBLE,
            p95_rt DOUBLE,
            n_samples BIGINT
        )
    """
    )

    total_batches = (total_services + args.batch_size - 1) // args.batch_size
    print(f"🔍 Phase 1b: Processing in {total_batches} batches to bypass OOM limits...")

    for i in range(0, total_services, args.batch_size):
        batch = unique_msnames[i : i + args.batch_size]
        msnames_sql_list = ", ".join([f"'{m}'" for m in batch])
        batch_num = (i // args.batch_size) + 1

        print(
            f"📦 Running Join for Batch {batch_num}/{total_batches} ({len(batch)} services)...",
            flush=True,
        )

        query = f"""
        INSERT INTO ms_aggregated
        WITH cpu_data AS (
            SELECT 
                timestamp, msinstanceid, msname,
                ROUND(CASE WHEN cpu_utilization > 1.0 THEN cpu_utilization / 100.0 ELSE cpu_utilization END, 2) as cpu_bin
            FROM read_parquet('{cpu_parquet_path}')
            WHERE msname IN ({msnames_sql_list})
        ),
        rt_data AS (
            SELECT 
                timestamp, msinstanceid, msname,
                (providerrpc_mcr + http_mcr + providermq_mcr) as total_mcr,
                ((providerrpc_rt * providerrpc_mcr) + (http_rt * http_mcr) + (providermq_rt * providermq_mcr)) as total_rt_sum
            FROM read_parquet('{rt_parquet_path}')
            WHERE msname IN ({msnames_sql_list})
              AND (providerrpc_mcr + http_mcr + providermq_mcr) > 0
        )
        SELECT 
            c.msname, 
            c.cpu_bin, 
            approx_quantile(r.total_rt_sum / r.total_mcr, 0.95) as p95_rt,
            COUNT(*) as n_samples
        FROM cpu_data c
        JOIN rt_data r 
            ON c.timestamp = r.timestamp 
            AND c.msinstanceid = r.msinstanceid 
            AND c.msname = r.msname
        GROUP BY c.msname, c.cpu_bin
        HAVING COUNT(*) >= 3
        """
        con.execute(query)

    print("✅ Aggregation complete. Fetching final compressed results into memory...")
    df = con.execute(
        "SELECT msname, cpu_bin, p95_rt FROM ms_aggregated ORDER BY msname, cpu_bin"
    ).df()
    con.close()

    if os.path.exists(db_path):
        os.remove(db_path)

    print("🔍 Phase 2: Piecewise Regression Analysis...")
    results_data = []
    grouped = df.groupby("msname")

    for msname, group in grouped:
        res = analyze_microservice_arrays(
            group["cpu_bin"].to_numpy(), group["p95_rt"].to_numpy()
        )
        if res[0] is not None:
            threshold, info, x_arr, y_arr, base_rt, deg_rt = res
            results_data.append(
                {
                    "msname": msname,
                    "threshold": threshold,
                    "status": info["status"],
                    "x": x_arr,
                    "y": y_arr,
                    "base_rt": base_rt,
                    "deg_rt": deg_rt,
                }
            )

    if results_data:
        thresholds = [r["threshold"] for r in results_data]
        mean_thresh = np.mean(thresholds)

        print("\n" + "=" * 40)
        print("📊 HYBRID PWLF SATURATION RESULTS")
        print("=" * 40)
        print(f"Services with Valid Knees:  {len(results_data)}")
        print(f"Calculated Mean MAX_CPU:    {mean_thresh:.2f}")
        print("=" * 40)

        plt.figure(figsize=(10, 6))
        plt.hist(thresholds, bins=20, color="skyblue", edgecolor="black")
        plt.axvline(
            mean_thresh, color="red", linestyle="--", label=f"Mean: {mean_thresh:.2f}"
        )
        plt.title("Distribution of CPU Saturation (Operational Hybrid)")
        plt.xlabel("CPU Utilization Breakpoint")
        plt.ylabel("Microservice Count")
        plt.legend()
        plt.savefig(dist_img_path)
        print(f"📈 Saved distribution to {dist_img_path}")

        if len(results_data) >= 3:
            results_data.sort(key=lambda item: item["threshold"])

            min_ex = results_data[0]
            max_ex = results_data[-1]
            avg_ex = min(
                results_data, key=lambda item: abs(item["threshold"] - mean_thresh)
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            examples = [
                ("Low Threshold", min_ex),
                ("Avg Threshold", avg_ex),
                ("High Threshold", max_ex),
            ]

            for ax, (title_prefix, ex) in zip(axes, examples):
                ax.plot(
                    ex["x"],
                    ex["y"],
                    marker="o",
                    linestyle="-",
                    alpha=0.6,
                    label="Raw P95 RT",
                )
                ax.axhline(
                    ex["base_rt"],
                    color="green",
                    linestyle=":",
                    label=f"Baseline: {ex['base_rt']:.1f}ms",
                )
                if ex["deg_rt"]:
                    ax.axhline(
                        ex["deg_rt"],
                        color="orange",
                        linestyle=":",
                        label=f"Degraded: {ex['deg_rt']:.1f}ms",
                    )

                label_text = (
                    f"Max Safe CPU: {ex['threshold']}"
                    if ex["status"] == "Safe"
                    else f"Saturation CPU: {ex['threshold']}"
                )
                ax.axvline(
                    ex["threshold"],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=label_text,
                )

                ax.set_title(f"{title_prefix}: {ex['msname']}")
                ax.set_xlabel("CPU Utilization")
                ax.set_ylabel("Latency (ms)")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(exam_img_path)
            print(f"📈 Saved examples to {exam_img_path}")

    else:
        print("❌ No saturation knees detected.")


if __name__ == "__main__":
    main()

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


def analyze_microservice_arrays(x, y):
    if len(x) < 8:
        return None, None

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    y_smoothed = medfilt(y, kernel_size=3)
    y_rcmin = np.minimum.accumulate(y_smoothed[::-1])[::-1]
    baseline_idx = max(2, len(y_rcmin) // 3)
    baseline_rt = np.median(y_rcmin[:baseline_idx])
    max_rt = np.max(y_rcmin)

    if baseline_rt == 0 or np.isnan(baseline_rt):
        return None, None

    if max_rt < (baseline_rt * 1.5) and max_rt < (baseline_rt + 2.0):
        return None, {"status": "Safe"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            my_pwlf = pwlf.PiecewiseLinFit(x, y_rcmin)
            breakpoints = my_pwlf.fit(2)
            knee = breakpoints[1]
            slopes = my_pwlf.calc_slopes()

            slope1, slope2 = slopes[0], slopes[1]

            if slope2 > 0 and slope2 > slope1:
                if 0.15 <= knee <= 0.95:
                    return round(knee, 2), {"status": "Saturated"}

        return None, {"status": "No clear upward knee"}

    except Exception:
        return None, None


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
        threshold, info = analyze_microservice_arrays(
            group["cpu_bin"].to_numpy(), group["p95_rt"].to_numpy()
        )

        if threshold:
            results_data.append({"msname": msname, "threshold": threshold})

    if results_data:
        thresholds = [r["threshold"] for r in results_data]
        print("\n" + "=" * 40)
        print("📊 HYBRID PWLF SATURATION RESULTS")
        print("=" * 40)
        print(f"Services with Valid Knees:  {len(results_data)}")
        print(f"Calculated Mean MAX_CPU:    {np.mean(thresholds):.2f}")
        print("=" * 40)

        plt.figure(figsize=(10, 6))
        plt.hist(thresholds, bins=20, color="lightgreen", edgecolor="black")
        plt.axvline(
            np.mean(thresholds),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(thresholds):.2f}",
        )
        plt.title("Distribution of CPU Saturation (Hybrid Regression)")
        plt.xlabel("CPU Utilization Breakpoint")
        plt.ylabel("Microservice Count")
        plt.legend()
        plt.savefig("threshold_distribution.png")
        print("📈 Saved distribution to threshold_distribution.png")
    else:
        print("❌ No saturation knees detected.")


if __name__ == "__main__":
    main()

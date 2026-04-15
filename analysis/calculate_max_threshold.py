import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import pwlf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths

def analyze_microservice_arrays(x, y):
    if len(x) < 8:
        return None, None

    try:
        my_pwlf = pwlf.PiecewiseLinFit(x, y)

        breakpoints = my_pwlf.fit(2)
        knee = breakpoints[1]

        slopes = my_pwlf.calc_slopes()

        slope_ratio = 3.0
        if slopes[1] < (slopes[0] * slope_ratio) and slopes[1] < 5.0:
            return None, {"status": "Linear/No Knee"}

        r2 = my_pwlf.prediction_explained_variance(x, y)
        if r2 < 0.60:
            return None, {"status": "Too Noisy"}

        if 0.10 <= knee <= 0.95:
            return round(knee, 2), {"status": "Saturated", "r2": r2}

        return None, {"status": "Out of Bounds"}

    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=None, help="Limit applied to total services"
    )
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

    print(f"🔍 Phase 1b: Batch processing ({args.batch_size} services/loop)...")
    for i in range(0, total_services, args.batch_size):
        batch = unique_msnames[i : i + args.batch_size]
        msnames_sql_list = ", ".join([f"'{m}'" for m in batch])

        query = f"""
        INSERT INTO ms_aggregated
        WITH cpu_data AS (
            SELECT timestamp, msinstanceid, msname,
                   ROUND(CASE WHEN cpu_utilization > 1.0 THEN cpu_utilization / 100.0 ELSE cpu_utilization END, 2) as cpu_bin
            FROM read_parquet('{cpu_parquet_path}')
            WHERE msname IN ({msnames_sql_list})
        ),
        rt_data AS (
            SELECT timestamp, msinstanceid, msname,
                   (providerrpc_mcr + http_mcr + providermq_mcr) as total_mcr,
                   ((providerrpc_rt * providerrpc_mcr) + (http_rt * http_mcr) + (providermq_rt * providermq_mcr)) as total_rt_sum
            FROM read_parquet('{rt_parquet_path}')
            WHERE msname IN ({msnames_sql_list})
              AND (providerrpc_mcr + http_mcr + providermq_mcr) > 0
        )
        SELECT c.msname, c.cpu_bin, 
               approx_quantile(r.total_rt_sum / r.total_mcr, 0.95) as p95_rt,
               COUNT(*) as n_samples
        FROM cpu_data c
        JOIN rt_data r ON c.timestamp = r.timestamp AND c.msinstanceid = r.msinstanceid AND c.msname = r.msname
        GROUP BY c.msname, c.cpu_bin
        HAVING COUNT(*) >= 3
        """
        con.execute(query)

    print("✅ Aggregation complete. Fetching results...")
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
        group = group.sort_values("cpu_bin")
        threshold, info = analyze_microservice_arrays(
            group["cpu_bin"].to_numpy(), group["p95_rt"].to_numpy()
        )

        if threshold:
            results_data.append({"msname": msname, "threshold": threshold})

    if results_data:
        thresholds = [r["threshold"] for r in results_data]
        print("\n" + "=" * 40)
        print("📊 PIECEWISE REGRESSION RESULTS")
        print("=" * 40)
        print(f"Services with Valid Knees:  {len(results_data)}")
        print(f"Calculated Mean MAX_CPU:    {np.mean(thresholds):.2f}")
        print("=" * 40)

        plt.figure(figsize=(10, 6))
        plt.hist(thresholds, bins=20, color="lightcoral", edgecolor="black")
        plt.axvline(
            np.mean(thresholds),
            color="blue",
            linestyle="--",
            label=f"Mean: {np.mean(thresholds):.2f}",
        )
        plt.title("Distribution of CPU Saturation (Piecewise Regression)")
        plt.xlabel("CPU Utilization Breakpoint")
        plt.ylabel("Microservice Count")
        plt.legend()
        plt.savefig("threshold_distribution.png")
    else:
        print("❌ No clear saturation knees detected.")


if __name__ == "__main__":
    main()

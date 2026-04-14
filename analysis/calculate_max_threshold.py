import os
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import duckdb

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
    parser.add_argument("--count", type=int, default=None, help="Limit applied after aggregation")
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument("--temp_dir", type=str, default="/dataset/duckdb_temp")
    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        print(f"📁 Creating directory: {args.temp_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)

    print("🔍 Phase 1: Massive One-Pass Aggregation (DuckDB)...")
    
    db_path = os.path.join(args.temp_dir, "alibaba_processing.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    con = duckdb.connect(db_path)
    
    con.execute("PRAGMA threads=4") 
    con.execute("PRAGMA memory_limit='50GB'") 
    con.execute("PRAGMA max_temp_directory_size='2TiB'")
    
    cpu_parquet_path = os.path.join(Paths.PARQUET_THRESHOLD_MSRESOURCE, "*.parquet")
    rt_parquet_path = os.path.join(Paths.PARQUET_THRESHOLD_MSRTMCRE, "*.parquet")

    query = f"""
    CREATE TABLE ms_aggregated AS
    WITH cpu_data AS (
        SELECT 
            timestamp, msinstanceid, msname,
            ROUND(CASE WHEN cpu_utilization > 1.0 THEN cpu_utilization / 100.0 ELSE cpu_utilization END, 2) as cpu_bin
        FROM read_parquet('{cpu_parquet_path}')
    ),
    rt_data AS (
        SELECT 
            timestamp, msinstanceid, msname,
            (providerrpc_mcr + http_mcr + providermq_mcr) as total_mcr,
            ((providerrpc_rt * providerrpc_mcr) + (http_rt * http_mcr) + (providermq_rt * providermq_mcr)) as total_rt_sum
        FROM read_parquet('{rt_parquet_path}')
        WHERE (providerrpc_mcr + http_mcr + providermq_mcr) > 0
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

    print("⏳ Running disk-backed join & aggregation. This may take a while, but it will not crash...")
    con.execute(query)

    print("✅ Aggregation complete. Fetching compressed results into memory...")
    df = con.execute("SELECT msname, cpu_bin, p95_rt FROM ms_aggregated ORDER BY msname, cpu_bin").df()
    con.close()
    
    if os.path.exists(db_path):
        os.remove(db_path)

    print("🔍 Phase 2: Calculating Saturation Knees...")
    results_data = []
    
    grouped = df.groupby("msname")
    
    ms_names_processed = 0
    for msname, group in grouped:
        if args.count and ms_names_processed >= args.count:
            break
            
        group = group.sort_values("cpu_bin")
        threshold, _ = analyze_microservice_arrays(
            group["cpu_bin"].to_numpy(), 
            group["p95_rt"].to_numpy()
        )
        
        if threshold:
            results_data.append({"msname": msname, "threshold": threshold})
            
        ms_names_processed += 1

    if results_data:
        thresholds = [r["threshold"] for r in results_data]
        print("\n" + "="*40)
        print("📊 FINAL SATURATION RESULTS")
        print("="*40)
        print(f"Services Calibrated:  {len(results_data)}")
        print(f"Recommended MAX_CPU:  {np.mean(thresholds):.2f}")
        print("="*40)

        plt.figure(figsize=(10, 6))
        plt.hist(thresholds, bins=20, color="skyblue", edgecolor="black")
        plt.axvline(np.mean(thresholds), color="red", linestyle="--", label=f"Mean: {np.mean(thresholds):.2f}")
        plt.title("Distribution of CPU Saturation Thresholds")
        plt.xlabel("CPU Utilization")
        plt.ylabel("Microservice Count")
        plt.legend()
        plt.savefig("threshold_distribution.png")
        print("📈 Saved distribution to threshold_distribution.png")
    else:
        print("❌ No saturation knees detected.")

if __name__ == "__main__":
    main()

import os
import sys
import glob
import argparse
from datetime import datetime

import duckdb
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths, PREPROCESSING


def log(msg: str) -> None:
    """Log progress to stdout with timestamps.

    Always enabled (per request).
    """

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze msresource CPU utilization with OOM-safe DuckDB queries."
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=Paths.PARQUET_MSRESOURCE,
        help="Directory containing msresource parquet files.",
    )
    parser.add_argument(
        "--pred_horizon",
        type=int,
        default=PREPROCESSING.PRED_HORIZON,
        help="Prediction horizon for lagged correlation.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for CPU utilization histogram.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/dataset/duckdb_temp",
        help="Temporary directory for DuckDB spill files.",
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="8GB",
        help="DuckDB memory limit (e.g. 4GB, 16GB).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="DuckDB execution threads.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=Paths.LOGS_DIR,
        help="Directory to save plots.",
    )
    return parser.parse_args()


def plot_cpu_histogram(bin_edges, counts, out_path):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10, 6))
    plt.bar(
        centers,
        counts,
        width=width,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title("CPU Utilization Distribution (msname-aggregated)")
    plt.xlabel("CPU Utilization")
    plt.ylabel("Data Points")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"CPU utilization histogram saved to {out_path}")


def plot_corr_histogram(values, title, out_path):
    if len(values) == 0:
        log(f"No correlation values available for {title}.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, color="orchid", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Correlation")
    plt.ylabel("Microservice Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Correlation histogram saved to {out_path}")


def main():
    args = parse_args()

    parquet_glob = os.path.join(args.parquet_dir, "*.parquet")
    parquet_files = glob.glob(parquet_glob)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {args.parquet_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpu_hist_path = os.path.join(
        args.out_dir, f"cpu_utilization_hist_{timestamp}.png"
    )
    corr_lag1_path = os.path.join(
        args.out_dir, f"cpu_corr_lag1_{timestamp}.png"
    )
    corr_lag_h_path = os.path.join(
        args.out_dir, f"cpu_corr_lag{args.pred_horizon}_{timestamp}.png"
    )

    db_path = os.path.join(args.temp_dir, f"msresource_cpu_{timestamp}.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)

    log("Connecting to DuckDB ...")
    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{args.temp_dir}'")

    cpu_expr = """
    CASE
        WHEN cpu_utilization > 1.0 THEN cpu_utilization / 100.0
        ELSE cpu_utilization
    END
    """

    log(f"Reading parquet files from: {parquet_glob}")
    log("Creating aggregated view ms_agg ...")
    con.execute(
        f"""
        CREATE TEMP VIEW ms_agg AS
        SELECT
            COALESCE(timestamp_dt, to_timestamp(timestamp / 1000.0)) AS ts,
            msname,
            AVG({cpu_expr}) AS cpu_util
        FROM read_parquet('{parquet_glob}')
        WHERE msname IS NOT NULL
        GROUP BY ts, msname
        """
    )

    log("Computing average CPU across microservices ...")
    avg_df = con.execute(
        """
        SELECT
            AVG(ms_mean) AS avg_across_ms,
            COUNT(*) AS ms_count
        FROM (
            SELECT msname, AVG(cpu_util) AS ms_mean
            FROM ms_agg
            GROUP BY msname
        )
        """
    ).fetchdf()

    avg_across_ms = avg_df["avg_across_ms"].iloc[0]
    ms_count = int(avg_df["ms_count"].iloc[0])

    print("\n" + "=" * 50)
    print("📊 MSRESOURCE CPU UTILIZATION SUMMARY")
    print("=" * 50)
    print(f"Microservices counted: {ms_count}")
    print(f"Avg CPU across MSs:    {avg_across_ms:.6f}")
    print("=" * 50 + "\n")

    # DuckDB doesn't support Postgres' width_bucket(). Use histogram() instead.
    log(f"Computing histogram with {args.bins} bins ...")
    hist_df = con.execute(
        f"""
        SELECT
            UNNEST(histogram(
                LEAST(GREATEST(cpu_util, 0.0), 1.0),
                {args.bins}
            )) AS count
        FROM ms_agg
        WHERE cpu_util IS NOT NULL
        """
    ).fetchdf()

    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    counts = hist_df["count"].to_numpy(dtype=int)

    # Defensive: ensure counts matches the number of bins
    if len(counts) != args.bins:
        log(
            f"Warning: histogram returned {len(counts)} counts, expected {args.bins}. Padding/truncating."
        )
        counts = np.pad(
            counts,
            (0, max(0, args.bins - len(counts))),
            constant_values=0,
        )[: args.bins]

    plot_cpu_histogram(bin_edges, counts, cpu_hist_path)

    log(
        f"Computing lag correlations (lag1 and lag{args.pred_horizon}) per microservice ..."
    )
    corr_df = con.execute(
        f"""
        WITH lagged AS (
            SELECT
                msname,
                cpu_util,
                lag(cpu_util, 1) OVER (
                    PARTITION BY msname
                    ORDER BY ts
                ) AS cpu_lag1,
                lag(cpu_util, {args.pred_horizon}) OVER (
                    PARTITION BY msname
                    ORDER BY ts
                ) AS cpu_lag_h
            FROM ms_agg
        )
        SELECT
            msname,
            corr(cpu_util, cpu_lag1) AS corr_lag1,
            corr(cpu_util, cpu_lag_h) AS corr_lag_h
        FROM lagged
        GROUP BY msname
        """
    ).fetchdf()

    lag1_vals = corr_df["corr_lag1"].dropna().to_numpy()
    lagh_vals = corr_df["corr_lag_h"].dropna().to_numpy()

    print("📈 Correlation Summary")
    if len(lag1_vals):
        avg_lag1 = np.mean(lag1_vals)
        print(f"Avg corr(t, t+1):           {avg_lag1:.6f} (n={len(lag1_vals)})")
    else:
        print("Avg corr(t, t+1):           n/a (insufficient data)")
    if len(lagh_vals):
        avg_lagh = np.mean(lagh_vals)
        print(
            f"Avg corr(t, t+{args.pred_horizon}): {avg_lagh:.6f} (n={len(lagh_vals)})"
        )
    else:
        print(
            f"Avg corr(t, t+{args.pred_horizon}): n/a (insufficient data)"
        )

    plot_corr_histogram(
        lag1_vals,
        "Correlation Distribution: t vs t+1",
        corr_lag1_path,
    )
    plot_corr_histogram(
        lagh_vals,
        f"Correlation Distribution: t vs t+{args.pred_horizon}",
        corr_lag_h_path,
    )

    log("Closing DuckDB connection ...")
    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    log("Done.")


if __name__ == "__main__":
    main()

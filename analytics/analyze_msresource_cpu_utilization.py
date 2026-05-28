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

DEFAULT_CORR_BINS = 30
DEFAULT_CORR_RANGE = (-1.0, 1.0)
DEFAULT_CORR_HORIZON_MAX = 30


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
        "--cpu_lower_bound",
        type=float,
        default=None,
        help="Optional CPU utilization lower bound; values below are excluded.",
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
        default="32GB",
        help="DuckDB memory limit (e.g. 4GB, 16GB).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
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


def plot_corr_histogram(
    values,
    title,
    out_path,
    y_max=None,
    bins=DEFAULT_CORR_BINS,
    bin_range=DEFAULT_CORR_RANGE,
):
    if len(values) == 0:
        log(f"No correlation values available for {title}.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(
        values,
        bins=bins,
        range=bin_range,
        color="orchid",
        edgecolor="black",
        alpha=0.7,
    )
    if y_max is not None:
        plt.ylim(0, y_max)
    plt.title(title)
    plt.xlabel("Correlation")
    plt.ylabel("Microservice Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Correlation histogram saved to {out_path}")


def plot_avg_corr_by_horizon(horizons, avg_corrs, out_path):
    if len(horizons) == 0:
        log("No horizon correlation data available.")
        return

    horizons = np.asarray(horizons, dtype=int)
    avg_corrs = np.asarray(avg_corrs, dtype=float)
    valid_mask = np.isfinite(avg_corrs)
    if not np.any(valid_mask):
        log("No valid average correlations available to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(horizons[valid_mask], avg_corrs[valid_mask], marker="o", color="teal")
    plt.title("Average Correlation by Horizon")
    plt.xlabel("Horizon (t+k)")
    plt.ylabel("Average Correlation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Average correlation-by-horizon plot saved to {out_path}")


def main():
    args = parse_args()

    parquet_glob = os.path.join(args.parquet_dir, "*.parquet")
    parquet_files = glob.glob(parquet_glob)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {args.parquet_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpu_hist_path = os.path.join(args.out_dir, f"cpu_utilization_hist_{timestamp}.png")
    corr_lag1_path = os.path.join(args.out_dir, f"cpu_corr_lag1_{timestamp}.png")
    corr_lag_h_path = os.path.join(
        args.out_dir, f"cpu_corr_lag{args.pred_horizon}_{timestamp}.png"
    )
    corr_avg_path = os.path.join(args.out_dir, f"cpu_corr_horizons_{timestamp}.png")

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
    con.execute(f"""
        CREATE TEMP VIEW ms_agg AS
        SELECT
            COALESCE(timestamp_dt, to_timestamp(timestamp / 1000.0)) AS ts,
            msname,
            AVG({cpu_expr}) AS cpu_util
        FROM read_parquet('{parquet_glob}')
        WHERE msname IS NOT NULL
        GROUP BY ts, msname
        """)

    log("Creating filtered view ms_filtered ...")
    if args.cpu_lower_bound is None:
        con.execute("CREATE TEMP VIEW ms_filtered AS SELECT * FROM ms_agg")
    else:
        log(f"Applying CPU lower bound: {args.cpu_lower_bound}")
        con.execute(
            "CREATE TEMP VIEW ms_filtered AS SELECT * FROM ms_agg WHERE cpu_util >= ?",
            [args.cpu_lower_bound],
        )

    log("Computing average CPU across microservices ...")
    avg_df = con.execute("""
        SELECT
            AVG(ms_mean) AS avg_across_ms,
            COUNT(*) AS ms_count
        FROM (
            SELECT msname, AVG(cpu_util) AS ms_mean
            FROM ms_filtered
            GROUP BY msname
        )
        """).fetchdf()

    avg_across_ms = avg_df["avg_across_ms"].iloc[0]
    ms_count = int(avg_df["ms_count"].iloc[0])

    print("\n" + "=" * 50)
    print("📊 MSRESOURCE CPU UTILIZATION SUMMARY")
    print("=" * 50)
    print(f"Microservices counted: {ms_count}")
    if avg_across_ms is None or np.isnan(avg_across_ms):
        print("Avg CPU across MSs:    n/a (insufficient data)")
    else:
        print(f"Avg CPU across MSs:    {avg_across_ms:.6f}")
    print("=" * 50 + "\n")

    log(f"Computing histogram with {args.bins} bins ...")
    hist_df = con.execute(f"""
        WITH clamped AS (
            SELECT
                LEAST(GREATEST(cpu_util, 0.0), 1.0) AS cpu_clamped
            FROM ms_filtered
            WHERE cpu_util IS NOT NULL
        )
        SELECT
            CAST(
                CASE
                    WHEN cpu_clamped <= 0.0 THEN 1
                    WHEN cpu_clamped >= 1.0 THEN {args.bins}
                    ELSE LEAST({args.bins}, 1 + FLOOR(cpu_clamped * {args.bins}))
                END AS INTEGER
            ) AS bucket,
            COUNT(*) AS count
        FROM clamped
        GROUP BY bucket
        ORDER BY bucket
        """).fetchdf()

    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)

    counts = np.zeros(args.bins, dtype=int)
    for _, row in hist_df.iterrows():
        idx = int(row["bucket"]) - 1  # convert 1-based → 0-based
        if 0 <= idx < args.bins:
            counts[idx] = int(row["count"])

    plot_cpu_histogram(bin_edges, counts, cpu_hist_path)

    log(
        f"Computing lag correlations (lag1 and lag{args.pred_horizon}) per microservice ..."
    )
    corr_df = con.execute(f"""
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
            FROM ms_filtered
        )
        SELECT
            msname,
            corr(cpu_util, cpu_lag1) AS corr_lag1,
            corr(cpu_util, cpu_lag_h) AS corr_lag_h
        FROM lagged
        GROUP BY msname
        """).fetchdf()

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
        print(f"Avg corr(t, t+{args.pred_horizon}): n/a (insufficient data)")

    corr_bins = DEFAULT_CORR_BINS
    corr_range = DEFAULT_CORR_RANGE
    lag1_counts = (
        np.histogram(lag1_vals, bins=corr_bins, range=corr_range)[0]
        if len(lag1_vals)
        else np.array([])
    )
    lagh_counts = (
        np.histogram(lagh_vals, bins=corr_bins, range=corr_range)[0]
        if len(lagh_vals)
        else np.array([])
    )
    y_max = None
    if len(lag1_counts) > 0 or len(lagh_counts) > 0:
        y_max = max(
            lag1_counts.max() if len(lag1_counts) else 0,
            lagh_counts.max() if len(lagh_counts) else 0,
        )
        if y_max == 0:
            y_max = None

    plot_corr_histogram(
        lag1_vals,
        "Correlation Distribution: t vs t+1",
        corr_lag1_path,
        y_max=y_max,
        bins=corr_bins,
        bin_range=corr_range,
    )
    plot_corr_histogram(
        lagh_vals,
        f"Correlation Distribution: t vs t+{args.pred_horizon}",
        corr_lag_h_path,
        y_max=y_max,
        bins=corr_bins,
        bin_range=corr_range,
    )

    log("Computing average correlations for horizons t+1..t+30 ...")
    horizons = list(range(1, DEFAULT_CORR_HORIZON_MAX + 1))
    avg_corrs = []
    for horizon in horizons:
        avg_corr_df = con.execute(
            """
            WITH lagged AS (
                SELECT
                    msname,
                    cpu_util,
                    lag(cpu_util, ?) OVER (
                        PARTITION BY msname
                        ORDER BY ts
                    ) AS cpu_lag
                FROM ms_filtered
            )
            SELECT AVG(corr_val) AS avg_corr
            FROM (
                SELECT msname, corr(cpu_util, cpu_lag) AS corr_val
                FROM lagged
                GROUP BY msname
            )
            """,
            [horizon],
        ).fetchdf()
        avg_corr = avg_corr_df["avg_corr"].iloc[0]
        avg_corrs.append(avg_corr if avg_corr is not None else np.nan)

    plot_avg_corr_by_horizon(horizons, avg_corrs, corr_avg_path)

    log("Closing DuckDB connection ...")
    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    log("Done.")


if __name__ == "__main__":
    main()

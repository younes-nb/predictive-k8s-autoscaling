import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths

TRACE_PROJECT = "google.com:google-cluster-data"
DEFAULT_BINS = 128
ALL_CELLS = ["clusterdata_2019_a", "clusterdata_2019_b", "clusterdata_2019_c",
             "clusterdata_2019_d", "clusterdata_2019_e", "clusterdata_2019_f",
             "clusterdata_2019_g", "clusterdata_2019_h"]

MACHINE_CAPACITY_EVENT_TYPES = (0, 2)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Google Cluster Data v3 CPU/memory utilization via BigQuery "
        "(normalized to [0,1] per machine, mirrors analyze_msresource_cpu_utilization.py). "
        "Analyzes all cells (a-h) by default; override with --cells."
    )
    default_billing_project = os.getenv("GCP_BILLING_PROJECT")
    parser.add_argument(
        "--billing_project",
        type=str,
        default=default_billing_project,
        help="Your GCP project id used to run/bill the BigQuery query job. "
        "If not provided, reads from $GCP_BILLING_PROJECT env var.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="+",
        default=ALL_CELLS,
        help="Trace dataset names to analyze (default: all cells a-h). "
        "E.g., --cells clusterdata_2019_a clusterdata_2019_b",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=Paths.LOGS_DIR,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--maximum_bytes_billed_gb",
        type=float,
        default=50.0,
        help="Safety cap on bytes scanned per query, in GB (default: 50).",
    )
    return parser.parse_args()


def build_query(cell: str) -> str:
    usage_tbl = f"`{TRACE_PROJECT}`.{cell}.instance_usage"
    machine_tbl = f"`{TRACE_PROJECT}`.{cell}.machine_events"

    event_types_sql = ", ".join(str(t) for t in MACHINE_CAPACITY_EVENT_TYPES)

    return f"""
    WITH latest_capacity AS (
      SELECT
        machine_id,
        ARRAY_AGG(capacity ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS capacity
      FROM {machine_tbl}
      WHERE type IN ({event_types_sql})
      GROUP BY machine_id
    ),
    usage_agg AS (
      SELECT
        machine_id,
        AVG(average_usage.cpu)    AS avg_cpu_ncu,
        AVG(average_usage.memory) AS avg_mem_ncu,
        MAX(average_usage.cpu)    AS max_cpu_ncu,
        MIN(average_usage.cpu)    AS min_cpu_ncu,
        MAX(average_usage.memory) AS max_mem_ncu,
        MIN(average_usage.memory) AS min_mem_ncu
      FROM {usage_tbl}
      WHERE machine_id != 0 AND machine_id != -1
      GROUP BY machine_id
    ),
    joined AS (
      SELECT
        u.machine_id,
        SAFE_DIVIDE(u.avg_cpu_ncu, c.capacity.cpu)    AS avg_cpu,
        SAFE_DIVIDE(u.avg_mem_ncu, c.capacity.memory) AS avg_mem,
        SAFE_DIVIDE(u.max_cpu_ncu, c.capacity.cpu)    AS max_cpu,
        SAFE_DIVIDE(u.min_cpu_ncu, c.capacity.cpu)    AS min_cpu,
        SAFE_DIVIDE(u.max_mem_ncu, c.capacity.memory) AS max_mem,
        SAFE_DIVIDE(u.min_mem_ncu, c.capacity.memory) AS min_mem
      FROM usage_agg u
      JOIN latest_capacity c USING (machine_id)
      WHERE c.capacity.cpu > 0 AND c.capacity.memory > 0
    )
    SELECT
      machine_id,
      LEAST(GREATEST(avg_cpu, 0.0), 1.0) AS avg_cpu,
      LEAST(GREATEST(avg_mem, 0.0), 1.0) AS avg_mem,
      LEAST(GREATEST(max_cpu, 0.0), 1.0) AS max_cpu,
      LEAST(GREATEST(min_cpu, 0.0), 1.0) AS min_cpu,
      LEAST(GREATEST(max_mem, 0.0), 1.0) AS max_mem,
      LEAST(GREATEST(min_mem, 0.0), 1.0) AS min_mem
    FROM joined
    WHERE avg_cpu BETWEEN 0.0 AND 1.0
      AND avg_mem BETWEEN 0.0 AND 1.0
    ORDER BY machine_id
    """


def plot_histogram(values, title, xlabel, out_path, color="steelblue", bins=DEFAULT_BINS):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        log(f"No data for {title}, skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, range=(0.0, 1.0), color=color, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Machine Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Saved plot: {out_path}")
    plt.close()


def main():
    args = parse_args()

    if not args.billing_project:
        print("ERROR: --billing_project not provided and $GCP_BILLING_PROJECT not set.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = bigquery.Client(project=args.billing_project)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(args.maximum_bytes_billed_gb * 1024 ** 3),
    )

    log(f"Analyzing {len(args.cells)} cell(s) (billed to {args.billing_project})...")

    all_dfs: Dict[str, pd.DataFrame] = {}
    cell_summaries: List[Dict] = []

    for cell in args.cells:
        log(f"Querying {cell}...")
        query = build_query(cell)
        try:
            df = client.query(query, job_config=job_config).result().to_dataframe()
            n_machines = len(df)
            if n_machines == 0:
                log(f"  {cell}: No rows returned, skipping.")
                continue
            all_dfs[cell] = df
            cell_summaries.append({
                "cell": cell,
                "n_machines": n_machines,
                "avg_cpu": float(df["avg_cpu"].mean()),
                "avg_mem": float(df["avg_mem"].mean()),
                "p50_cpu": float(df["avg_cpu"].quantile(0.5)),
                "p95_cpu": float(df["avg_cpu"].quantile(0.95)),
                "p50_mem": float(df["avg_mem"].quantile(0.5)),
                "p95_mem": float(df["avg_mem"].quantile(0.95)),
            })
            log(f"  {cell}: {n_machines} machines aggregated.")
        except Exception as e:
            log(f"  ERROR querying {cell}: {e}")
            continue

    if not all_dfs:
        log("No data returned from any cell. Check billing_project and cell names.")
        return

    for cell, df in all_dfs.items():
        plot_histogram(
            df["avg_cpu"].to_numpy(),
            f"Per-Machine Average CPU Utilization ({cell})",
            "Average CPU Utilization",
            os.path.join(args.out_dir, f"gcd_avg_cpu_hist_{cell}_{ts}.png"),
            color="steelblue",
            bins=args.bins,
        )
        plot_histogram(
            df["avg_mem"].to_numpy(),
            f"Per-Machine Average Memory Utilization ({cell})",
            "Average Memory Utilization",
            os.path.join(args.out_dir, f"gcd_avg_memory_hist_{cell}_{ts}.png"),
            color="forestgreen",
            bins=args.bins,
        )
        plot_histogram(
            df["max_cpu"].to_numpy(),
            f"Max CPU per Machine ({cell})",
            "Max CPU Utilization",
            os.path.join(args.out_dir, f"gcd_max_cpu_hist_{cell}_{ts}.png"),
            color="coral",
            bins=args.bins,
        )
        plot_histogram(
            df["max_mem"].to_numpy(),
            f"Max Memory per Machine ({cell})",
            "Max Memory Utilization",
            os.path.join(args.out_dir, f"gcd_max_memory_hist_{cell}_{ts}.png"),
            color="coral",
            bins=args.bins,
        )

    print("\n" + "=" * 80)
    print("  GOOGLE CLUSTER DATA v3 — CPU/MEMORY UTILIZATION SUMMARY (ALL CELLS)")
    print("=" * 80)
    print(f"Analysis timestamp:              {ts}")
    print(f"Billing project:                 {args.billing_project}")
    print(f"Cells analyzed:                  {', '.join(args.cells)}")
    print()

    summary_df = pd.DataFrame(cell_summaries)
    if not summary_df.empty:
        total_machines = summary_df["n_machines"].sum()
        global_avg_cpu = summary_df["avg_cpu"].mean()
        global_avg_mem = summary_df["avg_mem"].mean()
        global_p95_cpu = summary_df["p95_cpu"].mean()
        global_p95_mem = summary_df["p95_mem"].mean()

        print(f"  AGGREGATE (across {len(cell_summaries)} cells):")
        print(f"  Total machines:                {total_machines}")
        print(f"  Global mean CPU utilization:   {global_avg_cpu:.6f}")
        print(f"  Global mean memory util:       {global_avg_mem:.6f}")
        print(f"  Global mean p95 CPU:           {global_p95_cpu:.6f}")
        print(f"  Global mean p95 memory:        {global_p95_mem:.6f}")
        print()
        print("  PER-CELL BREAKDOWN:")
        print("  Cell                  Machines  Mean CPU  Mean Mem  P95 CPU  P95 Mem")
        print("  " + "-" * 76)
        for _, row in summary_df.iterrows():
            print(f"  {row['cell']:20}  {row['n_machines']:8}  "
                  f"{row['avg_cpu']:8.6f}  {row['avg_mem']:8.6f}  "
                  f"{row['p50_cpu']:7.6f}  {row['p50_mem']:7.6f}")
    print("=" * 80 + "\n")

    log("Done.")


if __name__ == "__main__":
    main()

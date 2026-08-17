#!/usr/bin/env python
"""Compute correlations between current CPU/Memory/MCR and future values at horizons t+1 to t+5.

Uses Alibaba v2022 traces:
- msresource: cpu_utilization, memory_utilization (30s granularity)
- msrtmcre: http_mcr (frontend), providerrpc_mcr (backend) (3min granularity)

Outputs: correlation tables and heatmap plots per service.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import DATASET_TABLES
from shared.features import FEATURES


# Frontend services use http_mcr, backend services use providerrpc_mcr
FRONTEND_SERVICES = ["frontend"]
BACKEND_SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "paymentservice", "productcatalogservice",
    "recommendationservice", "shippingservice", "redis-cart"
]


def get_mcr_column(service_name: str) -> str:
    """Return the MCR column to use for a given service."""
    if service_name in FRONTEND_SERVICES:
        return "http_mcr"
    return "providerrpc_mcr"


def load_parquet_data(parquet_root: str, table: str, columns: list) -> pl.DataFrame:
    """Load parquet data for a given table."""
    parquet_dir = Path(parquet_root) / table
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    print(f"Loading {len(files)} parquet files for {table}...")
    df = pl.scan_parquet([str(f) for f in files]).select(columns).collect()
    return df


def resample_to_1min(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
    """Resample data to 1-minute granularity by taking the last value in each minute."""
    df = df.with_columns(
        (pl.col(timestamp_col) // 60_000 * 60_000).alias("minute_ts")
    )
    df = df.sort(["msname", "msinstanceid", "minute_ts"])
    df = df.group_by(["msname", "msinstanceid", "minute_ts"]).last()
    return df.drop("timestamp").rename({"minute_ts": "timestamp"})


def compute_correlations(
    msresource_df: pl.DataFrame,
    msrtmcre_df: pl.DataFrame,
    horizons: list = [1, 2, 3, 4, 5],
    mcr_col: str = "http_mcr"
) -> dict:
    """Compute correlations between current and future values for each horizon."""
    
    # Resample both to 1-minute granularity
    msresource_resampled = resample_to_1min(msresource_df)
    msrtmcre_resampled = resample_to_1min(msrtmcre_df)
    
    # Join on service and instance
    joined = msresource_resampled.join(
        msrtmcre_resampled.select(["msname", "msinstanceid", "timestamp", mcr_col]),
        on=["msname", "msinstanceid", "timestamp"],
        how="inner"
    )
    
    results = {}
    for service in joined["msname"].unique():
        service_df = joined.filter(pl.col("msname") == service)
        
        if len(service_df) < 100:
            print(f"  Skipping {service}: insufficient data ({len(service_df)} rows)")
            continue
        
        # Sort by instance and timestamp
        service_df = service_df.sort(["msinstanceid", "timestamp"])
        
        # Compute per-instance correlations, then average
        instance_corrs = {h: [] for h in horizons}
        
        for instance in service_df["msinstanceid"].unique():
            inst_df = service_df.filter(pl.col("msinstanceid") == instance)
            
            if len(inst_df) < horizons[-1] + 10:
                continue
            
            cpu_vals = inst_df["cpu_utilization"].to_numpy()
            mem_vals = inst_df["memory_utilization"].to_numpy()
            mcr_vals = inst_df[mcr_col].to_numpy()
            
            for h in horizons:
                if len(cpu_vals) > h:
                    # Correlation with future CPU at t+h
                    corr_cpu = np.corrcoef(cpu_vals[:-h], cpu_vals[h:])[0, 1]
                    corr_mem = np.corrcoef(mem_vals[:-h], mem_vals[h:])[0, 1]
                    corr_mcr_cpu = np.corrcoef(mcr_vals[:-h], cpu_vals[h:])[0, 1]
                    corr_mcr_mem = np.corrcoef(mcr_vals[:-h], mem_vals[h:])[0, 1]
                    corr_cpu_mcr = np.corrcoef(cpu_vals[:-h], mcr_vals[h:])[0, 1]
                    corr_mem_mcr = np.corrcoef(mem_vals[:-h], mcr_vals[h:])[0, 1]
                    
                    if not np.isnan(corr_cpu):
                        instance_corrs[h].append({
                            "cpu_future_cpu": corr_cpu,
                            "mem_future_mem": corr_mem,
                            "mcr_future_cpu": corr_mcr_cpu,
                            "mcr_future_mem": corr_mcr_mem,
                            "cpu_future_mcr": corr_cpu_mcr,
                            "mem_future_mcr": corr_mem_mcr,
                        })
        
        # Average across instances
        avg_corrs = {}
        for h in horizons:
            if instance_corrs[h]:
                avg_corrs[h] = {k: np.mean([d[k] for d in instance_corrs[h]]) 
                                for k in instance_corrs[h][0].keys()}
                avg_corrs[h]["n_instances"] = len(instance_corrs[h])
                avg_corrs[h]["n_samples"] = len(service_df)
        
        results[service] = avg_corrs
    
    return results


def plot_correlation_heatmap(results: dict, output_dir: str, mcr_col: str):
    """Create heatmap plots of correlations per service."""
    
    for service, corrs in results.items():
        if not corrs:
            continue
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Correlation Analysis: {service} (MCR: {mcr_col})", fontsize=14)
        
        horizons = sorted(corrs.keys())
        metrics = [
            ("cpu_future_cpu", "CPU(t) vs CPU(t+h)"),
            ("mem_future_mem", "Memory(t) vs Memory(t+h)"),
            ("mcr_future_cpu", f"{mcr_col}(t) vs CPU(t+h)"),
            ("mcr_future_mem", f"{mcr_col}(t) vs Memory(t+h)"),
            ("cpu_future_mcr", f"CPU(t) vs {mcr_col}(t+h)"),
            ("mem_future_mcr", f"Memory(t) vs {mcr_col}(t+h)"),
        ]
        
        for idx, (metric_key, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Create correlation matrix
            data = []
            for h in horizons:
                if h in corrs and metric_key in corrs[h]:
                    data.append(corrs[h][metric_key])
                else:
                    data.append(np.nan)
            
            # Plot as bar chart
            colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in data]
            bars = ax.bar([f"t+{h}" for h in horizons], data, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Correlation")
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, data):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_{service}_{mcr_col}.png", dpi=150, bbox_inches='tight')
        plt.close()


def print_correlation_table(results: dict, mcr_col: str):
    """Print correlation tables for each service."""
    
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS (MCR: {mcr_col})")
    print(f"{'='*80}")
    
    for service, corrs in results.items():
        if not corrs:
            continue
        
        print(f"\n{service}:")
        print(f"{'Horizon':<10} {'CPU->CPU':>10} {'Mem->Mem':>10} {'MCR->CPU':>10} {'MCR->Mem':>10} {'CPU->MCR':>10} {'Mem->MCR':>10} {'N_inst':>8}")
        print("-" * 80)
        
        for h in sorted(corrs.keys()):
            c = corrs[h]
            print(f"t+{h:<8} {c.get('cpu_future_cpu', np.nan):>10.4f} "
                  f"{c.get('mem_future_mem', np.nan):>10.4f} "
                  f"{c.get('mcr_future_cpu', np.nan):>10.4f} "
                  f"{c.get('mcr_future_mem', np.nan):>10.4f} "
                  f"{c.get('cpu_future_mcr', np.nan):>10.4f} "
                  f"{c.get('mem_future_mcr', np.nan):>10.4f} "
                  f"{c.get('n_instances', 0):>8d}")


def save_correlation_csv(results: dict, output_dir: str, mcr_col: str):
    """Save correlation results to CSV."""
    
    rows = []
    for service, corrs in results.items():
        for h, c in corrs.items():
            row = {
                "service": service,
                "horizon": h,
                "mcr_type": mcr_col,
                **c
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/correlations_{mcr_col}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved correlations to {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Compute CPU/Memory/MCR correlations at horizons t+1..t+5")
    ap.add_argument("--parquet_root", default="/dataset/parquet",
                    help="Root directory for parquet data")
    ap.add_argument("--output_dir", default="/proj/k8sautoscaledl-PG0/analytics_out/correlations",
                    help="Output directory for plots and tables")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="Horizons to compute (in minutes)")
    ap.add_argument("--services", nargs="+", default=None,
                    help="Specific services to analyze (default: all)")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load msresource data (CPU, Memory)
    print("Loading msresource data...")
    msresource_cols = ["timestamp", "msname", "msinstanceid", "cpu_utilization", "memory_utilization"]
    msresource_df = load_parquet_data(args.parquet_root, "msresource", msresource_cols)
    print(f"  Loaded {len(msresource_df)} rows")
    
    # Load msrtmcre data (MCR)
    print("Loading msrtmcre data...")
    mcr_cols = ["timestamp", "msname", "msinstanceid", "http_mcr", "providerrpc_mcr"]
    msrtmcre_df = load_parquet_data(args.parquet_root, "msrtmcre", mcr_cols)
    print(f"  Loaded {len(msrtmcre_df)} rows")
    
    # Get unique services
    services = msresource_df["msname"].unique().to_list()
    if args.services:
        services = [s for s in services if s in args.services]
    
    print(f"\nAnalyzing services: {services}")
    
    # Compute correlations for frontend (http_mcr)
    print("\n--- Frontend services (http_mcr) ---")
    frontend_results = compute_correlations(
        msresource_df.filter(pl.col("msname").is_in(FRONTEND_SERVICES)),
        msrtmcre_df.filter(pl.col("msname").is_in(FRONTEND_SERVICES)),
        horizons=args.horizons,
        mcr_col="http_mcr"
    )
    print_correlation_table(frontend_results, "http_mcr")
    plot_correlation_heatmap(frontend_results, args.output_dir, "http_mcr")
    save_correlation_csv(frontend_results, args.output_dir, "http_mcr")
    
    # Compute correlations for backend (providerrpc_mcr)
    print("\n--- Backend services (providerrpc_mcr) ---")
    backend_results = compute_correlations(
        msresource_df.filter(pl.col("msname").is_in(BACKEND_SERVICES)),
        msrtmcre_df.filter(pl.col("msname").is_in(BACKEND_SERVICES)),
        horizons=args.horizons,
        mcr_col="providerrpc_mcr"
    )
    print_correlation_table(backend_results, "providerrpc_mcr")
    plot_correlation_heatmap(backend_results, args.output_dir, "providerrpc_mcr")
    save_correlation_csv(backend_results, args.output_dir, "providerrpc_mcr")
    
    # Combined summary
    all_results = {**frontend_results, **backend_results}
    save_correlation_csv(all_results, args.output_dir, "all")
    
    print(f"\nDone! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()

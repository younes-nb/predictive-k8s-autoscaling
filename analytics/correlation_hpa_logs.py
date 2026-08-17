#!/usr/bin/env python
"""Compute correlations from HPA historical logs CSV.

Input: /proj/k8sautoscaledl-PG0/hpa_historical_logs.csv
Columns: timestamp, msname, replicas, http_mcr, providerrpc_mcr, cpu_utilization, memory_utilization

Computes both Pearson (linear) and Distance (non-linear) correlations between
current CPU/Memory/MCR/Replicas and future CPU/Memory at horizons t+1 to t+5.
Frontend (frontend) uses http_mcr, backends use providerrpc_mcr.

Goal: Predict CPU and Memory, so we track:
- CPU(t) → CPU(t+h)
- Memory(t) → Memory(t+h)
- MCR(t) → CPU(t+h)
- MCR(t) → Memory(t+h)
- Replicas(t) → CPU(t+h)
- Replicas(t) → Memory(t+h)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


FRONTEND_SERVICES = ["frontend"]
BACKEND_SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "paymentservice", "productcatalogservice",
    "recommendationservice", "shippingservice"
]

METRICS = [
    ("cpu_future_cpu", "CPU→CPU"),
    ("mem_future_mem", "Mem→Mem"),
    ("mcr_future_cpu", "MCR→CPU"),
    ("mcr_future_mem", "MCR→Mem"),
    ("replicas_future_cpu", "Rep→CPU"),
    ("replicas_future_mem", "Rep→Mem"),
]


def get_mcr_column(service_name: str) -> str:
    """Return the MCR column to use for a given service."""
    if service_name in FRONTEND_SERVICES:
        return "http_mcr"
    return "providerrpc_mcr"


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation (dCor): SOTA measure that detects BOTH linear and
    non-linear dependence (Székely, Rizzo & Bakirov 2007). dCor=0 iff X and Y
    are independent; dCor in [0, 1]. O(n^2) in time and memory.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    # Pairwise distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # Double-centering
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    # Distance covariance and distance variances
    dCov2 = (A * B).sum() / (n * n)
    dVarX2 = (A * A).sum() / (n * n)
    dVarY2 = (B * B).sum() / (n * n)

    if dVarX2 <= 0 or dVarY2 <= 0:
        return 0.0
    return float(np.sqrt(dCov2) / np.sqrt(np.sqrt(dVarX2 * dVarY2)))


def compute_correlations(
    df: pd.DataFrame,
    horizons: list = [1, 2, 3, 4, 5]
) -> dict:
    """Compute Pearson + Distance correlations between current and future CPU/Memory."""
    
    results = {}
    
    for service in df["msname"].unique():
        service_df = df[df["msname"] == service].copy()
        service_df = service_df.sort_values("timestamp").reset_index(drop=True)
        
        if len(service_df) < horizons[-1] + 20:
            print(f"  Skipping {service}: insufficient data ({len(service_df)} rows)")
            continue
        
        mcr_col = get_mcr_column(service)
        
        cpu_vals = service_df["cpu_utilization"].values.astype(float)
        mem_vals = service_df["memory_utilization"].values.astype(float)
        mcr_vals = service_df[mcr_col].values.astype(float)
        rep_vals = service_df["replicas"].values.astype(float)
        
        corrs = {}
        for h in horizons:
            if len(cpu_vals) > h:
                pairs = [
                    ("cpu_future_cpu", cpu_vals[:-h], cpu_vals[h:]),
                    ("mem_future_mem", mem_vals[:-h], mem_vals[h:]),
                    ("mcr_future_cpu", mcr_vals[:-h], cpu_vals[h:]),
                    ("mcr_future_mem", mcr_vals[:-h], mem_vals[h:]),
                    ("replicas_future_cpu", rep_vals[:-h], cpu_vals[h:]),
                    ("replicas_future_mem", rep_vals[:-h], mem_vals[h:]),
                ]
                
                entry = {"n_samples": len(cpu_vals) - h}
                for key, x, y in pairs:
                    # Pearson (linear)
                    p = np.corrcoef(x, y)[0, 1]
                    entry[f"{key}_pearson"] = p if not np.isnan(p) else 0.0
                    # Distance correlation (non-linear)
                    entry[f"{key}_dcor"] = distance_correlation(x, y)
                
                corrs[h] = entry
        
        results[service] = corrs
    
    return results


def compute_average_correlations(results: dict, horizons: list) -> dict:
    """Compute average correlations across all services for each horizon."""
    
    avg_corrs = {}
    
    for h in horizons:
        avg_corrs[h] = {}
        
        for key, _ in METRICS:
            for corr_type in ["pearson", "dcor"]:
                metric = f"{key}_{corr_type}"
                values = [results[s][h][metric] for s in results if h in results[s]]
                if values:
                    avg_corrs[h][metric] = np.mean(values)
                    avg_corrs[h][f"{metric}_std"] = np.std(values)
                    avg_corrs[h][f"{metric}_min"] = np.min(values)
                    avg_corrs[h][f"{metric}_max"] = np.max(values)
                else:
                    avg_corrs[h][metric] = 0
                    avg_corrs[h][f"{metric}_std"] = 0
                    avg_corrs[h][f"{metric}_min"] = 0
                    avg_corrs[h][f"{metric}_max"] = 0
        
        avg_corrs[h]["n_services"] = len([s for s in results if h in results[s]])
    
    return avg_corrs


def print_average_table(avg_corrs: dict, horizons: list):
    """Print average correlation table across all services."""
    
    print(f"\n{'='*140}")
    print(f"AVERAGE CORRELATIONS ACROSS ALL SERVICES (Pearson = linear, dCor = non-linear)")
    print(f"{'='*140}")
    print(f"{'Horizon':<10} {'CPU→CPU':>10} {'Mem→Mem':>10} {'MCR→CPU':>10} {'MCR→Mem':>10} {'Rep→CPU':>10} {'Rep→Mem':>10} {'N':>4}")
    print(f"{'':<10} {'Pearson':>10} {'Pearson':>10} {'Pearson':>10} {'Pearson':>10} {'Pearson':>10} {'Pearson':>10} {'':>4}")
    print("-" * 90)
    
    for h in horizons:
        c = avg_corrs[h]
        print(f"t+{h:<8} {c.get('cpu_future_cpu_pearson', 0):>10.4f} "
              f"{c.get('mem_future_mem_pearson', 0):>10.4f} "
              f"{c.get('mcr_future_cpu_pearson', 0):>10.4f} "
              f"{c.get('mcr_future_mem_pearson', 0):>10.4f} "
              f"{c.get('replicas_future_cpu_pearson', 0):>10.4f} "
              f"{c.get('replicas_future_mem_pearson', 0):>10.4f} "
              f"{c.get('n_services', 0):>4d}")
    
    print(f"\n{'':<10} {'dCor':>10} {'dCor':>10} {'dCor':>10} {'dCor':>10} {'dCor':>10} {'dCor':>10}")
    print("-" * 90)
    
    for h in horizons:
        c = avg_corrs[h]
        print(f"t+{h:<8} {c.get('cpu_future_cpu_dcor', 0):>10.4f} "
              f"{c.get('mem_future_mem_dcor', 0):>10.4f} "
              f"{c.get('mcr_future_cpu_dcor', 0):>10.4f} "
              f"{c.get('mcr_future_mem_dcor', 0):>10.4f} "
              f"{c.get('replicas_future_cpu_dcor', 0):>10.4f} "
              f"{c.get('replicas_future_mem_dcor', 0):>10.4f}")
    
    # Also print with std dev
    print(f"\n{'='*140}")
    print(f"DETAILED STATISTICS (mean ± std)")
    print(f"{'='*140}")
    
    for h in horizons:
        c = avg_corrs[h]
        print(f"\nt+{h}:")
        for key, label in METRICS:
            p_mean = c.get(f"{key}_pearson", 0)
            p_std = c.get(f"{key}_pearson_std", 0)
            d_mean = c.get(f"{key}_dcor", 0)
            d_std = c.get(f"{key}_dcor_std", 0)
            print(f"  {label}: Pearson {p_mean:.4f} ± {p_std:.4f} | dCor {d_mean:.4f} ± {d_std:.4f}")


def print_correlation_table(results: dict):
    """Print correlation tables for each service."""
    
    print(f"\n{'='*140}")
    print(f"CORRELATION ANALYSIS FROM HPA HISTORICAL LOGS (Predicting CPU & Memory)")
    print(f"{'='*140}")
    
    for service, corrs in results.items():
        if not corrs:
            continue
        
        mcr_col = get_mcr_column(service)
        print(f"\n{service} (MCR: {mcr_col}):")
        
        for corr_type, type_label in [("pearson", "Pearson"), ("dcor", "dCor")]:
            print(f"\n  [{type_label}]")
            print(f"  {'Horizon':<10} {'CPU→CPU':>10} {'Mem→Mem':>10} {'MCR→CPU':>10} {'MCR→Mem':>10} {'Rep→CPU':>10} {'Rep→Mem':>10}")
            print("  " + "-" * 70)
            
            for h in sorted(corrs.keys()):
                c = corrs[h]
                print(f"  t+{h:<8} {c.get(f'cpu_future_cpu_{corr_type}', 0):>10.4f} "
                      f"{c.get(f'mem_future_mem_{corr_type}', 0):>10.4f} "
                      f"{c.get(f'mcr_future_cpu_{corr_type}', 0):>10.4f} "
                      f"{c.get(f'mcr_future_mem_{corr_type}', 0):>10.4f} "
                      f"{c.get(f'replicas_future_cpu_{corr_type}', 0):>10.4f} "
                      f"{c.get(f'replicas_future_mem_{corr_type}', 0):>10.4f}")


def plot_correlation_bars(results: dict, output_dir: str):
    """Create bar plots of correlations per service."""
    
    for service, corrs in results.items():
        if not corrs:
            continue
        
        mcr_col = get_mcr_column(service)
        horizons = sorted(corrs.keys())
        
        for corr_type, type_label in [("pearson", "Pearson"), ("dcor", "Distance (dCor)")]:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f"Correlation Analysis: {service} ({type_label}, MCR: {mcr_col})", fontsize=14)
            
            metric_labels = [
                ("cpu_future_cpu", "CPU(t) → CPU(t+h)"),
                ("mem_future_mem", "Memory(t) → Memory(t+h)"),
                ("mcr_future_cpu", f"{mcr_col}(t) → CPU(t+h)"),
                ("mcr_future_mem", f"{mcr_col}(t) → Memory(t+h)"),
                ("replicas_future_cpu", "Replicas(t) → CPU(t+h)"),
                ("replicas_future_mem", "Replicas(t) → Memory(t+h)"),
            ]
            
            for idx, (metric_key, title) in enumerate(metric_labels):
                ax = axes[idx // 3, idx % 3]
                
                data = [corrs[h].get(f"{metric_key}_{corr_type}", np.nan) for h in horizons]
                
                colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in data]
                bars = ax.bar([f"t+{h}" for h in horizons], data, color=colors, alpha=0.7, edgecolor='black')
                
                ax.set_title(title, fontsize=11)
                ax.set_ylabel(f"{type_label} Correlation")
                ax.set_ylim(-1, 1)
                ax.axhline(y=0, color='black', linewidth=0.5)
                ax.axhline(y=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
                
                for bar, val in zip(bars, data):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width()/2, 
                                bar.get_height() + (0.02 if val >= 0 else -0.05),
                                f'{val:.3f}', ha='center', 
                                va='bottom' if val >= 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_{service}_{corr_type}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Combined heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_keys = [m[0] for m in metric_labels]
            metric_labels_short = [m[1] for m in metric_labels]
            
            heatmap_data = np.array([[corrs[h].get(f"{k}_{corr_type}", np.nan) for k in metric_keys] for h in horizons])
            
            im = ax.imshow(heatmap_data.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels([f"t+{h}" for h in horizons])
            ax.set_yticks(range(len(metric_labels_short)))
            ax.set_yticklabels(metric_labels_short)
            ax.set_xlabel("Horizon (minutes)")
            ax.set_title(f"{type_label} Correlation Heatmap: {service}")
            
            for i in range(len(metric_labels_short)):
                for j in range(len(horizons)):
                    val = heatmap_data[j, i]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                                color='white' if abs(val) > 0.5 else 'black', fontsize=10)
            
            plt.colorbar(im, ax=ax, label=f'{type_label} Correlation')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap_{service}_{corr_type}.png", dpi=150, bbox_inches='tight')
            plt.close()


def plot_average_correlations(avg_corrs: dict, horizons: list, output_dir: str):
    """Create plots for average correlations across all services."""
    
    for corr_type, type_label in [("pearson", "Pearson"), ("dcor", "Distance (dCor)")]:
        metric_labels = [
            ("cpu_future_cpu", "CPU(t) → CPU(t+h)"),
            ("mem_future_mem", "Memory(t) → Memory(t+h)"),
            ("mcr_future_cpu", "MCR(t) → CPU(t+h)"),
            ("mcr_future_mem", "MCR(t) → Memory(t+h)"),
            ("replicas_future_cpu", "Replicas(t) → CPU(t+h)"),
            ("replicas_future_mem", "Replicas(t) → Memory(t+h)"),
        ]
        
        # Bar plot with error bars
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"Average {type_label} Correlations Across All Services", fontsize=14)
        
        for idx, (metric_key, title) in enumerate(metric_labels):
            ax = axes[idx // 3, idx % 3]
            
            means = [avg_corrs[h].get(f"{metric_key}_{corr_type}", 0) for h in horizons]
            stds = [avg_corrs[h].get(f"{metric_key}_{corr_type}_std", 0) for h in horizons]
            
            colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in means]
            bars = ax.bar([f"t+{h}" for h in horizons], means, yerr=stds, 
                          color=colors, alpha=0.7, edgecolor='black', capsize=5)
            
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(f"Average {type_label} Correlation")
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            
            for bar, val, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + std + 0.02,
                        f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_average_{corr_type}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Heatmap of averages
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_keys = [m[0] for m in metric_labels]
        metric_labels_short = [m[1] for m in metric_labels]
        
        heatmap_data = np.array([[avg_corrs[h].get(f"{k}_{corr_type}", np.nan) for k in metric_keys] for h in horizons])
        
        im = ax.imshow(heatmap_data.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"t+{h}" for h in horizons])
        ax.set_yticks(range(len(metric_labels_short)))
        ax.set_yticklabels(metric_labels_short)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_title(f"Average {type_label} Correlation Heatmap Across All Services")
        
        for i in range(len(metric_labels_short)):
            for j in range(len(horizons)):
                val = heatmap_data[j, i]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                            color='white' if abs(val) > 0.5 else 'black', fontsize=10)
        
        plt.colorbar(im, ax=ax, label=f'Average {type_label} Correlation')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_average_heatmap_{corr_type}.png", dpi=150, bbox_inches='tight')
        plt.close()


def save_correlation_csv(results: dict, avg_corrs: dict, output_dir: str):
    """Save correlation results to CSV."""
    
    # Per-service
    rows = []
    for service, corrs in results.items():
        mcr_col = get_mcr_column(service)
        for h, c in corrs.items():
            row = {
                "service": service,
                "mcr_type": mcr_col,
                "horizon": h,
                **c
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/correlations_hpa_logs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-service correlations to {csv_path}")
    
    # Average across services
    avg_rows = []
    for h, c in avg_corrs.items():
        row = {"horizon": h, "n_services": c.get("n_services", 0)}
        for key, _ in METRICS:
            for corr_type in ["pearson", "dcor"]:
                metric = f"{key}_{corr_type}"
                row[f"{metric}_mean"] = c.get(metric, 0)
                row[f"{metric}_std"] = c.get(f"{metric}_std", 0)
                row[f"{metric}_min"] = c.get(f"{metric}_min", 0)
                row[f"{metric}_max"] = c.get(f"{metric}_max", 0)
        avg_rows.append(row)
    
    avg_df = pd.DataFrame(avg_rows)
    avg_csv_path = f"{output_dir}/correlations_average.csv"
    avg_df.to_csv(avg_csv_path, index=False)
    print(f"Saved average correlations to {avg_csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Compute Pearson + Distance correlations from HPA logs CSV")
    ap.add_argument("--input_csv", default="/proj/k8sautoscaledl-PG0/hpa_historical_logs.csv",
                    help="Path to HPA historical logs CSV")
    ap.add_argument("--output_dir", default="/proj/k8sautoscaledl-PG0/analytics_out/correlations_hpa",
                    help="Output directory for plots and tables")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="Horizons to compute (in minutes)")
    ap.add_argument("--services", nargs="+", default=None,
                    help="Specific services to analyze (default: all)")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load HPA logs CSV
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {df['msname'].nunique()} services")
    
    # Filter services if specified
    if args.services:
        df = df[df["msname"].isin(args.services)]
    
    # Compute correlations
    print("\nComputing correlations (Pearson + dCor)...")
    results = compute_correlations(df, horizons=args.horizons)
    
    # Compute averages
    print("\nComputing average across all services...")
    avg_corrs = compute_average_correlations(results, args.horizons)
    
    # Print tables
    print_correlation_table(results)
    print_average_table(avg_corrs, args.horizons)
    
    # Create plots
    print("\nGenerating plots...")
    plot_correlation_bars(results, args.output_dir)
    plot_average_correlations(avg_corrs, args.horizons, args.output_dir)
    
    # Save CSV
    save_correlation_csv(results, avg_corrs, args.output_dir)
    
    print(f"\nDone! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()

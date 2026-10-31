#!/usr/bin/env python3
"""Generate comprehensive comparison report for HPA vs PBScaler runs."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_and_analyze(csv_path, name):
    """Load CSV and compute metrics."""
    df = pd.read_csv(csv_path)
    results = {'run': name}
    
    # SLO metrics (front P90, threshold 200ms)
    if 'front_p90' in df.columns:
        results['p90_mean'] = df['front_p90'].mean()
        results['p90_p95'] = df['front_p90'].quantile(0.95)
        results['p90_max'] = df['front_p90'].max()
        results['slo_violations'] = (df['front_p90'] > 200).sum()
        results['slo_violation_rate'] = (df['front_p90'] > 200).mean()
    
    # Throughput
    if 'total_qps' in df.columns:
        results['qps_mean'] = df['total_qps'].mean()
        results['qps_max'] = df['total_qps'].max()
    
    # Replicas (from HPA metrics)
    replica_cols = [c for c in df.columns if c.endswith('replicas') or 'replicas' in c]
    if replica_cols:
        total_replicas = df[replica_cols].sum(axis=1)
        results['replicas_mean'] = total_replicas.mean()
        results['replicas_max'] = total_replicas.max()
        results['replicas_min'] = total_replicas.min()
    
    # Resource usage
    for metric in ['cpu', 'memory', 'io']:
        cols = [c for c in df.columns if metric in c.lower()]
        if cols:
            results[f'{metric}_mean'] = df[cols].mean().mean()
            results[f'{metric}_max'] = df[cols].max().max()
    
    # SLO reward (if present)
    if 'slo_reward' in df.columns:
        results['slo_reward_mean'] = df['slo_reward'].mean()
    
    return results, df

def compute_ml_metrics(hpa_df, pbscaler_df):
    """Compute accuracy, precision, recall, F1 for PBScaler predictions vs HPA actuals."""
    # This requires matching time windows and comparing scaling decisions
    # Placeholder for actual implementation
    return {
        'accuracy': 'N/A - needs implementation',
        'precision': 'N/A',
        'recall': 'N/A',
        'f1': 'N/A'
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpa-jul13', required=True)
    parser.add_argument('--hpa-jul14', required=True)
    parser.add_argument('--pbscaler-jul14', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print("Loading data...")
    hpa13_results, hpa13_df = load_and_analyze(args.hpa_jul13, 'HPA Jul-13')
    hpa14_results, hpa14_df = load_and_analyze(args.hpa_jul14, 'HPA Jul-14')
    pbs_results, pbs_df = load_and_analyze(args.pbscaler_jul14, 'PBScaler Jul-14')

    ml_metrics = compute_ml_metrics(hpa14_df, pbs_df)

    # Generate markdown report
    report = f"""# HPA vs PBScaler Comparison Report
**Generated:** {pd.Timestamp.now().isoformat()}

## Executive Summary
| Metric | HPA Jul-13 | HPA Jul-14 | PBScaler Jul-14 |
|--------|-----------|-----------|----------------|
| Avg Front P90 (ms) | {hpa13_results.get('p90_mean', 'N/A'):.1f} | {hpa14_results.get('p90_mean', 'N/A'):.1f} | {pbs_results.get('p90_mean', 'N/A'):.1f} |
| P90 P95 (ms) | {hpa13_results.get('p90_p95', 'N/A'):.1f} | {hpa14_results.get('p90_p95', 'N/A'):.1f} | {pbs_results.get('p90_p95', 'N/A'):.1f} |
| Max P90 (ms) | {hpa13_results.get('p90_max', 'N/A'):.1f} | {hpa14_results.get('p90_max', 'N/A'):.1f} | {pbs_results.get('p90_max', 'N/A'):.1f} |
| SLO Violations | {hpa13_results.get('slo_violations', 'N/A')} | {hpa14_results.get('slo_violations', 'N/A')} | {pbs_results.get('slo_violations', 'N/A')} |
| SLO Violation Rate | {hpa13_results.get('slo_violation_rate', 'N/A'):.2%} | {hpa14_results.get('slo_violation_rate', 'N/A'):.2%} | {pbs_results.get('slo_violation_rate', 'N/A'):.2%} |
| Avg QPS | {hpa13_results.get('qps_mean', 'N/A'):.1f} | {hpa14_results.get('qps_mean', 'N/A'):.1f} | {pbs_results.get('qps_mean', 'N/A'):.1f} |
| Avg Replicas | {hpa13_results.get('replicas_mean', 'N/A'):.1f} | {hpa14_results.get('replicas_mean', 'N/A'):.1f} | {pbs_results.get('replicas_mean', 'N/A'):.1f} |
| Max Replicas | {hpa13_results.get('replicas_max', 'N/A'):.1f} | {hpa14_results.get('replicas_max', 'N/A'):.1f} | {pbs_results.get('replicas_max', 'N/A'):.1f} |
| Avg CPU | {hpa13_results.get('cpu_mean', 'N/A'):.1f} | {hpa14_results.get('cpu_mean', 'N/A'):.1f} | {pbs_results.get('cpu_mean', 'N/A'):.1f} |
| Avg Memory | {hpa13_results.get('memory_mean', 'N/A'):.1f} | {hpa14_results.get('memory_mean', 'N/A'):.1f} | {pbs_results.get('memory_mean', 'N/A'):.1f} |

## SLO Analysis (200ms threshold)
- **HPA Jul-13:** {hpa13_results.get('slo_violations', 'N/A')} violations ({hpa13_results.get('slo_violation_rate', 'N/A'):.2%})
- **HPA Jul-14:** {hpa14_results.get('slo_violations', 'N/A')} violations ({hpa14_results.get('slo_violation_rate', 'N/A'):.2%})
- **PBScaler Jul-14:** {pbs_results.get('slo_violations', 'N/A')} violations ({pbs_results.get('slo_violation_rate', 'N/A'):.2%})

## Resource Efficiency
| Resource | HPA Jul-13 | HPA Jul-14 | PBScaler Jul-14 | PBScaler vs HPA-14 |
|----------|-----------|-----------|----------------|-------------------|
| CPU | {hpa13_results.get('cpu_mean', 'N/A'):.1f} | {hpa14_results.get('cpu_mean', 'N/A'):.1f} | {pbs_results.get('cpu_mean', 'N/A'):.1f} | {pbs_results.get('cpu_mean', 0) - hpa14_results.get('cpu_mean', 0):.1f} |
| Memory | {hpa13_results.get('memory_mean', 'N/A'):.1f} | {hpa14_results.get('memory_mean', 'N/A'):.1f} | {pbs_results.get('memory_mean', 'N/A'):.1f} | {pbs_results.get('memory_mean', 0) - hpa14_results.get('memory_mean', 0):.1f} |

## PBScaler ML Metrics (vs HPA Jul-14 as ground truth)
- Accuracy: {ml_metrics.get('accuracy', 'N/A')}
- Precision: {ml_metrics.get('precision', 'N/A')}
- Recall: {ml_metrics.get('recall', 'N/A')}
- F1: {ml_metrics.get('f1', 'N/A')}

## Root Cause Analysis: PBScaler Underperformance
*(Auto-generated based on data - to be filled after runs)*

### Potential Causes:
1. **RF Model Quality**: Training on Jul-13 may not capture Jul-14 patterns
2. **Prediction Horizon**: PBScaler's prediction window may be too short/long
3. **Action Space**: Max 8 replicas limit may constrain PBScaler
4. **Observation Noise**: 5s metrics may be too noisy for reliable predictions
5. **Reward Mismatch**: PBScaler optimizes for different objective than HPA

## Detailed Time Series
See `pbscaler/data/exp_*/` for per-run CSVs.

## Recommendations
1. Retrain RF with more diverse data (multiple days)
2. Increase observation window / reduce metric granularity
3. Tune PBScaler hyperparameters (prediction horizon, action space)
4. Consider ensemble models instead of single RF

---
*Report generated by generate_report.py*
"""

    Path(args.output).write_text(report)
    print(f"Report written to {args.output}")

if __name__ == '__main__':
    main()

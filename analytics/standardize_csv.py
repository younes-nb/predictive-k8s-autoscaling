#!/usr/bin/env python3
"""Standardize wide-dynamic-range CSV features with train-zone stats.

cpu_ms_infra channels span 1e-4..6e4 (byte rates vs utils) and blow up
unnormalized LSTM training (NaN loss). This maps every numeric column
except targets/counts/phases to per-service z-scores fitted on rows
[0, train_frac) ONLY (no leak: test rows use train params). Stats saved
beside the output for production reuse.

Kept raw: timestamp, msname, cpu_utilization, memory_utilization
(targets), replicas, desired_replicas (small ints), tod_sin/cos (phases).

Run from repo root:
    python analytics/standardize_csv.py --csv in.csv --out out_std.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

KEEP_RAW = {"timestamp", "msname", "cpu_utilization", "memory_utilization",
            "replicas", "desired_replicas", "tod_sin", "tod_cos"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--id-col", default="msname")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    stats = {}
    frames = []
    for svc, g in df.groupby(args.id_col):
        g = g.reset_index(drop=True)
        ntr = int(len(g) * args.train_frac)
        sstat = {}
        for c in g.columns:
            if c in KEEP_RAW or not pd.api.types.is_numeric_dtype(g[c]):
                continue
            mu = float(g[c].iloc[:ntr].mean())
            sd = float(g[c].iloc[:ntr].std())
            if not np.isfinite(mu):
                mu = 0.0
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1.0
            sstat[c] = [mu, sd]
            g[c] = ((g[c] - mu) / sd).astype("float32")
        stats[svc] = sstat
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).sort_values(
        [args.id_col, "timestamp"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w") as f:
        json.dump({"train_frac": args.train_frac, "keep_raw": sorted(KEEP_RAW),
                   "services": stats}, f)
    print(f"standardized {len(stats)} services -> {args.out}")


if __name__ == "__main__":
    main()

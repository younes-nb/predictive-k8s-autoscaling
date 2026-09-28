#!/usr/bin/env python3
"""Auxiliary experiment export: JVM runtime series (adservice) + sharp RPS.

JVM_* only exist for adservice (agentless jmx_exporter, deploy/jmx/);
other services get NaN (no agent), NEVER zeros. RPS_SHARP uses a 30s rate
window (vs 1m in the main export) to test whether sharper timing helps.
Merged into analytics/twostage.py via --aux-csv (same grid => pass the
same --start/--end/--step).

Run from repo root:
    python analytics/aux_export.py --start "2026-09-23 12:07:53" \\
        --end <ts> --step 10 --out /tmp/opencode/newtest/aux_10s.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pytz

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analytics.export_hpa import (  # noqa: E402
    fetch_range,
    pod_to_deployment,
    PROMETHEUS_URL,
)

TEHRAN = pytz.timezone("Asia/Tehran")
NS = "online-boutique"

QUERIES = {
    "jvm_heap": (
        'sum by (pod) (jvm_memory_bytes_used{area="heap",'
        f'namespace="{NS}"}})',
        "pod",
    ),
    "jvm_nonheap": (
        'sum by (pod) (jvm_memory_bytes_used{area="nonheap",'
        f'namespace="{NS}"}})',
        "pod",
    ),
    "jvm_gc": (
        'sum by (pod) (rate(jvm_gc_collection_seconds_sum{'
        f'namespace="{NS}"}}[2m]))',
        "pod",
    ),
    "jvm_threads": (
        f'max by (pod) (jvm_threads_ThreadCount{{namespace="{NS}"}})',
        "pod",
    ),
    "rps_sharp": (
        'sum(rate(istio_requests_total{reporter="destination",'
        f' destination_workload_namespace="{NS}"}}[30s]))'
        " by (destination_workload)",
        "destination_workload",
    ),
}


def log(msg):
    print(msg, flush=True)


def parse_time(s):
    try:
        return float(s)
    except ValueError:
        return TEHRAN.localize(
            __import__("datetime").datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        ).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    end_ts = time.time() if not args.end else parse_time(args.end)
    start_ts = parse_time(args.start)
    # Identical grid construction to analytics/export_hpa.py (exact start,
    # NOT step-floored): cross-CSV merges only match when both sides share
    # the anchor. Pass the same --start/--end/--step.
    n_points = int((end_ts - start_ts) // args.step) + 1
    grid = np.array([start_ts + k * args.step for k in range(n_points)],
                    dtype=int)
    idx = pd.DatetimeIndex(pd.to_datetime(grid, unit="s", utc=True))
    frames = {}
    for name, (query, label) in QUERIES.items():
        log(f"querying {name} ...")
        try:
            result = fetch_range(PROMETHEUS_URL, query, grid[0], grid[-1],
                                 args.step)
        except Exception as e:
            log(f"  {name} failed: {e}")
            continue
        per = {}
        for series in result:
            m = series["metric"]
            ent = m.get(label, "")
            if label == "pod":
                ent = pod_to_deployment(ent)
            if not ent or ent == "redis-cart":
                continue
            ts = np.array([int(v[0]) for v in series["values"]])
            try:
                vals = np.array([float(v[1]) for v in series["values"]])
            except ValueError:
                continue
            s = pd.Series(vals, index=pd.to_datetime(ts, unit="s", utc=True))
            s = s[~s.index.duplicated(keep="last")].reindex(idx)
            per.setdefault(ent, []).append(s)
        merged = {}
        for ent, lst in per.items():
            cat = pd.concat(lst, axis=1)
            merged[ent] = (cat.max(axis=1) if name == "jvm_threads"
                           else cat.mean(axis=1))
        frames[name] = merged
        log(f"  {name}: {len(merged)} entities")
    entities = set()
    for d in frames.values():
        entities.update(d.keys())
    out = []
    for ent in sorted(entities):
        full = pd.DataFrame(index=idx)
        for name, d in frames.items():
            full[name] = d[ent].values if ent in d else np.nan
        full["timestamp"] = full.index.tz_convert(TEHRAN).strftime(
            "%Y-%m-%d %H:%M:%S")
        full["msname"] = ent
        out.append(full)
    df = pd.concat(out, ignore_index=True)
    df = df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    log(f"saved {len(df)} rows -> {args.out}")
    for c in ("jvm_heap", "rps_sharp"):
        if c in df.columns:
            log(f"  {c}: finite frac={df[c].notna().mean():.3f} "
                f"max={df[c].max():.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-edge RPS export (dynamic call-graph weights for graph models).

One query: reporter="source" rates by (source_workload, destination_workload).
Output: timestamp, src, dst, rps. Same grid convention as export_hpa.py.

Run from repo root:
    python analytics/export_edges.py --start "2026-09-23 12:07:53" \\
        --end "2026-09-25 12:08:13" --step 10 --out /tmp/opencode/full48/edges_10s.csv
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

from analytics.export_hpa import fetch_range, PROMETHEUS_URL  # noqa: E402

TEHRAN = pytz.timezone("Asia/Tehran")
NS_DEFAULT = "train-ticket"


def build_query(ns):
    return (
        'sum by (source_workload, destination_workload) (rate('
        'istio_requests_total{reporter="source",'
        f' source_workload_namespace="{ns}",'
        f' destination_workload_namespace="{ns}"}}[1m]))'
    )


def log(msg):
    print(msg, flush=True)


def parse_time(s):
    try:
        return float(s)
    except ValueError:
        import datetime
        return TEHRAN.localize(
            datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--namespace", default=NS_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    QUERY = build_query(args.namespace)
    end_ts = time.time() if not args.end else parse_time(args.end)
    start_ts = parse_time(args.start)
    n_points = int((end_ts - start_ts) // args.step) + 1
    grid = [start_ts + k * args.step for k in range(n_points)]
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(grid), unit="s", utc=True))
    result = fetch_range(PROMETHEUS_URL, QUERY, grid[0], grid[-1], args.step)
    log(f"{len(result)} edges")
    frames = []
    for series in result:
        m = series["metric"]
        src, dst = m.get("source_workload", ""), m.get("destination_workload", "")
        if not src or not dst or dst == "redis-cart":
            continue
        ts = np.array([int(v[0]) for v in series["values"]])
        vals = np.array([float(vv[1]) for vv in series["values"]])
        s = pd.Series(vals, index=pd.to_datetime(ts, unit="s", utc=True))
        s = s[~s.index.duplicated(keep="last")].reindex(idx).fillna(0.0)
        f = pd.DataFrame({"rps": s.values},
                         index=pd.DatetimeIndex(s.index))
        f["timestamp"] = f.index.tz_convert(TEHRAN).strftime("%Y-%m-%d %H:%M:%S")
        f["src"] = src
        f["dst"] = dst
        frames.append(f[["timestamp", "src", "dst", "rps"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "src", "dst"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    log(f"saved {len(df)} rows -> {args.out}")
    log("top edges by mean rps:\n" + str(
        df.groupby(["src", "dst"]).rps.mean().sort_values(
            ascending=False).head(12).round(2)))


if __name__ == "__main__":
    main()

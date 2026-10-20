#!/usr/bin/env python3
"""Relabel collector run rows from the k6 client-side latency truth.

The collector's front_p90 (istio source-side gateway→ui histogram p90)
produces seconds-scale spikes in windows where k6's own client sees normal
latency (verified on fullrun3: k6 weighted per-window p95 med was 59 ms in
"violation" windows — identical to base). This script pulls the k6
remote-written trend series (K6_PROMETHEUS_RW_TREND_STATS=avg,p(95),p(99))
from Mimir for the run's span, computes a per-window user-side p95 weighted
by per-endpoint request rate, and rewrites slo_reward:

    0 = violation  (k6 weighted window p95 > SLO_MS, default 200)
    1 = clean
    -1 = kept if the collector row had no label *and* k6 data missing

Also writes the k6 per-window p95/fail-fraction as extra columns
(`k6_p95`, `k6_fail`) so they can be used as features by a predictor.

Usage (needs a Mimir port-forward, usually 18081):
    python scripts/pull_k6_labels.py \
        --run pbscaler/data/fullrun3_20261019_125146 \
        --in data/train_ticket/train_data.csv \
        --out data/train_ticket/train_data_k6.csv \
        --url http://127.0.0.1:18081/prometheus
"""

import argparse
import csv
import random

import numpy as np
import pandas as pd
import requests


def pull_matrix(url, expr, start, end, step=10, timeout=120):
    r = requests.post(f"{url}/api/v1/query_range",
                      data={"query": expr, "start": start, "end": end, "step": step},
                      timeout=timeout).json()
    out = {}
    for srs in r["data"]["result"]:
        eid = (srs["metric"].get("name", "?"), srs["metric"].get("status", ""))
        out[eid] = pd.Series({float(t): float(v) for t, v in srs["values"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="collector run dir (metrics/train_data)")
    ap.add_argument("--in", dest="inp", required=True, help="train_data.csv to relabel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18081/prometheus")
    ap.add_argument("--slo-ms", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src = pd.read_csv(args.run + "/train_data.csv")
    src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True)
    start = src["timestamp"].min(); end = src["timestamp"].max()
    print(f"span {start} .. {end}")

    d95 = pull_matrix(args.url, "k6_http_req_duration_p95", start.timestamp(), end.timestamp())
    counters = pull_matrix(args.url, "k6_http_reqs_total", start.timestamp(), end.timestamp())
    print(f"k6 series: p95={len(d95)}, counters={len(counters)}")
    qps = {k: s.diff().clip(lower=0) / 10.0 for k, s in counters.items()}
    times = sorted(set().union(*[set(s.index) for s in d95.values()]))

    glo = pd.Series(index=times, dtype=float)
    fail = pd.Series(index=times, dtype=float)
    ids = sorted(set().union(*[set(s.index) for s in qps.values()]))
    fail_num = pd.Series(0.0, index=ids)
    fail_den = pd.Series(0.0, index=ids)
    for (name, status), s in counters.items():
        s_back = s.reindex(ids).fillna(0.0)
        fail_den = fail_den.add(s_back, fill_value=0)
        if status and status != "200":
            fail_num = fail_num.add(s_back, fill_value=0)
    for t in times:
        num = den = 0.0
        for (name, status), p_s in d95.items():
            qv = qps.get((name, status))
            if qv is None: continue
            if t not in qv.index or t not in p_s.index: continue
            qq, pv = qv.loc[t], p_s.loc[t]
            if not qq or qq != qq or not pv or pv != pv: continue
            num += qq * pv * 1000
            den += qq
        glo.loc[t] = (num / den) if den else np.nan
        if t in fail_den.index and fail_den.loc[t] > 0:
            fail.loc[t] = fail_num.loc[t] / fail_den.loc[t]

    rows, viol = [], 0
    for _, r in src.iterrows():
        ts = r["timestamp"].timestamp()
        i = glo.index.searchsorted(ts)
        cand = [glo.index[i - 1]] + ([glo.index[i]] if i < len(glo.index) else [])
        best = min(cand, key=lambda c: abs(c - ts))
        p95v = glo.loc[best] if abs(best - ts) <= 7 and np.isfinite(glo.loc[best]) else np.nan
        fv = fail.loc[best] if best in fail.index else np.nan
        old = int(r["slo_reward"])
        new = old
        if np.isfinite(p95v):
            new = 0 if p95v > args.slo_ms else 1
        if new != -1:
            viol += (new == 0)
        rows.append((p95v, fv, new))

    out = src.copy()
    out.insert(out.columns.get_loc("front_p90") + 1, "k6_p95", [round(x, 2) if np.isfinite(x) else "" for x, c, l in rows])
    out.insert(out.columns.get_loc("front_p90") + 2, "k6_fail", [round(x, 4) if np.isfinite(x) else "" for x, c, l in rows])
    out["slo_reward"] = [l for x, c, l in rows]
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    old_n = int((src["slo_reward"] == 0).sum())
    relabeled = int((out["slo_reward"] == 0).sum())
    flip = int(((src["slo_reward"] == 0) & (out["slo_reward"] != 0)).sum()) + \
           int(((src["slo_reward"] != 0) & (out["slo_reward"] == 0)).sum())
    print(f"old violations={old_n}  new violations={relabeled}  rows with label flips={flip}")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Path-level trace features from Tempo + indicator test vs CPU.

No parent links exist in Tempo's JSON (verified), so paths are service
SETS per trace + per-service durations + status/size attributes:
  - heavy_frac: traces touching checkout/payment/email (buy path, big fanout)
  - browse_frac: traces with only frontend+productcatalog/recommendation/ad
  - spans_per_trace (work per request), root p50/p99, per-service p99
    (checkout_sub_max, productcatalog_leaf), err_frac, mean req bytes.
Per-minute aggregation, then CCF leads + strict spike precursors vs CPU
(from a --cpu-csv export_hpa 60s CSV), blended-vs-path comparison.

Sampling: Tempo keeps 10%; per-minute estimates reweight counts x10
(shares/latencies need no reweighting).

Run from repo root:
    python analytics/trace_features.py --start "2026-09-25 23:50:21" \\
        --cpu-csv /tmp/opencode/newtest2/cpu_1m.csv \\
        --out-dir analytics/data/tracepaths
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests

TEMPO = "http://localhost:9090" if False else "http://localhost:3100"
TEHRAN = pytz.timezone("Asia/Tehran")

HEAVY_SERVICES = {"checkoutservice", "paymentservice", "emailservice"}
BROWSE_ONLY = {"frontend", "productcatalogservice", "recommendationservice",
               "adservice", "currencyservice", "shippingservice", "cartservice"}

STATUS_KEYS = ("http.status_code", "grpc.status_code", "status.code",
               "response_flags", "response.code")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_time(s):
    try:
        return float(s)
    except ValueError:
        return TEHRAN.localize(
            datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp()


def svc_of(batch):
    for a in batch.get("resource", {}).get("attributes", []):
        if a.get("key") == "service.name":
            return str(a.get("value", {}).get("stringValue", "?")).split(".")[0]
    return "?"


def flat_attrs(obj):
    out = {}
    for a in obj.get("attributes", []):
        v = a.get("value", {})
        out[a.get("key", "")] = (v.get("stringValue", "")
                                 or v.get("intValue", "")
                                 or v.get("boolValue", ""))
    return out


def search_slice(start, end, limit):
    r = requests.get(f"{TEMPO}/api/search",
                     params={"start": int(start), "end": int(end),
                             "limit": limit}, timeout=60)
    r.raise_for_status()
    return [t["traceID"] for t in r.json().get("traces", [])]


def fetch_trace(tid, retries=2):
    for _ in range(retries + 1):
        try:
            r = requests.get(f"{TEMPO}/api/traces/{tid}", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1)
    return None


def trace_record(t):
    """Per-trace path record (service sets + durations + errors)."""
    if t is None:
        return None
    spans = []
    for b in t.get("batches", []):
        svc = svc_of(b)
        for s in b.get("scopeSpans", []):
            for sp in s.get("spans", []):
                try:
                    dur = (int(sp["endTimeUnixNano"])
                           - int(sp["startTimeUnixNano"])) / 1e6
                except (KeyError, ValueError):
                    continue
                at = flat_attrs(sp)
                err = False
                for k in STATUS_KEYS:
                    v = str(at.get(k, ""))
                    if v and (v.startswith("5") or v.upper() in
                              ("ERROR", "STATUS_CODE_ERROR", "2")):
                        err = True
                try:
                    req = int(float(at.get("request_size", 0) or 0))
                except ValueError:
                    req = 0
                spans.append((svc, sp.get("name", "")[:80], dur, err, req,
                              int(sp.get("startTimeUnixNano", 0))))
    if not spans:
        return None
    svcs = {s for s, _, _, _, _, _ in spans}
    rec = {"services": svcs, "n_spans": len(spans),
           "root_ms": max(d for _, _, d, _, _, _ in spans),
           "err": any(e for _, _, _, e, _, _ in spans),
           "req_bytes": sum(r for _, _, _, _, r, _ in spans),
           "t0": min(t for _, _, _, _, _, t in spans)}
    for s, _, d, _, _, _ in spans:
        rec.setdefault(f"dur_{s}", []).append(d)
    for k in list(rec):
        if k.startswith("dur_"):
            rec[k] = rec[k]
    return rec


def export_traces(start_ts, end_ts, out_dir, slice_min=15, cap=120):
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    t = start_ts
    sl = 0
    while t < end_ts:
        e = min(t + slice_min * 60, end_ts)
        try:
            ids = search_slice(t, e, cap)
        except Exception as ex:
            log(f"slice {sl}: search failed: {ex}")
            ids = []
        got = 0
        for tid in ids:
            r = trace_record(fetch_trace(tid))
            if r:
                recs.append(r)
                got += 1
        log(f"slice {sl}: {len(ids)} ids -> {got} records "
            f"({datetime.fromtimestamp(t, TEHRAN).strftime('%H:%M')})")
        sl += 1
        t = e
    log(f"total records: {len(recs)}")
    return recs


def to_minute_frame(recs):
    rows = []
    for r in recs:
        ts = pd.Timestamp(r["t0"], unit="ns", tz="UTC").tz_convert(TEHRAN)
        minute = ts.floor("min")
        svcs = r["services"]
        rows.append({
            "minute": minute,
            "heavy": int(bool(svcs & HEAVY_SERVICES)),
            "browse": int(svcs <= (BROWSE_ONLY | {"frontend"})),
            "n_spans": r["n_spans"],
            "root_ms": r["root_ms"],
            "err": int(r["err"]),
            "req_bytes": r["req_bytes"],
            "checkout_ms": max(r.get("dur_checkoutservice", [0]) or [0]),
            "product_ms": max(r.get("dur_productcatalogservice", [0]) or [0]),
            "frontend_ms": max(r.get("dur_frontend", [0]) or [0]),
        })
    d = pd.DataFrame(rows)
    g = d.groupby("minute")
    feat = pd.DataFrame({
        "n_traces": g.size(),
        "heavy_frac": g["heavy"].mean(),
        "browse_frac": g["browse"].mean(),
        "spans_per_trace": g["n_spans"].mean(),
        "root_p50": g["root_ms"].median(),
        "root_p99": g["root_ms"].quantile(0.99),
        "checkout_p99": g["checkout_ms"].quantile(0.99),
        "product_p99": g["product_ms"].quantile(0.99),
        "frontend_p99": g["frontend_ms"].quantile(0.99),
        "err_frac": g["err"].mean(),
        "req_bytes_mean": g["req_bytes"].mean(),
    })
    # reweight counts for 10% sampling (shares/latencies unaffected)
    feat["trace_rps"] = feat["n_traces"] / 60.0 * 10.0
    return feat


def ccf(x, y, max_lag=30):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 60 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    n = len(x)
    return {k: float(np.dot(xz[:n - k], yz[k:]) / (n - k))
            for k in range(max_lag + 1)}


def analyze(feat, cpu_csv, out_dir):
    cpu = pd.read_csv(cpu_csv, parse_dates=["timestamp"])
    cpu["minute"] = cpu["timestamp"].dt.floor("min")
    # trace frame index is tz-aware (Tehran); localize the cpu grid to match,
    # else reindex misses everything and all trace candidates drop out.
    try:
        cpu["minute"] = cpu["minute"].dt.tz_localize("Asia/Tehran")
    except TypeError:
        cpu["minute"] = cpu["minute"].dt.tz_convert("Asia/Tehran")
    keep = ["timestamp", "msname", "cpu_utilization", "rps_total",
            "p99_latency", "active_in"]
    keep = [c for c in keep if c in cpu.columns]
    res_ccf, res_pre = [], []
    for svc, g in cpu.groupby("msname"):
        g = g.sort_values("minute").reset_index(drop=True)
        f = feat.reindex(g["minute"].to_numpy())
        f.index = range(len(f))
        y = g["cpu_utilization"].to_numpy(float)
        if np.nanstd(y) < 1e-3:
            continue
        cands = {"trace_rps": f["trace_rps"].to_numpy(float),
                 "heavy_frac": f["heavy_frac"].ffill().bfill().to_numpy(float),
                 "spans_per_trace": f["spans_per_trace"].ffill().bfill().to_numpy(float),
                 "root_p99": f["root_p99"].ffill().bfill().to_numpy(float),
                 "checkout_p99": f["checkout_p99"].ffill().bfill().to_numpy(float),
                 "err_frac": f["err_frac"].fillna(0.0).to_numpy(float)}
        if "rps_total" in g:
            cands["rps_total(blended)"] = g["rps_total"].to_numpy(float)
        if "p99_latency" in g:
            cands["p99(blended)"] = g["p99_latency"].ffill().bfill().to_numpy(float)
        for name, x in cands.items():
            cc = ccf(x, y)
            if cc is None:
                continue
            kb = max(cc, key=lambda k: abs(cc[k]))
            res_ccf.append(dict(service=svc, candidate=name,
                                best_lead_min=kb, best_corr=round(cc[kb], 3),
                                lag0=round(cc[0], 3),
                                lead1=round(cc[1], 3),
                                lead3=round(cc[3], 3),
                                lead6=round(cc[6], 3)))
        # strict precursors on cpu spikes (delta .2, refractory, quiet-then-cross)
        z = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        sig = z > 2.5
        ons, i, last = [], 0, -10 ** 9
        while i < len(z):
            if sig[i] and i - last > 30 and not sig[max(0, i - 6):i].any():
                ons.append(i)
                last = i
            i += 1
        for name, x in cands.items():
            xs = pd.Series(x).ffill().bfill()
            xz = ((xs - xs.mean()) / (xs.std() + 1e-12)).to_numpy()
            hits, leads = 0, []
            for o in ons:
                q0, q1 = max(0, o - 24), max(0, o - 12)
                if q1 <= q0 or np.nanmax(np.abs(xz[q0:q1])) > 1.0:
                    continue
                over = np.nonzero(xz[q1:o] > 2.0)[0]
                if len(over):
                    hits += 1
                    leads.append(o - (q1 + over[0]))
            res_pre.append(dict(
                service=svc, candidate=name, n_events=len(ons),
                hit_rate=round(hits / max(1, len(ons)), 3),
                median_lead_min=round(float(np.median(leads)), 1) if leads else None))
            log(f"[{svc}] {name}: CCF computed, spikes={len(ons)}, "
                f"hit={hits}/{len(ons)}")
    pd.DataFrame(res_ccf).to_csv(f"{out_dir}/trace_ccf.csv", index=False)
    pd.DataFrame(res_pre).to_csv(f"{out_dir}/trace_precursors.csv", index=False)
    feat.to_csv(f"{out_dir}/trace_path_1m.csv")
    log(f"wrote -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cpu-csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/tracepaths")
    ap.add_argument("--slice-min", type=int, default=15)
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse existing trace_path_1m.csv, only analyze")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.skip_fetch:
        end_ts = time.time() if not args.end else parse_time(args.end)
        recs = export_traces(parse_time(args.start), end_ts, args.out_dir,
                             args.slice_min, args.cap)
        feat = to_minute_frame(recs)
        feat.to_csv(f"{args.out_dir}/trace_path_1m.csv")
    else:
        feat = pd.read_csv(f"{args.out_dir}/trace_path_1m.csv",
                           parse_dates=["minute"], index_col="minute")
    analyze(feat, args.cpu_csv, args.out_dir)


if __name__ == "__main__":
    main()

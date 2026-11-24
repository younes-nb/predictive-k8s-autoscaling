#!/usr/bin/env python3
"""Compare delivered k6 iterations per trace bucket against the scheduled MCR.

Usage:
  python3 analyze_delivery.py --samples samples_X.json.gz --mcr mcr_slice.csv \
      --max-requests 25000 --start-min 529 --test-min 60 [--compress 1] \
      [--json out.json]

Bucket k covers [t0 + k*hold, t0 + (k+1)*hold) with hold = 60/compress seconds,
where t0 is the first iteration timestamp. Expected iterations for bucket k are
round(MCR[(start_min + k) % len(MCR)] * compress), the same formula the k6
generator uses.
"""
import argparse
import collections
import csv
import gzip
import json
from datetime import datetime, timezone


def ts(s):
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, tail = s.split(".", 1)
        frac = tail.split("+")[0].split("-")[0]
        tz = tail[len(frac):]
        s = head + "." + (frac + "000000")[:6] + tz
    return datetime.fromisoformat(s).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--mcr", required=True)
    ap.add_argument("--max-requests", type=float, required=True)
    ap.add_argument("--start-min", type=int, required=True)
    ap.add_argument("--test-min", type=int, default=60)
    ap.add_argument("--compress", type=float, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.mcr)))
    vals = [float(r["http_mcr"]) for r in rows if r.get("http_mcr", "").strip()]
    peak = max(vals)
    n = len(vals)
    hold = 60.0 / a.compress

    expected = []
    for k in range(a.test_min):
        raw = vals[(a.start_min + k) % n]
        expected.append(round(raw / peak * a.max_requests * a.compress))

    metric = None
    for cand in ("iterations_started_total", "iterations"):
        probe = 0
        for line in gzip.open(a.samples, "rt"):
            if '"%s"' % cand in line:
                probe += 1
                if probe >= 1:
                    metric = cand
                    break
        if metric:
            break
    if metric is None:
        print("no iterations metric found in", a.samples)
        return 1

    it = []
    completed = 0
    for line in gzip.open(a.samples, "rt"):
        if '"iterations"' not in line and '"iterations_started_total"' not in line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("metric") == "iterations":
            completed += 1
        if o.get("metric") != metric:
            continue
        t = (o.get("data") or {}).get("time")
        if t:
            it.append(ts(t))
    it.sort()
    if not it:
        print("no iterations found in", a.samples)
        return 1

    t0 = it[0]
    buckets = collections.Counter(int((t - t0) // hold) for t in it)

    print(f"# Delivery check — {a.samples}")
    print(f"- metric: {metric}  count: {len(it)}  completed: {completed}")
    print(f"- first: {datetime.fromtimestamp(t0, tz=timezone.utc):%H:%M:%S}Z")
    print(f"- hold: {hold:g}s  buckets expected: {a.test_min}")
    print()
    print("| bucket | start | expected | delivered | ratio |")
    print("|---|---|---|---|---|")
    tot_exp = tot_del = 0
    out_rows = []
    for k in range(a.test_min):
        e = expected[k]
        d = buckets.get(k, 0)
        tot_exp += e
        tot_del += d
        ratio = (d / e) if e else (1.0 if d == 0 else float("inf"))
        flag = "" if e == 0 else f"{ratio*100:5.1f}%"
        print(f"| {k} | {datetime.fromtimestamp(t0 + k*hold, tz=timezone.utc):%H:%M:%S} "
              f"| {e} | {d} | {flag} |")
        out_rows.append({"bucket": k, "expected": e, "delivered": d,
                         "ratio": round(ratio, 4) if e else None})
    print()
    print(f"- totals: delivered {tot_del} / expected {tot_exp} "
          f"({100*tot_del/tot_exp:.2f}%)")
    beyond = sum(v for k, v in buckets.items() if k >= a.test_min)
    if beyond:
        print(f"- iterations beyond the scheduled window: {beyond}")

    if a.json:
        json.dump({"t0": t0, "hold_s": hold, "expected_total": tot_exp,
                   "delivered_total": tot_del,
                   "delivery_pct": round(100 * tot_del / tot_exp, 2),
                   "buckets": out_rows}, open(a.json, "w"), indent=2)
        print(f"(json → {a.json})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
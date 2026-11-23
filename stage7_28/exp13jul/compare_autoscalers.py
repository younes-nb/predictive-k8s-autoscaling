#!/usr/bin/env python3
"""Compare autoscaler arms on one benchmark (14-Jul 1 h, 30k or 10k).

Usage:
    python3 compare_autoscalers.py <run_dir> <run_dir> <run_dir> [--label A B C]

Emits a markdown report: headline SLO / delivery / resource table, per-service
latency vs replica allocation, and the efficiency frontier. Everything is read
from each run's own ``dataset_*/metrics.csv`` so no live Mimir query is needed.
"""

import argparse
import csv
import glob
import json
import os
import statistics
from datetime import datetime


def ep(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def load(run_dir):
    ds = sorted(glob.glob(os.path.join(run_dir, "dataset_*")))
    if not ds:
        raise SystemExit(f"no dataset in {run_dir}")
    ds = ds[-1]
    rows = list(csv.DictReader(open(os.path.join(ds, "metrics.csv"))))
    meta = json.load(open(os.path.join(ds, "meta.json")))
    params = {}
    p = os.path.join(run_dir, "params.json")
    if os.path.exists(p):
        params = json.load(open(p))
    summary = {}
    s = os.path.join(run_dir, "k6_summary.json")
    if os.path.exists(s):
        summary = json.load(open(s)).get("metrics", {})
    return {"dir": run_dir, "rows": rows, "meta": meta, "params": params, "summary": summary}


def q(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]


def arm_stats(a, cut_s=None):
    rows = a["rows"]
    if cut_s is not None:
        tmax = max(ep(r["timestamp"]) for r in rows)
        rows = [r for r in rows if ep(r["timestamp"]) >= tmax - cut_s]
    c = {"0": 0, "1": 0, "-1": 0}
    for r in rows:
        c[r.get("slo_reward", "-1")] = c.get(r.get("slo_reward", "-1"), 0) + 1
    v, ok, un = c["0"], c["1"], c["-1"]
    p90 = [float(r["front_p90"]) for r in rows if r["front_p90"] not in ("", "nan")]
    p50 = [float(r["front_avg"]) for r in rows if r["front_avg"] not in ("", "nan")]
    m = a["summary"]
    svcs = [k[:-6] for k in rows[0] if k.endswith("&count")] if rows else []
    # Per-timestamp total across services, then averaged over the window.
    tot = [sum(float(r.get(s + "&count") or 0) for s in svcs) for r in rows] if rows else []
    per = [statistics.mean([float(r.get(s + "&count") or 0) for r in rows]) if rows else 0.0
           for s in svcs]
    return {
        "rows": len(rows), "viol": v, "ok": ok, "unlab": un,
        "rate": pct(v, v + ok),
        "outage_aware": pct(v + un, len(rows)),
        "outage": pct(un, len(rows)),
        "p50": q(p90, .5), "p95": q(p90, .95), "avg": statistics.mean(p90) if p90 else float("nan"),
        "front_avg": statistics.mean(p50) if p50 else float("nan"),
        "iters": m.get("iterations", {}).get("count"),
        "it_min": m.get("iterations", {}).get("rate", 0) * 60,
        "req_s": m.get("http_reqs", {}).get("rate", 0),
        "drop": m.get("dropped_iterations", {}).get("count"),
        # The k6 console metric is not in the JSON summary; derive the offered
        # count from dropped + completed when it is absent.
        "sched": (m.get("scheduled_iterations_expected", {}).get("count")
                  or ((m.get("dropped_iterations", {}).get("count") or 0)
                      + (m.get("iterations", {}).get("count") or 0))),
        "fail": m.get("http_req_failed", {}).get("value", 0) * 100,
        "replicas": statistics.mean(tot) if tot else float("nan"),
        "peak": max(tot) if tot else float("nan"),
        "per_svc": dict(zip(svcs, per)),
        "cov": a["meta"].get("coverage_pct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--last-min", type=int, default=0,
                    help="also report the final N minutes (0 = skip)")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    arms = [load(r) for r in args.runs]
    labels = args.label or [a["params"].get("shape", os.path.basename(a["dir"])) for a in arms]

    w = []
    w.append("# Autoscaler comparison\n")
    w.append("Each arm is a 60-min run of the same trace day at the same offered load.\n")

    w.append("\n## 1. Headline\n")
    w.append("| arm | SLO viol | rate | outage-aware | front p50 | front p95 | delivered iters | req/s | dropped % | req failed % | mean replicas | peak |")
    w.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, a in zip(labels, arms):
        s = q_stats = arm_stats(a)
        sched = s["sched"] or 0
        drop = pct(s["drop"] or 0, sched)
        w.append(f"| {lab} | {s['viol']} | {s['rate']:.1f} % | {s['outage_aware']:.1f} % | "
                 f"{s['p50']:.0f} ms | {s['p95']:.0f} ms | {s['iters'] or 0:,} | {s['req_s']:.0f} | "
                 f"{drop:.1f} % | {s['fail']:.2f} % | {s['replicas']:.1f} | {s['peak']:.0f} |")

    if args.last_min:
        w.append(f"\n## 2. Final {args.last_min} minutes\n")
        w.append("| arm | SLO viol | rate | outage-aware | front p50 | front p95 | mean replicas |")
        w.append("|---|---:|---:|---:|---:|---:|---:|")
        for lab, a in zip(labels, arms):
            s = arm_stats(a, args.last_min * 60)
            w.append(f"| {lab} | {s['viol']} | {s['rate']:.1f} % | {s['outage_aware']:.1f} % | "
                     f"{s['p50']:.0f} ms | {s['p95']:.0f} ms | {s['replicas']:.1f} |")

    # per-service: latency vs allocation
    w.append("\n## 3. Per-service latency vs replica allocation (final 30 min)\n")
    svcs = sorted(arm_stats(arms[0])["per_svc"])
    w.append("| service | " + " | ".join(f"p90 {l}" for l in labels) + " | "
             + " | ".join(f"reps {l}" for l in labels) + " |")
    w.append("|---|" + "---:|" * (2 * len(arms)))
    def p90_of(a, svc):
        rows = a["rows"]
        tmax = max(ep(r["timestamp"]) for r in rows)
        rows = [r for r in rows if ep(r["timestamp"]) >= tmax - 1800]
        v = sorted(float(r[svc + "&p90"]) for r in rows if r.get(svc + "&p90") not in (None, "", "nan"))
        return q(v, .9)
    for svc in svcs:
        p = [p90_of(a, svc) for a in arms]
        r = [arm_stats(a, 1800)["per_svc"].get(svc, float("nan")) for a in arms]
        w.append(f"| {svc} | " + " | ".join(f"{x:.0f}" for x in p) + " | "
                 + " | ".join(f"{x:.1f}" for x in r) + " |")

    w.append("\n## 4. Efficiency\n")
    w.append("Violations per 100 mean replicas is a crude price tag: lower is better, "
             "but only meaningful at comparable delivered load.\n")
    w.append("| arm | SLO rate | mean replicas | viol per 100 replicas | req/s per replica |")
    w.append("|---|---:|---:|---:|---:|")
    for lab, a in zip(labels, arms):
        s = arm_stats(a)
        w.append(f"| {lab} | {s['rate']:.1f} % | {s['replicas']:.1f} | "
                 f"{100 * s['rate'] / 100 / max(s['replicas'], 1) * 100:.2f} | "
                 f"{s['req_s'] / max(s['replicas'], 1):.2f} |")

    text = "\n".join(w) + "\n"
    if args.output:
        open(args.output, "w").write(text)
        print("wrote", args.output)
    print(text)


if __name__ == "__main__":
    main()
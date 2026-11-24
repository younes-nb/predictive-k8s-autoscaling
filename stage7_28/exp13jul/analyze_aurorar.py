#!/usr/bin/env python3
"""Behaviour + weakness report for one AURORA-R run.

Everything is read from the run directory, so the report can be regenerated
after the cluster is gone:

  * ``dataset_*/metrics.csv``  -- 5 s ticks, per-service RED + replicas, SLO label
  * ``aurora_r_controller.log`` -- optional snapshot of the controller's own
    decision stream (``kubectl logs deploy/aurora-r``)

Usage:
    python3 analyze_aurorar.py <run_dir> [--label NAME]
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter

REASON_RE = re.compile(r'"reason":\s*"([^"]*)"')
DECISION_RE = re.compile(r'"decision":\s*(\{.*?\}|null),\s*"reason"')
SERVICE_RE = re.compile(r'"service":\s*"([^"]*)"')
DIRECTION_RE = re.compile(r'"direction":\s*"([^"]*)"')


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def load_dataset(run_dir):
    ds = sorted(glob.glob(os.path.join(run_dir, "dataset_*")))
    if not ds:
        return None, None
    path = os.path.join(ds[-1], "metrics.csv")
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        return list(csv.DictReader(fh)), path


def fnum(row, key):
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return v


def analyse_dataset(rows):
    ticks = len(rows)
    labelled = [r for r in rows if fnum(r, "slo_reward") in (0.0, 1.0)]
    viol = sum(1 for r in labelled if fnum(r, "slo_reward") == 0.0)
    nodata = sum(1 for r in rows if fnum(r, "slo_reward") == -1.0)
    p90 = [fnum(r, "front_p90") for r in rows if fnum(r, "front_p90") == fnum(r, "front_p90")]
    p50 = [fnum(r, "front_avg") for r in rows if fnum(r, "front_avg") == fnum(r, "front_avg")]

    svc = {}
    for r in rows:
        for k in r:
            if "&count" not in k:
                continue
            name = k.split("&", 1)[0]
            if name in ("front", "total"):
                continue
            s = svc.setdefault(name, {"count": [], "p99": [], "p90": [], "cpu": []})
            s["count"].append(fnum(r, k))
            for kind in ("p99", "p90", "cpu_usage"):
                kk = f"{name}&{kind}"
                if kk in r:
                    s[kind].append(fnum(r, kk))

    def mean(xs):
        xs = [x for x in xs if x == x]
        return statistics.mean(xs) if xs else float("nan")

    churn = 0
    for s in svc.values():
        c = [x for x in s["count"] if x == x]
        churn += sum(abs(b - a) for a, b in zip(c, c[1:]))

    top = sorted(
        ((n, max([x for x in s["p99"] if x == x] or [float("nan")]),
          mean(s["count"]), max(s["count"] or [0]))
         for n, s in svc.items()),
        key=lambda t: (t[1] if t[1] == t[1] else -1), reverse=True)

    return {
        "ticks": ticks,
        "labelled": len(labelled),
        "violations": viol,
        "nodata": nodata,
        "rate": pct(viol, len(labelled)),
        "outage_aware": pct(viol + nodata, ticks),
        "coverage": pct(len(labelled), ticks),
        "front_p50": mean(p50),
        "front_p90": mean(p90),
        "front_p90_max": max(p90) if p90 else float("nan"),
        "replica_churn": churn,
        "services": len(svc),
        "top": top[:8],
        "mean_replicas": mean([x for s in svc.values() for x in s["count"]]),
    }


def analyse_controller(run_dir):
    path = os.path.join(run_dir, "aurora_r_controller.log")
    if not os.path.exists(path):
        return None
    reasons = Counter()
    actions = []
    decisions = 0
    slo_valid = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"mode"' not in line:
                continue
            decisions += 1
            m = REASON_RE.search(line)
            reason = m.group(1) if m else "unknown"
            # normalise the two long "no evidence" variants
            key = re.sub(r"\d+(\.\d+)?", "N", reason)
            reasons[key] += 1
            if "SLO absent" not in reason:
                slo_valid += 1
            dm = DECISION_RE.search(line)
            if dm and dm.group(1) != "null":
                sm = SERVICE_RE.search(dm.group(1))
                dirn = DIRECTION_RE.search(dm.group(1))
                if sm:
                    actions.append((sm.group(1), dirn.group(1) if dirn else "?"))
    return {
        "decisions": decisions,
        "slo_valid_pct": pct(slo_valid, decisions),
        "reasons": reasons.most_common(8),
        "actions": actions,
        "action_counts": Counter(a[1] for a in actions),
        "action_services": Counter(a[0] for a in actions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    run = os.path.basename(args.run_dir.rstrip("/"))
    label = args.label or run

    rows, ds_path = load_dataset(args.run_dir)
    print(f"# AURORA-R behaviour -- {label}\n")
    print(f"run: `{run}`\n")
    if not rows:
        print("no dataset in this run directory\n")
    else:
        d = analyse_dataset(rows)
        print("## Load evidence\n")
        print("| metric | value |")
        print("|---|---:|")
        print(f"| ticks (5 s) | {d['ticks']} |")
        print(f"| labelled ticks | {d['labelled']} ({d['coverage']:.1f} %) |")
        print(f"| SLO violations | {d['violations']} ({d['rate']:.1f} %) |")
        print(f"| outage-aware rate | {d['outage_aware']:.1f} % |")
        print(f"| front avg p50 | {d['front_p50']:.0f} ms |")
        print(f"| front avg p90 | {d['front_p90']:.0f} ms |")
        print(f"| front worst p90 | {d['front_p90_max']:.0f} ms |")
        print(f"| mean replicas | {d['mean_replicas']:.1f} |")
        print(f"| replica churn (sum abs delta) | {d['replica_churn']:.0f} |")
        print(f"| services in dataset | {d['services']} |")
        print()
        print("## Slowest services (max p99 over the run)\n")
        print("| service | max p99 | mean replicas | max replicas |")
        print("|---|---:|---:|---:|")
        for n, p99, mean_c, max_c in d["top"]:
            print(f"| {n} | {p99:.0f} ms | {mean_c:.1f} | {max_c:.0f} |")
        print()

    c = analyse_controller(args.run_dir)
    if c is None:
        print("## Controller\n\nno `aurora_r_controller.log` snapshot in the run directory\n")
        return
    print("## Controller decision stream\n")
    print("| metric | value |")
    print("|---|---:|")
    print(f"| decisions | {c['decisions']} |")
    print(f"| decisions with a usable SLO | {c['slo_valid_pct']:.1f} % |")
    print(f"| actions | {len(c['actions'])} |")
    for k, v in c["action_counts"].items():
        print(f"| actions {k} | {v} |")
    print()
    print("### Decisions by reason\n")
    print("| reason | count |")
    print("|---|---:|")
    for k, v in c["reasons"]:
        print(f"| {k} | {v} |")
    print()
    if c["actions"]:
        print("### Actions by service\n")
        print("| service | actions |")
        print("|---|---:|")
        for k, v in c["action_services"].most_common():
            print(f"| {k} | {v} |")
        print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Summarize a failure_recorder failures.jsonl into a root-cause view.

Stdlib only. Reads the records written by failure_recorder.py and prints a
markdown report plus a compact JSON summary.

Usage:
  python3 stage7_28/exp13jul/summarize_failures.py <failures.jsonl> [--json out.json]
"""
import argparse
import collections
import json
import os
from datetime import datetime, timezone


def load(path):
    events, beats = [], []
    if not os.path.exists(path):
        return events, beats
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type")
        if t == "FAILURE_EVENT":
            events.append(r)
        elif t in ("HEARTBEAT", "RECORDER_STOP"):
            beats.append(r)
    return events, beats


def utc(epoch):
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%H:%M:%SZ")
    except (TypeError, ValueError, OSError):
        return "?"


def top(counter, n=8):
    return counter.most_common(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    events, beats = load(args.path)
    out = {"path": args.path, "events": len(events), "heartbeats": len(beats)}

    print(f"# Failure summary — {os.path.basename(os.path.dirname(args.path)) or args.path}")
    print()
    print(f"- failure events: **{len(events)}**")
    print(f"- heartbeats: {len(beats)}")
    if not events:
        print("\nNo FAILURE_EVENT records (no service/code crossed the recorder "
              "threshold during the window).")
        if args.json:
            json.dump(out, open(args.json, "w"), indent=2)
        return

    t0 = min(e.get("epoch", 0) for e in events)
    t1 = max(e.get("epoch", 0) for e in events)
    out["window"] = {"start": t0, "end": t1, "duration_s": t1 - t0}
    print(f"- observed window: {utc(t0)} → {utc(t1)} ({int(t1 - t0)} s)")
    kinds = collections.Counter(e.get("kind") for e in events)
    print(f"- kinds: {dict(kinds)}")
    srcs = collections.Counter(e.get("exc_source") for e in events)
    print(f"- exception source: {dict(srcs)}")

    # ---- per service/code ----
    sc = collections.defaultdict(lambda: {"events": 0, "peak_rate": 0.0, "first": None,
                                          "last": None, "max_pods": 0, "max_unready": 0,
                                          "hpa": set()})
    for e in events:
        k = (e.get("service"), e.get("code"))
        d = sc[k]
        d["events"] += 1
        d["peak_rate"] = max(d["peak_rate"], float(e.get("rate_1m") or 0))
        ep = e.get("epoch")
        if ep is not None:
            d["first"] = ep if d["first"] is None else min(d["first"], ep)
            d["last"] = ep if d["last"] is None else max(d["last"], ep)
        d["max_pods"] = max(d["max_pods"], len(e.get("pods") or []))
        d["max_unready"] = max(d["max_unready"], int(e.get("unready_pods") or 0))
        h = e.get("hpa") or {}
        if h.get("current") is not None:
            d["hpa"].add((h.get("current"), h.get("desired"),
                          h.get("min"), h.get("max")))

    print("\n## Service × status code")
    print("| service | code | events | peak 1m rate | max pods | max unready | first | last | HPA cur/des/min/max |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for (svc, code), d in sorted(sc.items(), key=lambda kv: (kv[0][1], -kv[1]["peak_rate"])):
        hpa = ";".join(f"{c}/{de}/{mi}/{ma}" for c, de, mi, ma in sorted(d["hpa"])[:3]) or "-"
        print(f"| {svc} | {code} | {d['events']} | {d['peak_rate']:.3f} | {d['max_pods']} | "
              f"{d['max_unready']} | {utc(d['first'])} | {utc(d['last'])} | {hpa} |")
        rows.append({"service": svc, "code": code, "events": d["events"],
                     "peak_rate_1m": round(d["peak_rate"], 3),
                     "max_pods": d["max_pods"], "max_unready": d["max_unready"],
                     "first": d["first"], "last": d["last"]})
    out["service_code"] = rows

    # ---- classification ----
    cls = collections.Counter()
    for e in events:
        for c in (e.get("classify") or []):
            cls[c] += 1
    out["classify"] = dict(cls)
    print("\n## Classification tags")
    for c, n in top(cls, 15):
        print(f"- `{c}`: {n}")

    # ---- exception classes ----
    exc = collections.Counter()
    exc_by_svc = collections.defaultdict(collections.Counter)
    for e in events:
        for k, v in (e.get("exception_classes") or {}).items():
            n = v if isinstance(v, int) else len(v or [])
            exc[k] += n
            exc_by_svc[e.get("service")][k] += n
    out["exception_classes"] = dict(exc)
    print("\n## Application exception classes (Loki)")
    for k, n in top(exc, 20):
        per = sorted(((cnt.get(k, 0), s) for s, cnt in exc_by_svc.items() if cnt.get(k)),
                     reverse=True)[:3]
        svcs = ", ".join(f"{s}×{c}" for c, s in per)
        print(f"- `{k}`: {n}  ({svcs})")

    # ---- upstream hops ----
    hops = collections.Counter()
    for e in events:
        for h in (e.get("upstream_5xx_hops_1m") or []):
            hops[(h.get("src"), h.get("dst"), h.get("code"),
                  h.get("response_flags"))] += 1
    out["upstream_hops"] = [{"src": s, "dst": d, "code": c, "flags": f, "n": n}
                            for (s, d, c, f), n in hops.most_common(20)]
    print("\n## Upstream 5xx hop attribution")
    for (s, d, c, f), n in top(hops, 12):
        print(f"- {s} → {d} [{c}] flags={f or '-'}: {n} events")

    # ---- scale events ----
    scale = collections.Counter()
    for e in events:
        for s in (e.get("scale_events") or []):
            scale[str(s)[:120]] += 1
    out["scale_events"] = dict(scale)
    if scale:
        print("\n## Scale events co-observed")
        for k, n in top(scale, 10):
            print(f"- {k}: {n}")

    # ---- 5-minute timeline ----
    buckets = collections.defaultdict(collections.Counter)
    for e in events:
        b = int((e.get("epoch", t0) - t0) // 300) * 5
        buckets[b][f"{e.get('service')}:{e.get('code')}"] += 1
    out["timeline_5m"] = {str(b): dict(c) for b, c in sorted(buckets.items())}
    print("\n## 5-minute timeline (event counts)")
    print("| t+min | events | top service:code |")
    print("|---|---|---|")
    for b in sorted(buckets):
        tot = sum(buckets[b].values())
        lead = buckets[b].most_common(3)
        print(f"| {b} | {tot} | {', '.join(f'{k}×{v}' for k, v in lead)} |")

    # ---- heartbeats ----
    if beats:
        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        rest = [x for x in (_num(b.get("restarts_sum")) for b in beats) if x is not None]
        cpu = [x for x in (_num(b.get("node_cpu_max_pct")) for b in beats) if x is not None]
        loki = [b.get("loki_up") for b in beats if b.get("loki_up") is not None]
        print("\n## Heartbeats")
        if rest:
            print(f"- container restarts: start={rest[0]} end={rest[-1]} max={max(rest)}")
        if cpu:
            print(f"- max node CPU: {max(cpu):.1f}%")
        if loki:
            print(f"- loki_up samples: {sum(1 for x in loki if x)}/{len(loki)}")
        out["heartbeat"] = {"restarts_start": rest[0] if rest else None,
                            "restarts_end": rest[-1] if rest else None,
                            "restarts_max": max(rest) if rest else None,
                            "node_cpu_max_pct": max(cpu) if cpu else None,
                            "loki_up": (sum(1 for x in loki if x), len(loki)) if loki else None}

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"\n(json summary → {args.json})")


if __name__ == "__main__":
    main()

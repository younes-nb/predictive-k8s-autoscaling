#!/usr/bin/env python3
"""Backfill per-pod state for FAILURE_EVENT records written before the
failure_recorder selector fix (2026-11-16 08:31:16Z, run
hpa1h25k_jul13_rr_20261116T081238Z).

Those events have correct envelopes, HPA state and exception classes but zero
app_cpu / pod_ws_mb / thr_max / net_*. Mimir still holds the series, so the
values can be re-queried at the event's own timestamp.

Usage:
  python3 backfill_pod_state.py <failures.jsonl> <out.jsonl> [--mimir URL]
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request

NS = "train-ticket"
QUERIES = {
    "app_cpu": ('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",'
                'pod=~"%s",container!="",container!="POD",container!="istio-proxy"}[2m]))'),
    "proxy_cpu": ('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",'
                  'pod=~"%s",container="istio-proxy"}[2m]))'),
    "pod_ws_mb": ('sum by (pod) (container_memory_working_set_bytes{namespace="%s",'
                  'pod=~"%s",container!="",container!="POD"})'),
    "thr_max": ('sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{namespace="%s",'
                'pod=~"%s"}[2m])) / sum by (pod) '
                '(rate(container_cpu_cfs_periods_total{namespace="%s",pod=~"%s"}[2m]))'),
    "net_rx_kbps": ('sum by (pod) (rate(container_network_receive_bytes_total{namespace="%s",'
                    'pod=~"%s"}[2m]))'),
    "net_tx_kbps": ('sum by (pod) (rate(container_network_transmit_bytes_total{namespace="%s",'
                    'pod=~"%s"}[2m]))'),
}


def q(mimir, expr, t):
    url = mimir + "/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": "%.3f" % t})
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.load(r)["data"]["result"]
    except Exception as e:
        sys.stderr.write("warn query: %s\n" % e)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mimir", default="http://127.0.0.1:18080/prometheus")
    ap.add_argument("--only-before", default=None,
                    help="only backfill events with utc < this (default: all zeroed events)")
    a = ap.parse_args()

    n_fixed = n_skip = 0
    with open(a.src) as f, open(a.dst, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "FAILURE_EVENT":
                out.write(json.dumps(r) + "\n")
                continue
            if a.only_before and r.get("utc", "") >= a.only_before:
                out.write(json.dumps(r) + "\n")
                n_skip += 1
                continue
            pods = r.get("pods") or []
            if not pods or all(p.get("app_cpu") for p in pods):
                out.write(json.dumps(r) + "\n")
                n_skip += 1
                continue

            # pod regex from the pod names actually present
            names = sorted({p["pod"] for p in pods})
            regex = "|".join(names)
            t = float(r.get("epoch") or 0)
            maps = {}
            for field, tmpl in QUERIES.items():
                if field == "thr_max":
                    expr = tmpl % (NS, regex, NS, regex)
                else:
                    expr = tmpl % (NS, regex)
                vals = {}
                for s in q(a.mimir, expr, t):
                    pod = (s.get("metric") or {}).get("pod")
                    if pod:
                        vals[pod] = float(s["value"][1])
                maps[field] = vals

            for p in pods:
                pn = p["pod"]
                if "app_cpu" in maps:
                    p["app_cpu"] = round(maps["app_cpu"].get(pn, 0.0), 3)
                if "proxy_cpu" in maps:
                    p["proxy_cpu"] = round(maps["proxy_cpu"].get(pn, 0.0), 3)
                if "pod_ws_mb" in maps:
                    p["pod_ws_mb"] = round(maps["pod_ws_mb"].get(pn, 0.0) / 1048576.0, 1)
                if "thr_max" in maps:
                    p["thr_max"] = round(maps["thr_max"].get(pn, 0.0), 3)
                if "net_rx_kbps" in maps:
                    p["net_rx_kbps"] = round(maps["net_rx_kbps"].get(pn, 0.0) / 1024.0, 1)
                if "net_tx_kbps" in maps:
                    p["net_tx_kbps"] = round(maps["net_tx_kbps"].get(pn, 0.0) / 1024.0, 1)
            r["backfilled"] = True
            n_fixed += 1
            out.write(json.dumps(r) + "\n")
    print("backfilled=%d untouched=%d -> %s" % (n_fixed, n_skip, a.dst))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Post-run Mimir dump: save every metric relevant to an autoscaling experiment.

Run anywhere the Mimir Prometheus API is reachable (in-cluster pod or port-forward):

    PBS_MIMIR_URL=http://mimir.monitoring.svc.cluster.local:8080 \
    python3 mimir_dump.py --start '2026-11-07T15:00:00Z' --end '2026-11-07T17:30:00Z' \
        --out /tmp/<run>/mimir_dump --namespace train-ticket

Writes one long-format CSV per query (each row: timestamp + labels + value) plus
manifest.json recording exactly what was dumped. Bucket/counter series for a run
window are only queryable ~15-20 h after the run ends, so this must run
promptly after each experiment.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

NS_DEFAULT = "train-ticket"

# name -> promql. Namespace is substituted for %s.
QUERIES = [
    # --- scaling state -----------------------------------------------------
    ("replicas_spec", 'sum by (deployment) (kube_deployment_spec_replicas{namespace="%s"})'),
    ("replicas_available", 'sum by (deployment) (kube_deployment_status_replicas_available{namespace="%s"})'),
    ("replicas_ready", 'sum by (deployment) (kube_deployment_status_replicas_ready{namespace="%s"})'),
    ("hpa_current", 'kube_horizontalpodautoscaler_status_current_replicas{namespace="%s"}'),
    ("hpa_desired", 'kube_horizontalpodautoscaler_status_desired_replicas{namespace="%s"}'),
    ("hpa_min", 'max by (horizontalpodautoscaler) (kube_horizontalpodautoscaler_spec_min_replicas{namespace="%s"})'),
    ("hpa_max", 'max by (horizontalpodautoscaler) (kube_horizontalpodautoscaler_spec_max_replicas{namespace="%s"})'),
    ("hpa_target_metric", 'max by (horizontalpodautoscaler, metric_name) (kube_horizontalpodautoscaler_status_target_metric_value{namespace="%s"})'),
    # --- per-pod resources (cAdvisor) --------------------------------------
    ("pod_cpu_1m", 'sum by (pod, container, node) (rate(container_cpu_usage_seconds_total{container!="",container!~"POD|istio-proxy|alloy",namespace="%s"}[1m]))'),
    ("pod_mem_usage", 'sum by (pod, container, node) (container_memory_usage_bytes{container!="",container!~"POD|istio-proxy|alloy",namespace="%s"})'),
    ("pod_mem_working_set", 'sum by (pod, container, node) (container_memory_working_set_bytes{container!="",container!~"POD|istio-proxy|alloy",namespace="%s"})'),
    ("pod_restarts", 'sum by (pod, container) (kube_pod_container_status_restarts_total{namespace="%s"})'),
    ("pod_limits_cpu", 'kube_pod_container_resource_limits{resource="cpu",namespace="%s"}'),
    ("pod_limits_mem", 'kube_pod_container_resource_limits{resource="memory",namespace="%s"}'),
    # --- istio traffic / latency -------------------------------------------
    ("svc_qps_1m", 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination",namespace="%s"}[1m]))'),
    ("edge_qps_1m", 'sum by (source_workload, destination_workload) (rate(istio_requests_total{reporter="destination",namespace="%s"}[1m]))'),
    ("svc_by_code_1m", 'sum by (destination_workload, response_code) (rate(istio_requests_total{reporter="destination",namespace="%s"}[1m]))'),
    ("svc_duration_buckets_1m", 'sum by (destination_workload, le) (rate(istio_request_duration_milliseconds_bucket{reporter="destination",namespace="%s"}[1m]))'),
    ("front_edge_duration_buckets_1m", 'sum by (source_workload, destination_workload, le) (rate(istio_request_duration_milliseconds_bucket{reporter="destination",namespace="%s",destination_workload=~"ts-ui-dashboard"}[1m]))'),
    ("tcp_sent_1m", 'sum by (destination_workload) (rate(istio_tcp_sent_bytes_total{reporter="destination",namespace="%s"}[1m]))'),
    ("tcp_received_1m", 'sum by (destination_workload) (rate(istio_tcp_received_bytes_total{reporter="destination",namespace="%s"}[1m]))'),
    # --- node level ---------------------------------------------------------
    ("node_cpu_1m", 'sum by (node) (rate(container_cpu_usage_seconds_total{container!="",container!~"POD|istio-proxy",namespace!=""}[1m]))'),
    ("node_mem", 'sum by (node) (container_memory_usage_bytes{container!="",container!~"POD|istio-proxy",namespace!=""})'),
    ("node_allocatable", 'kube_node_status_allocatable{resource=~"cpu|memory|pods"}'),
    ("tt_pods_per_node", 'sum by (node) (kube_pod_info{namespace="%s"})'),
    # --- scraping health ----------------------------------------------------
    ("up", 'up{job!=""}'),
]


def parse_time(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return s  # send raw; server may accept epoch/NAT


def dump_query(base_url, qname, query, start, end, step, out_dir, timeout=180, retries=3):
    payload = {"query": query, "start": start, "end": end, "step": step}
    for attempt in range(retries):
        try:
            r = requests.get(
                base_url.rstrip("/") + "/api/v1/query_range",
                params=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            if body.get("status") != "success":
                raise RuntimeError("api error: %s" % body.get("error", body)[:300])
            result = body.get("data", {}).get("result", [])
            break
        except Exception as e:
            if attempt == retries - 1:
                print("FAIL %-38s %s" % (qname, e), flush=True)
                return None
            time.sleep(5 * (attempt + 1))

    label_cols = []
    rows = []
    for series in result:
        labels = series.get("metric", {})
        for k, v in labels.items():
            if k not in label_cols:
                label_cols.append(k)
        for ts, val in series.get("values", []):
            rows.append((labels, float(ts), val))

    labels_sorted = sorted(label_cols)
    path = os.path.join(out_dir, "%s.csv" % qname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + labels_sorted + ["value"])
        for labels, ts, val in rows:
            w.writerow([ts] + [labels.get(c, "") for c in labels_sorted] +
                       [val if val not in ("NaN", None) else ""])
    print("ok   %-38s %6d series %8d rows -> %s"
          % (qname, len(result), len(rows), os.path.basename(path)), flush=True)
    return {"query": query, "path": os.path.basename(path),
            "series": len(result), "rows": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC RFC3339 or epoch (minus a small buffer)")
    ap.add_argument("--end", required=True, help="UTC RFC3339 or epoch (plus a small buffer)")
    ap.add_argument("--out", required=True, help="output dir (created)")
    ap.add_argument("--namespace", default=NS_DEFAULT)
    ap.add_argument("--step", default="5s")
    ap.add_argument("--url", default=os.environ.get("PBS_MIMIR_URL",
                       "http://mimir.monitoring.svc.cluster.local:8080/prometheus"))
    args = ap.parse_args()

    base = args.url.rstrip("/")
    # Mimir API is at <base>/api/v1/ where base is the prometheus URL
    # e.g., http://mimir.monitoring.svc.cluster.local:8080/prometheus -> /api/v1
    if base.endswith("/prometheus"):
        api = base + "/api/v1"
    elif base.endswith("/prometheus/api/v1"):
        api = base
    else:
        api = base.rstrip("/") + "/prometheus/api/v1"
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    start = parse_time(args.start)
    end = parse_time(args.end)

    r = requests.get(api + "/query", params={"query": "up"}, timeout=30)
    r.raise_for_status()
    print("Mimir API reachable: %s (up series: %d)" % (api, len(r.json()["data"]["result"])))

    manifest = {
        "dumped_at": datetime.now(timezone.utc).isoformat() + "Z",
        "mimir_url": api,
        "namespace": args.namespace,
        "start": start, "end": end, "step": args.step,
        "queries": {},
    }
    for i, (qname, tmpl) in enumerate(QUERIES, 1):
        query = tmpl % args.namespace if "%s" in tmpl else tmpl
        info = dump_query(api, "%02d_%s" % (i, qname), query, start, end, args.step, out_dir)
        if info:
            manifest["queries"]["%02d_%s" % (i, qname)] = info

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n_ok = sum(1 for v in manifest["queries"].values() if v["rows"] > 0)
    print("dump complete: %d/%d queries produced rows -> %s" % (n_ok, len(QUERIES), out_dir))
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())

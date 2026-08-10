import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

PROM_SERVICE = "svc/prometheus-stack-kube-prom-prometheus"
PROM_NAMESPACE = "monitoring"
LOCAL_PORT = 9090
DEFAULT_PROM_URL = "http://localhost:9090"
STEP_DEFAULT = 30
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
TEHRAN = timezone(timedelta(hours=3, minutes=30))

DEFAULT_DEPLOYMENTS = [
    "frontend", "recommendationservice", "productcatalogservice", "cartservice",
    "currencyservice", "shippingservice", "emailservice", "paymentservice",
    "checkoutservice", "adservice",
]

METRIC_COLUMNS = [
    "cpa_actual_cpu", "cpa_actual_memory", "cpa_pred_cpu", "cpa_pred_mem",
    "cpa_delta_cpu", "cpa_delta_mem", "cpa_inference_time_s", "cpa_replicas",
]

COLUMN_MAP = {
    "cpa_actual_cpu": "cpu",
    "cpa_actual_memory": "memory",
    "cpa_pred_cpu": "pred_cpu",
    "cpa_pred_mem": "pred_mem",
    "cpa_delta_cpu": "delta_cpu",
    "cpa_delta_mem": "delta_mem",
    "cpa_inference_time_s": "inference_time_s",
    "cpa_replicas": "replicas",
}

CSV_HEADER = "timestamp,cpu,memory,pred_cpu,pred_mem,delta_cpu,delta_mem,inference_time_s,replicas"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull CPA experiment metrics from Prometheus into CSV files"
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start Timestamp (YYYY-MM-DD HH:MM:SS, Tehran)"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End Timestamp (YYYY-MM-DD HH:MM:SS, Tehran)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory to write CSV files"
    )
    parser.add_argument(
        "--prometheus-url", type=str, default=DEFAULT_PROM_URL,
        help="Prometheus API base URL (default localhost:9090 via kubectl port-forward)",
    )
    parser.add_argument(
        "--deployments", type=str, default="all",
        help="Comma-separated deployment list, or 'all'",
    )
    parser.add_argument(
        "--step", type=int, default=STEP_DEFAULT, help="Query step in seconds"
    )
    return parser.parse_args()


def parse_tehran(ts_str):
    return datetime.strptime(ts_str.strip(), TS_FORMAT).replace(tzinfo=TEHRAN).timestamp()


def tehran_wall(row_ts_unix):
    return (
        datetime.fromtimestamp(row_ts_unix, tz=timezone.utc)
        .astimezone(TEHRAN)
        .strftime(TS_FORMAT)
    )


def is_reachable(url):
    try:
        with urllib.request.urlopen(f"{url}/-/healthy", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_port_forward():
    proc = subprocess.Popen(
        ["kubectl", "port-forward", PROM_SERVICE, f"{LOCAL_PORT}:9090", "-n", PROM_NAMESPACE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if is_reachable(DEFAULT_PROM_URL):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("kubectl port-forward to Prometheus did not become ready")


def query_range(url, query, start, end, step):
    params = urllib.parse.urlencode({
        "query": query,
        "start": str(int(start)),
        "end": str(int(end)),
        "step": str(step),
    })
    with urllib.request.urlopen(f"{url}/api/v1/query_range?{params}", timeout=60) as resp:
        payload = json.load(resp)
    if payload["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload["data"]["result"]


def merge_series(series_list):
    merged = {}
    for series in series_list:
        for ts, value in series.get("values", []):
            try:
                merged[int(float(ts))] = float(value)
            except (ValueError, TypeError):
                continue
    return merged


def pull_deployment(url, deployment, start, end, step):
    series = {}
    for metric in METRIC_COLUMNS + ["cpa_row_timestamp"]:
        result = query_range(
            url, f'{metric}{{deployment="{deployment}"}}', start, end, step
        )
        series[metric] = merge_series(result)

    if not series["cpa_row_timestamp"]:
        return []

    union_ts = sorted(set().union(*[s.keys() for s in series.values()]))
    filled = {metric: {} for metric in series}
    last = {metric: math.nan for metric in series}
    for ts in union_ts:
        for metric in series:
            if ts in series[metric]:
                last[metric] = series[metric][ts]
            filled[metric][ts] = last[metric]

    seen_rows = {}
    for ts in union_ts:
        row_ts = filled["cpa_row_timestamp"][ts]
        if not math.isfinite(row_ts) or row_ts <= 0:
            continue
        if row_ts in seen_rows:
            continue
        seen_rows[row_ts] = {COLUMN_MAP[m]: filled[m][ts] for m in METRIC_COLUMNS}

    rows = []
    for row_ts in sorted(seen_rows):
        values = seen_rows[row_ts]
        if any(not math.isfinite(v) for v in values.values()):
            continue
        rows.append((tehran_wall(row_ts), values))
    return rows


def main():
    args = parse_args()
    start = parse_tehran(args.start)
    end = parse_tehran(args.end)

    if args.deployments.strip().lower() == "all":
        deployments = DEFAULT_DEPLOYMENTS
    else:
        deployments = [d.strip() for d in args.deployments.split(",") if d.strip()]

    os.makedirs(args.data_dir, exist_ok=True)

    proc = None
    if not is_reachable(args.prometheus_url):
        print(f"Prometheus not reachable at {args.prometheus_url}, starting kubectl port-forward...")
        proc = start_port_forward()

    try:
        total_rows = 0
        for deployment in deployments:
            rows = pull_deployment(args.prometheus_url, deployment, start, end, args.step)
            out_path = os.path.join(args.data_dir, f"{deployment}.csv")
            with open(out_path, "w") as f:
                f.write(CSV_HEADER + "\n")
                for ts_str, values in rows:
                    line = ",".join(
                        [ts_str] + [f"{values[col]:.4f}" for col in CSV_HEADER.split(",")[1:]]
                    )
                    f.write(line + "\n")
            print(f"{deployment}: {len(rows)} rows -> {out_path}")
            total_rows += len(rows)
        print(f"Total rows: {total_rows}")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()

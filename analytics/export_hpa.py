import argparse
import os
import subprocess
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests

PROMETHEUS_URL = "http://localhost:9090"
PROM_SERVICE = "svc/prometheus-stack-kube-prom-prometheus"
PROM_NAMESPACE = "monitoring"
LOCAL_PORT = 9090
STEP_SECONDS = 60
OUTPUT_DIR = "/proj/k8sautoscaledl-PG0"
OUTPUT_NAME = "hpa_logs.csv"

NAMESPACE = "online-boutique"

QUERIES = {
    "replicas": {
        "query": f'kube_deployment_status_replicas_available{{namespace="{NAMESPACE}"}}',
        "labels": ["deployment"],
    },
    "RPS_HTTP": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="http", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "RPS_GRPC": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="grpc", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "CPU": {
        "query": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container="server"}}[1m])) / sum by (pod) (kube_pod_container_resource_requests{{resource="cpu", namespace="{NAMESPACE}", container="server"}})',
        "labels": ["pod"],
    },
    "Memory": {
        "query": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{NAMESPACE}", container="server"}}) / sum by (pod) (kube_pod_container_resource_requests{{resource="memory", namespace="{NAMESPACE}", container="server"}})',
        "labels": ["pod"],
    },
}

FINAL_COLUMNS = ["timestamp", "msname", "cpu", "memory", "replicas", "mcr"]


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
        if is_reachable(PROMETHEUS_URL):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("kubectl port-forward to Prometheus did not become ready")


def pod_to_deployment(pod):
    parts = pod.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return pod


def fetch_metric_data(metric_name, query_info, start_ts, end_ts, prom_url):
    params = {
        "query": query_info["query"],
        "start": start_ts,
        "end": end_ts,
        "step": f"{STEP_SECONDS}s",
    }
    print(f"  -> Querying {metric_name}...")
    try:
        response = requests.get(
            f"{prom_url}/api/v1/query_range",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        if not results:
            print(f"  No data returned for {metric_name}.")
            return pd.DataFrame(columns=["ts", "entity", metric_name])

        rows = []
        for result in results:
            metric_labels = result["metric"]
            label_vals = {c: metric_labels.get(c, "") for c in query_info["labels"]}
            for value in result["values"]:
                try:
                    val = float(value[1])
                except ValueError:
                    val = np.nan
                rows.append([int(value[0]), *label_vals.values(), val])

        cols = ["ts"] + query_info["labels"] + [metric_name]
        df = pd.DataFrame(rows, columns=cols)
        if "pod" in query_info["labels"]:
            df["deployment"] = df["pod"].map(pod_to_deployment)
            df = (
                df.groupby(["ts", "deployment"], as_index=False)[metric_name]
                .mean()
            )
            df = df.rename(columns={"deployment": "entity"})
        else:
            key_col = query_info["labels"][0]
            df = df.rename(columns={key_col: "entity"})
        return df[["ts", "entity", metric_name]]

    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame(columns=["ts", "entity", metric_name])


def fetch_and_process_data(start_ts, end_ts, prom_url, out_path):
    print(f"Fetching metrics from {prom_url}...")
    tehran_tz = pytz.timezone("Asia/Tehran")
    step_sec = STEP_SECONDS

    proc = None
    if not is_reachable(prom_url):
        print(
            f"Prometheus not reachable at {prom_url}, "
            "starting kubectl port-forward..."
        )
        proc = start_port_forward()

    try:
        raw = {}
        for metric_name, query_info in QUERIES.items():
            raw[metric_name] = fetch_metric_data(
                metric_name, query_info, start_ts, end_ts, prom_url
            )

        n_points = int((end_ts - start_ts) // step_sec) + 1
        grid = [start_ts + k * step_sec for k in range(n_points)]
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(grid), unit="s", utc=True))

        def rekey(df, col):
            s = df.rename(columns={"entity": "msname"}).set_index("ts")[col]
            s.index = pd.to_datetime(pd.Index(s.index), unit="s", utc=True)
            return s.reindex(idx)

        frames = []
        services = set()
        for df in raw.values():
            services.update(df["entity"].unique().tolist())
        services.discard("redis-cart")
        services.discard("")

        for dep in sorted(services):
            full = pd.DataFrame(index=idx)
            if dep in raw["CPU"]["entity"].values:
                full["cpu"] = rekey(raw["CPU"][raw["CPU"]["entity"] == dep], "CPU")
            if dep in raw["Memory"]["entity"].values:
                full["memory"] = rekey(
                    raw["Memory"][raw["Memory"]["entity"] == dep], "Memory"
                )
            if dep in raw["replicas"]["entity"].values:
                full["replicas"] = rekey(
                    raw["replicas"][raw["replicas"]["entity"] == dep], "replicas"
                )
            http = (
                rekey(raw["RPS_HTTP"][raw["RPS_HTTP"]["entity"] == dep], "RPS_HTTP")
                if dep in raw["RPS_HTTP"]["entity"].values
                else pd.Series(0.0, index=idx)
            )
            grpc = (
                rekey(raw["RPS_GRPC"][raw["RPS_GRPC"]["entity"] == dep], "RPS_GRPC")
                if dep in raw["RPS_GRPC"]["entity"].values
                else pd.Series(0.0, index=idx)
            )
            full["mcr"] = http.fillna(0.0) + grpc.fillna(0.0)
            full["replicas"] = (
                full["replicas"].ffill().bfill() if "replicas" in full.columns else 1
            )
            full = full.dropna(subset=["cpu", "memory"], how="all")
            if full.empty:
                print(f"  [{dep}] skipped (no cpu/memory signal)")
                continue
            full["timestamp"] = full.index.tz_convert(tehran_tz).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            full["msname"] = dep
            full["replicas"] = full["replicas"].round().astype(int)
            frames.append(full.round({"cpu": 4, "memory": 4, "mcr": 3})[FINAL_COLUMNS])
            print(f"  [{dep}] {len(full)} rows")

        if not frames:
            raise SystemExit("No service frames could be assembled.")

        final_df = pd.concat(frames, ignore_index=True)
        final_df = final_df.sort_values(["msname", "timestamp"]).reset_index(drop=True)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(final_df)} records x "
              f"{final_df.shape[1]} cols to {out_path}")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export per-service cpu/memory/replicas/incoming-mcr "
                    "at 1-minute granularity."
    )
    parser.add_argument("--start", required=True,
                        help="Start time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--end", required=True,
                        help="End time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--prometheus-url", type=str, default=PROMETHEUS_URL,
                        help="Prometheus API base URL (default localhost:9090 via port-forward)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(OUTPUT_DIR, OUTPUT_NAME),
                        help="Output CSV path (default %(default)s)")

    args = parser.parse_args()
    tehran_tz = pytz.timezone("Asia/Tehran")

    def parse_time_arg(time_str):
        try:
            return float(time_str)
        except ValueError:
            dt_aware = tehran_tz.localize(
                datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            )
            return dt_aware.timestamp()

    start_timestamp = parse_time_arg(args.start)
    end_timestamp = parse_time_arg(args.end)

    fetch_and_process_data(start_timestamp, end_timestamp, args.prometheus_url, args.out)

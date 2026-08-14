import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
import pytz
import subprocess
import time
import urllib.request

PROMETHEUS_URL = "http://localhost:9090"
PROM_SERVICE = "svc/prometheus-stack-kube-prom-prometheus"
PROM_NAMESPACE = "monitoring"
LOCAL_PORT = 9090
STEP_SECONDS = 60
OUTPUT_DIR = "/proj/k8sautoscaledl-PG0"

NAMESPACE = "online-boutique"

QUERIES = {
    "replicas": {
        "query": f'kube_horizontalpodautoscaler_status_current_replicas{{namespace="{NAMESPACE}"}}',
        "labels": ["horizontalpodautoscaler", "hpa"],
    },
    "RPS": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
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


def fetch_metric_data(metric_name, query_info, start_ts, end_ts, tehran_tz, prom_url):
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
            timeout=60,
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])

        if not results:
            print(f"  No data returned for {metric_name}.")
            return pd.DataFrame(
                columns=["Timestamp", "Namespace", "Deployment", metric_name]
            )

        csv_data = []
        for result in results:
            metric_labels = result["metric"]

            namespace = metric_labels.get(
                "namespace",
                metric_labels.get("destination_workload_namespace", NAMESPACE),
            )

            entity = "unknown"
            for candidate in query_info["labels"]:
                if candidate in metric_labels:
                    entity = metric_labels[candidate]
                    break

            is_pod = "pod" in query_info["labels"] and entity != "unknown"
            if is_pod:
                parts = entity.split("-")
                if len(parts) >= 3:
                    entity = "-".join(parts[:-2])

            if entity.endswith("-hpa"):
                entity = entity[: -len("-hpa")]

            for value in result["values"]:
                dt_utc = datetime.fromtimestamp(value[0], pytz.utc)
                dt_tehran = dt_utc.astimezone(tehran_tz)
                timestamp = dt_tehran.strftime("%Y-%m-%d %H:%M:%S")

                try:
                    val = float(value[1])
                except ValueError:
                    val = 0.0

                csv_data.append([timestamp, namespace, entity, val])

        df = pd.DataFrame(
            csv_data, columns=["Timestamp", "Namespace", "Deployment", metric_name]
        )

        if "pod" in query_info["labels"]:
            df = (
                df.groupby(["Timestamp", "Namespace", "Deployment"])[metric_name]
                .mean()
                .reset_index()
            )

        return df

    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame(
            columns=["Timestamp", "Namespace", "Deployment", metric_name]
        )


def fetch_and_process_data(start_ts, end_ts, prom_url):
    print(f"Fetching metrics from {prom_url}...")
    tehran_tz = pytz.timezone("Asia/Tehran")

    proc = None
    if not is_reachable(prom_url):
        print(
            f"Prometheus not reachable at {prom_url}, "
            "starting kubectl port-forward..."
        )
        proc = start_port_forward()

    try:
        dfs = []
        for metric_name, query_info in QUERIES.items():
            df = fetch_metric_data(
                metric_name, query_info, start_ts, end_ts, tehran_tz, prom_url
            )
            if not df.empty:
                dfs.append(df)

        if not dfs:
            print("No data could be fetched across any metrics.")
            return

        print("\nMerging and transforming data...")
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = pd.merge(
                final_df, df, on=["Timestamp", "Namespace", "Deployment"], how="outer"
            )

        expected_cols = [
            "Timestamp", "Namespace", "Deployment",
            "replicas", "RPS", "RPS_HTTP", "RPS_GRPC", "CPU", "Memory",
        ]
        for col in expected_cols:
            if col not in final_df.columns:
                final_df[col] = 0

        final_df = final_df.fillna({
            "replicas": 0, "RPS": 0, "RPS_HTTP": 0, "RPS_GRPC": 0,
            "CPU": 0.0, "Memory": 0.0,
        })

        final_df = final_df[final_df["Deployment"] != "redis-cart"]

        final_df = final_df.rename(columns={
            "Timestamp": "timestamp",
            "Deployment": "msname",
            "RPS_HTTP": "http_mcr",
            "RPS_GRPC": "providerrpc_mcr",
            "CPU": "cpu_utilization",
            "Memory": "memory_utilization",
        })
        final_df = final_df.sort_values(by=["msname", "timestamp"])
        final_df = final_df[
            ["timestamp", "msname", "replicas",
             "http_mcr", "providerrpc_mcr", "cpu_utilization", "memory_utilization"]
        ]

        final_df["cpu_utilization"] = final_df["cpu_utilization"].round(4)
        final_df["memory_utilization"] = final_df["memory_utilization"].round(4)
        final_df["http_mcr"] = final_df["http_mcr"].round(2)
        final_df["providerrpc_mcr"] = final_df["providerrpc_mcr"].round(2)
        final_df["replicas"] = final_df["replicas"].astype(int)

        output_filename = os.path.join(OUTPUT_DIR, "hpa_historical_logs.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(output_filename, index=False)

        print(f"Successfully saved {len(final_df)} records to {output_filename}\n")

        if not final_df.empty:
            print("=" * 40)
            print("GLOBAL DATASET METRICS")
            print("=" * 40)
            print(f"Total Data Points:    {len(final_df)}")
            print(f"Avg Replicas:         {final_df['replicas'].mean():.2f}")
            print(f"Avg CPU:              {final_df['cpu_utilization'].mean():.2%}")
            print(f"Avg Memory:           {final_df['memory_utilization'].mean():.2%}")
            print("=" * 40)
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and normalize HPA and Resource data from Prometheus."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start time (e.g., 'YYYY-MM-DD HH:MM:SS' or Unix timestamp)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End time (e.g., 'YYYY-MM-DD HH:MM:SS' or Unix timestamp)",
    )
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default=PROMETHEUS_URL,
        help="Prometheus API base URL (default localhost:9090 via kubectl port-forward)",
    )

    args = parser.parse_args()
    tehran_tz = pytz.timezone("Asia/Tehran")

    def parse_time_arg(time_str):
        try:
            ts = float(time_str)
            return ts
        except ValueError:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            dt_aware = tehran_tz.localize(dt)
            return dt_aware.timestamp()

    start_timestamp = parse_time_arg(args.start)
    end_timestamp = parse_time_arg(args.end)

    fetch_and_process_data(start_timestamp, end_timestamp, args.prometheus_url)

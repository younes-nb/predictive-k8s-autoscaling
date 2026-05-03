import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
import pytz

PROMETHEUS_URL = "https://prometheus.younesnb.linkpc.net"
STEP_SECONDS = 60

PROM_USER = os.getenv("PROM_USER", "USERNAME")
PROM_PASS = os.getenv("PROM_PASS", "PASSWORD")
NAMESPACE = "online-boutique"

QUERIES = {
    "Replicas": {
        "query": f'kube_horizontalpodautoscaler_status_current_replicas{{namespace="{NAMESPACE}"}}',
        "labels": ["horizontalpodautoscaler", "hpa"],
    },
    "RPS": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "CPU": {
        "query": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container!="POD", container!=""}}[1m])) / sum by (pod) (kube_pod_container_resource_limits{{resource="cpu", namespace="{NAMESPACE}"}})',
        "labels": ["pod"],
    },
    "Memory": {
        "query": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{NAMESPACE}", container!="POD", container!=""}}) / sum by (pod) (kube_pod_container_resource_limits{{resource="memory", namespace="{NAMESPACE}"}})',
        "labels": ["pod"],
    },
}


def fetch_metric_data(metric_name, query_info, start_ts, end_ts, tehran_tz):
    params = {
        "query": query_info["query"],
        "start": start_ts,
        "end": end_ts,
        "step": f"{STEP_SECONDS}s",
    }

    print(f"  -> Querying {metric_name}...")

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params=params,
            auth=(PROM_USER, PROM_PASS),
            verify=True,
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])

        if not results:
            print(f"  ⚠️ No data returned for {metric_name}.")
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
        print(f"❌ Error fetching {metric_name}: {e}")
        return pd.DataFrame(
            columns=["Timestamp", "Namespace", "Deployment", metric_name]
        )


def fetch_and_process_data(start_ts, end_ts):
    print(f"Fetching metrics from {PROMETHEUS_URL}...")
    tehran_tz = pytz.timezone("Asia/Tehran")

    dfs = []
    for metric_name, query_info in QUERIES.items():
        df = fetch_metric_data(metric_name, query_info, start_ts, end_ts, tehran_tz)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("❌ No data could be fetched across any metrics.")
        return

    print("\nMerging and transforming data...")
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(
            final_df, df, on=["Timestamp", "Namespace", "Deployment"], how="outer"
        )

    final_df = final_df.fillna({"Replicas": 0, "RPS": 0, "CPU": 0.0, "Memory": 0.0})

    final_df["CPU"] = final_df["CPU"].clip(0.0, 1.0)
    final_df["Memory"] = final_df["Memory"].clip(0.0, 1.0)

    epsilon = 1e-6
    final_df = final_df.sort_values(by=["Deployment", "Timestamp"])

    final_df["log_rps"] = np.log(final_df["RPS"] + epsilon)

    final_df["log_ret"] = final_df.groupby("Deployment")["log_rps"].diff().fillna(0.0)

    final_df["MCR_Diff"] = np.tanh(final_df["log_ret"])

    final_df = final_df.drop(columns=["log_rps", "log_ret"])

    final_df = final_df[
        [
            "Timestamp",
            "Namespace",
            "Deployment",
            "Replicas",
            "RPS",
            "MCR_Diff",
            "CPU",
            "Memory",
        ]
    ]
    final_df["MCR_Diff"] = final_df["MCR_Diff"].round(5)
    final_df["CPU"] = final_df["CPU"].round(4)
    final_df["Memory"] = final_df["Memory"].round(4)
    final_df["RPS"] = final_df["RPS"].round(2)
    final_df["Replicas"] = final_df["Replicas"].astype(int)

    output_filename = "hpa_historical_logs.csv"
    final_df.to_csv(output_filename, index=False)

    print(f"✅ Successfully saved {len(final_df)} records to {output_filename}\n")

    if not final_df.empty:
        print("=" * 40)
        print("📊  GLOBAL DATASET METRICS")
        print("=" * 40)
        print(f"Total Data Points:    {len(final_df)}")
        print(f"Avg Replicas:         {final_df['Replicas'].mean():.2f}")
        print(f"Avg CPU:              {final_df['CPU'].mean():.2%}")
        print(f"Avg Memory:           {final_df['Memory'].mean():.2%}")
        print("=" * 40)


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

    fetch_and_process_data(start_timestamp, end_timestamp)

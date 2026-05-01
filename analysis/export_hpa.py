import requests
import pandas as pd
from datetime import datetime
import os
import argparse
import pytz

PROMETHEUS_URL = "https://prometheus.younesnb.linkpc.net"
STEP_SECONDS = 60

PROM_USER = os.getenv("PROM_USER", "USERNAME")
PROM_PASS = os.getenv("PROM_PASS", "PASSWORD")


def fetch_hpa_data(start_ts, end_ts):
    query = 'kube_horizontalpodautoscaler_status_current_replicas{namespace="online-boutique"}'

    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": f"{STEP_SECONDS}s",
    }

    print(f"Fetching HPA data from {PROMETHEUS_URL}...")

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
            print(
                "⚠️ Query succeeded, but Prometheus returned no HPA data. Check if the time range has data."
            )
            return

        tehran_tz = pytz.timezone("Asia/Tehran")

        csv_data = []
        for result in results:
            deployment = result["metric"].get("horizontalpodautoscaler") or result[
                "metric"
            ].get("hpa", "unknown")
            namespace = result["metric"].get("namespace", "unknown")

            for value in result["values"]:
                dt_utc = datetime.fromtimestamp(value[0], pytz.utc)
                dt_tehran = dt_utc.astimezone(tehran_tz)

                timestamp = dt_tehran.strftime("%Y-%m-%d %H:%M:%S")
                replicas = int(float(value[1]))

                csv_data.append([timestamp, namespace, deployment, replicas])

        df = pd.DataFrame(
            csv_data, columns=["Timestamp", "Namespace", "Deployment", "Replicas"]
        )

        df = df.sort_values(by=["Timestamp", "Deployment"])

        output_filename = "hpa_historical_logs.csv"
        df.to_csv(output_filename, index=False)
        print(f"✅ Successfully saved {len(df)} records to {output_filename}")

    except requests.exceptions.HTTPError as e:
        print(f"❌ Authentication or Server Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch HPA data from Prometheus.")
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

    fetch_hpa_data(start_timestamp, end_timestamp)

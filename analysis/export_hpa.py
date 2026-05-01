import requests
import pandas as pd
from datetime import datetime, timedelta
import os

PROMETHEUS_URL = "https://prometheus.younesnb.linkpc.net"
MINUTES_TO_FETCH = 60
STEP_SECONDS = 60

PROM_USER = os.getenv("PROM_USER", "your_nginx_username")
PROM_PASS = os.getenv("PROM_PASS", "your_nginx_password")


def fetch_hpa_data():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=MINUTES_TO_FETCH)

    query = 'kube_horizontalpodautoscaler_status_current_replicas{namespace="online-boutique"}'

    params = {
        "query": query,
        "start": start_time.timestamp(),
        "end": end_time.timestamp(),
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
                "⚠️ Query succeeded, but Prometheus returned no HPA data. Check if kube-state-metrics is running."
            )
            return

        csv_data = []
        for result in results:
            deployment = result["metric"].get("hpa", "unknown")
            namespace = result["metric"].get("namespace", "unknown")

            for value in result["values"]:
                timestamp = datetime.fromtimestamp(value[0]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                replicas = int(value[1])
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
    fetch_hpa_data()

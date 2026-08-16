import os
import socket
import time
from datetime import datetime, timedelta, timezone

from prometheus_client import CollectorRegistry, start_http_server

try:
    from prometheus_client import GaugeMetricFamily
except ImportError:
    from prometheus_client.metrics_core import GaugeMetricFamily

EXPERIMENT_METRICS_FILE = os.getenv(
    "EXPERIMENT_METRICS_FILE", "/tmp/experiment_metrics.csv"
)
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "unknown")
POD_NAME = os.getenv("POD_NAME") or socket.gethostname()

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
TEHRAN = timezone(timedelta(hours=3, minutes=30))
CSV_COLUMNS = [
    "timestamp", "cpu", "memory", "pred_cpu", "pred_mem",
    "lower_cpu", "upper_cpu", "lower_mem", "upper_mem",
    "inference_time_s", "replicas",
]

GAUGES = {
    "cpu": ("cpa_actual_cpu", "Current CPU usage normalized to pod limit"),
    "memory": ("cpa_actual_memory", "Current memory usage normalized to pod limit"),
    "pred_cpu": ("cpa_pred_cpu", "Predicted CPU usage (median q50)"),
    "pred_mem": ("cpa_pred_mem", "Predicted memory usage (median q50)"),
    "lower_cpu": ("cpa_lower_cpu", "Conformal lower bound (CPU)"),
    "upper_cpu": ("cpa_upper_cpu", "Conformal upper bound (CPU)"),
    "lower_mem": ("cpa_lower_mem", "Conformal lower bound (Memory)"),
    "upper_mem": ("cpa_upper_mem", "Conformal upper bound (Memory)"),
    "inference_time_s": ("cpa_inference_time_s", "Model inference time in seconds"),
    "replicas": ("cpa_replicas", "Current replica count"),
}


def _to_unix(ts_str):
    try:
        return (
            datetime.strptime(ts_str.strip(), TS_FORMAT)
            .replace(tzinfo=TEHRAN)
            .timestamp()
        )
    except ValueError:
        return 0.0


def read_last_row():
    if not os.path.exists(EXPERIMENT_METRICS_FILE):
        return None
    with open(EXPERIMENT_METRICS_FILE, "r") as f:
        for line in reversed(f.read().splitlines()):
            parts = line.split(",")
            if len(parts) == len(CSV_COLUMNS):
                return dict(zip(CSV_COLUMNS, parts))
    return None


def count_valid_rows():
    if not os.path.exists(EXPERIMENT_METRICS_FILE):
        return 0
    with open(EXPERIMENT_METRICS_FILE, "r") as f:
        return sum(
            1
            for line in f.read().splitlines()
            if line and len(line.split(",")) == len(CSV_COLUMNS)
        )


class CpaMetricsCollector:
    def collect(self):
        labels = (TARGET_DEPLOYMENT, POD_NAME)

        rows_total = GaugeMetricFamily(
            "cpa_metrics_rows_total",
            "Number of valid experiment rows recorded in the CSV",
            labels=["deployment", "pod"],
        )
        rows_total.add_metric(labels, count_valid_rows())
        yield rows_total

        row = read_last_row()
        if row is None:
            return

        row_ts = _to_unix(row["timestamp"])
        since = (time.time() - row_ts) if row_ts else float("nan")

        row_ts_g = GaugeMetricFamily(
            "cpa_row_timestamp",
            "Unix timestamp of the latest experiment CSV row",
            labels=["deployment", "pod"],
        )
        row_ts_g.add_metric(labels, row_ts)
        yield row_ts_g

        since_g = GaugeMetricFamily(
            "cpa_metrics_seconds_since_update",
            "Seconds elapsed since the latest experiment row was written",
            labels=["deployment", "pod"],
        )
        since_g.add_metric(labels, since)
        yield since_g

        for col, (name, help_) in GAUGES.items():
            family = GaugeMetricFamily(name, help_, labels=["deployment", "pod"])
            try:
                family.add_metric(labels, float(row[col]))
            except (ValueError, KeyError):
                continue
            yield family


if __name__ == "__main__":
    registry = CollectorRegistry()
    registry.register(CpaMetricsCollector())
    start_http_server(METRICS_PORT, addr="0.0.0.0", registry=registry)
    print(
        f"metrics-exporter listening on 0.0.0.0:{METRICS_PORT} "
        f"(file={EXPERIMENT_METRICS_FILE}, deployment={TARGET_DEPLOYMENT}, pod={POD_NAME})",
        flush=True,
    )
    while True:
        time.sleep(10)
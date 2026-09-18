import datetime
import os
import json
import time
import sys
import torch
import requests
import config
import model_builder
def query_prometheus(query, is_range=False, params=None):
    endpoint = "query_range" if is_range else "query"
    try:
        response = requests.get(
            f"{config.PROMETHEUS_URL}/api/v1/{endpoint}",
            params=params or {"query": query},
        )
        response.raise_for_status()
        return response.json()["data"]["result"]
    except Exception as e:
        sys.stderr.write(f"Prometheus Error: {e}\n")
        return []
def query_prometheus_range(query, start_ts, end_ts, step_s):
    try:
        response = requests.get(
            f"{config.PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start_ts,
                "end": end_ts,
                "step": f"{int(step_s)}s",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["data"]["result"]
    except Exception as e:
        sys.stderr.write(f"Prometheus Range Error: {e}\n")
        return []


def fetch_current_load():
    """Real CPU/memory for this CPA's deployment from Prometheus.

    The custom-pod-autoscaler operator feeds the scripts a JSON envelope
    with current_load/current_memory. When it is missing or zero (the
    deployed CPA specs declare no metrics queries), fall back to querying
    Prometheus directly for this pod's own container, so the loop still
    reacts to real load instead of scaling on zeros forever.
    """
    dep = config.DEPLOYMENT
    ns = config.NAMESPACE
    try:
        cpu_q = (
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}}[1m]))'
        )
        mem_q = (
            f'sum(container_memory_working_set_bytes{{namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}})'
        )
        req_q = (
            f'sum(kube_pod_container_resource_requests{{resource="cpu",'
            f'namespace="{ns}",pod=~"{dep}-.*",container="server"}})'
        )
        mem_req_q = (
            f'sum(kube_pod_container_resource_requests{{resource="memory",'
            f'namespace="{ns}",pod=~"{dep}-.*",container="server"}})'
        )
        cpu = _scalar(query_prometheus(cpu_q))
        mem = _scalar(query_prometheus(mem_q))
        cpu_req = _scalar(query_prometheus(req_q))
        mem_req = _scalar(query_prometheus(mem_req_q))
        cpu_norm = cpu / cpu_req if cpu_req > 0 else 0.0
        mem_norm = mem / mem_req if mem_req > 0 else 0.0
        return cpu_norm, mem_norm
    except Exception as e:
        sys.stderr.write(f"fetch_current_load error: {e}\n")
        return 0.0, 0.0


def _scalar(result):
    if not result:
        return 0.0
    vals = []
    for s in result:
        for _, v in s.get("values", []):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return sum(vals) if vals else 0.0
def load_state():
    defaults = {
        "history": [],
    }
    if os.path.exists(config.STATE_FILE):
        try:
            with open(config.STATE_FILE, "r") as f:
                loaded = json.load(f)
            for k, v in defaults.items():
                loaded.setdefault(k, v)
            return loaded
        except Exception:
            pass
    return defaults
def save_state(state):
    history = state.get("history", [])[-200:]
    payload = {**state, "history": history}
    try:
        with open(config.STATE_FILE, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        sys.stderr.write(f"State Save Error: {e}\n")
def load_model():
    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH, map_location="cpu")
        model_type = config.MODEL_TYPE or checkpoint.get("model_type", "lstm")
        model = model_builder.build_model(checkpoint, model_type)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        model = model_builder.build_model({}, config.MODEL_TYPE or "lstm")
    model.eval()
    return model
def log_to_file(msg):
    try:
        with open("/tmp/cpa_debug.log", "a") as f:
            f.write(f"{time.ctime()} - {msg}\n")
    except Exception:
        pass
def get_tehran_time():
    utc_now = datetime.datetime.utcnow()
    tehran_offset = datetime.timedelta(hours=3, minutes=30)
    tehran_time = utc_now + tehran_offset
    return tehran_time.strftime("%Y-%m-%d %H:%M:%S")
EXPERIMENT_CSV_COLUMNS = [
    "timestamp",
    "cpu",
    "memory",
    "pred_cpu",
    "pred_mem",
    "threshold",
    "error_bias",
    "inference_time_s",
    "replicas",
]
def log_metrics(
    timestamp,
    curr_cpu,
    curr_mem,
    pred_cpu,
    pred_mem,
    threshold,
    error_bias,
    inf_time,
    replicas,
):
    if not os.path.exists(config.EXPERIMENT_METRICS_FILE):
        with open(config.EXPERIMENT_METRICS_FILE, "w") as f:
            f.write(",".join(EXPERIMENT_CSV_COLUMNS) + "\n")
    with open(config.EXPERIMENT_METRICS_FILE, "a") as f:
        f.write(
            f"{timestamp},{curr_cpu:.4f},{curr_mem:.4f},{pred_cpu:.4f},{pred_mem:.4f},"
            f"{threshold:.4f},{error_bias:.4f},{inf_time:.4f},{replicas}\n"
        )

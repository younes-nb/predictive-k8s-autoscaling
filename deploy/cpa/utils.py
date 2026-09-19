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
        # instant queries -> "value": [ts, "val"]; range queries -> "values": [[ts, "val"], ...]
        if "value" in s:
            try:
                vals.append(float(s["value"][1]))
            except (TypeError, ValueError, IndexError):
                pass
        elif "values" in s:
            for _, v in s.get("values", []):
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
    return sum(vals) if vals else 0.0
def fetch_current_replicas():
    q = (
        f'kube_deployment_status_replicas_available{{namespace="{config.NAMESPACE}",'
        f'deployment="{config.DEPLOYMENT}"}}'
    )
    try:
        result = query_prometheus(q)
        if result:
            return float(result[0]["value"][1])
    except Exception as e:
        sys.stderr.write(f"fetch_current_replicas error: {e}\n")
    return None
def fetch_history():
    n = int(config.WINDOW_SIZE)
    if config.PREPROCESS_APPROACH != "swt" or config.FEATURE_SET != "cpu_mem_both":
        return [], False
    if (config.SWT_LEVEL + 1) + (config.MEM_SWT_LEVEL + 1) != config.INPUT_SIZE:
        return [], False
    try:
        from dataclasses import replace
        import numpy as np
        from preprocessing.swt.config import CFG as SWT_CFG
        from preprocessing.swt.decomposition import decompose_window
    except Exception as e:
        sys.stderr.write(f"History SWT import error: {e}\n")
        return [], False
    try:
        dep = config.DEPLOYMENT
        ns = config.NAMESPACE
        now = int(time.time())
        start = now - n * 60
        cpu_q = (
            f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}}[1m])) / sum by (pod) ('
            f'kube_pod_container_resource_requests{{resource="cpu",namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}})'
        )
        mem_q = (
            f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}}) / sum by (pod) ('
            f'kube_pod_container_resource_requests{{resource="memory",namespace="{ns}",'
            f'pod=~"{dep}-.*",container="server"}})'
        )
        cpu_res = query_prometheus_range(cpu_q, start, now, 60)
        mem_res = query_prometheus_range(mem_q, start, now, 60)
        if not cpu_res or not mem_res:
            return [], False
        def per_minute(res):
            buckets = {}
            for s in res:
                for t, v in s.get("values", []):
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if f == f:
                        buckets.setdefault(int(float(t)), []).append(f)
            out = []
            for t in sorted(buckets):
                vals = buckets[t]
                if vals:
                    out.append(sum(vals) / len(vals))
            return out
        cpu = per_minute(cpu_res)[-n:]
        mem = per_minute(mem_res)[-n:]
        if len(cpu) < n or len(mem) < n:
            return [], False
        cpu = np.round(np.asarray(cpu, dtype=np.float64), 2)
        mem = np.round(np.asarray(mem, dtype=np.float64), 2)
        if not (np.isfinite(cpu).all() and np.isfinite(mem).all()):
            return [], False
        cpu_ch = decompose_window(
            cpu, replace(SWT_CFG, SWT_LEVEL=config.SWT_LEVEL)
        )
        mem_ch = decompose_window(
            mem, replace(SWT_CFG, SWT_LEVEL=config.MEM_SWT_LEVEL)
        )
        if cpu_ch is None or mem_ch is None:
            return [], False
        X = np.concatenate([cpu_ch.T, mem_ch.T], axis=1)
        if X.shape != (n, config.INPUT_SIZE):
            return [], False
        return X.astype(float).tolist(), True
    except Exception as e:
        sys.stderr.write(f"fetch_history error: {e}\n")
        return [], False
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

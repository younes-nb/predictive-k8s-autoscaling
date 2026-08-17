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


def log_metrics(
    timestamp,
    curr_cpu,
    curr_mem,
    pred_cpu,
    pred_mem,
    delta_cpu,
    delta_mem,
    inf_time,
    replicas,
):
    if not os.path.exists(config.EXPERIMENT_METRICS_FILE):
        with open(config.EXPERIMENT_METRICS_FILE, "w") as f:
            f.write(
                "timestamp,cpu,memory,pred_cpu,pred_mem,delta_cpu,delta_mem,inference_time_s,replicas\n"
            )
    with open(config.EXPERIMENT_METRICS_FILE, "a") as f:
        f.write(
            f"{timestamp},{curr_cpu:.4f},{curr_mem:.4f},{pred_cpu:.4f},{pred_mem:.4f},{delta_cpu:.4f},{delta_mem:.4f},{inf_time:.4f},{replicas}\n"
        )

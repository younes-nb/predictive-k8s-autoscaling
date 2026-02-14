import datetime
import os
import json
import time
import sys
import torch
import requests
import config
from common.models import RNNForecaster


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
    if os.path.exists(config.STATE_FILE):
        try:
            with open(config.STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "history": [],
        "prev_threshold": config.BASE_THRESHOLD,
        "prev_sigma": 0.0,
        "last_uncertainty_time": 0,
    }


def save_state(history, prev_threshold, prev_sigma, last_time):
    now = time.time()
    valid_history = [
        x
        for x in history
        if x["time"] > (now - config.STABILIZATION_WINDOW_SECONDS - 60)
    ]
    try:
        with open(config.STATE_FILE, "w") as f:
            json.dump(
                {
                    "history": valid_history,
                    "prev_threshold": prev_threshold,
                    "prev_sigma": prev_sigma,
                    "last_uncertainty_time": last_time,
                },
                f,
            )
    except Exception as e:
        sys.stderr.write(f"State Save Error: {e}\n")


def load_model():
    model = RNNForecaster(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        horizon=config.HORIZON,
        rnn_type=config.RNN_TYPE,
        bidirectional=config.BIDIRECTIONAL,
    )
    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_adaptive_threshold(model, x_window):
    model.train()
    x_batch = x_window.repeat(config.MC_REPEATS, 1, 1)
    with torch.no_grad():
        preds = model(x_batch)

    sigma = preds[:, 0].std().item()

    raw_threshold = config.BASE_THRESHOLD - (config.K_FACTOR * sigma)
    clamped_threshold = max(
        config.MIN_THRESHOLD, min(config.MAX_THRESHOLD, raw_threshold)
    )

    return clamped_threshold, sigma


def log_to_file(msg):
    try:
        with open("/tmp/cpa_debug.log", "a") as f:
            f.write(f"{time.ctime()} - {msg}\n")
    except:
        pass


def get_tehran_time():
    utc_now = datetime.datetime.utcnow()
    tehran_offset = datetime.timedelta(hours=3, minutes=30)
    tehran_time = utc_now + tehran_offset
    return tehran_time.strftime("%Y-%m-%d %H:%M:%S")


def log_metrics(timestamp, curr_cpu, pred_cpu, threshold, sigma, inf_time, replicas):
    if not os.path.exists(config.EXPERIMENT_METRICS_FILE):
        with open(config.EXPERIMENT_METRICS_FILE, "w") as f:
            f.write(
                "timestamp_tehran,current_cpu_60th,predicted_cpu_max,threshold,sigma,inference_time_s,current_replicas\n"
            )
    with open(config.EXPERIMENT_METRICS_FILE, "a") as f:
        f.write(
            f"{timestamp},{curr_cpu:.4f},{pred_cpu:.4f},{threshold:.4f},{sigma:.4f},{inf_time:.4f},{replicas}\n"
        )

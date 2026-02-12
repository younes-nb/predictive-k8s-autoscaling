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
        "last_uncertainty_time": 0,
    }


def save_state(history, prev_threshold, last_time):
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
                    "last_uncertainty_time": last_time,
                },
                f,
            )
    except Exception as e:
        sys.stderr.write(f"State Save Error: {e}\n")


def load_model():
    model = RNNForecaster(
        config.INPUT_SIZE,
        config.HIDDEN_SIZE,
        config.NUM_LAYERS,
        config.DROPOUT,
        config.HORIZON,
        config.BIDIRECTIONAL,
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
    threshold = config.BASE_THRESHOLD - (config.K_FACTOR * sigma)
    return max(config.MIN_THRESHOLD, min(config.MAX_THRESHOLD, threshold))

import sys
import json
import time
import torch
import numpy as np
import config
import utils


def main():
    try:
        data = json.loads(sys.stdin.read())
        history_metrics = data.get("metrics", [])
        use_prediction = data.get("use_prediction", False)
        current_load = data.get("current_load", 0.0)
        current_replicas = data.get("current_replicas", 1)
    except:
        print(json.dumps({"targetReplicas": 1}))
        sys.exit(0)

    state = utils.load_state()
    adaptive_threshold = state["prev_threshold"]
    rec_history = state["history"]
    last_uncertainty_time = state["last_uncertainty_time"]

    now = time.time()
    mode = "Reactive"

    if use_prediction:
        if (now - last_uncertainty_time) >= config.UNCERTAINTY_INTERVAL_SECONDS:
            x_tensor = (
                torch.tensor(history_metrics).float().view(1, 60, config.INPUT_SIZE)
            )

            model = utils.load_model()
            adaptive_threshold = utils.get_adaptive_threshold(model, x_tensor)
            last_uncertainty_time = now
            mode = "Predictive (Updated)"
        else:
            mode = "Predictive (Cached)"
    else:
        adaptive_threshold = config.BASE_THRESHOLD

    raw_desired = int(np.ceil(current_replicas * (current_load / adaptive_threshold)))
    raw_desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, raw_desired))

    rec_history.append({"time": now, "replicas": raw_desired})
    window = [
        x["replicas"]
        for x in rec_history
        if x["time"] > (now - config.STABILIZATION_WINDOW_SECONDS)
    ]

    final_rec = max(window) if raw_desired < current_replicas else raw_desired

    utils.save_state(rec_history, adaptive_threshold, last_uncertainty_time)

    print(
        json.dumps(
            {
                "targetReplicas": int(final_rec),
                "logs": f"Mode: {mode}, Load: {current_load:.2f}, Thr: {adaptive_threshold:.3f}, Rec: {final_rec}",
            }
        )
    )


if __name__ == "__main__":
    main()

import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils


def log_to_file(msg):
    with open("/tmp/cpa_debug.log", "a") as f:
        f.write(f"{time.ctime()} - {msg}\n")


def main():
    try:
        raw_input = sys.stdin.read()
        log_to_file(f"RAW INPUT RECEIVED: {raw_input[:100]}...")  # Log first 100 chars

        if not raw_input:
            log_to_file("ERROR: Empty input received")
            print(json.dumps({"targetReplicas": 1}))
            return

        data = json.loads(raw_input)

        history_metrics = data.get("metrics", [])
        use_prediction = data.get("use_prediction", False)
        current_load = float(data.get("current_load", 0.0))
        current_replicas = int(data.get("current_replicas", 1))

        log_to_file(
            f"PARSED DATA: Load={current_load}, Reps={current_replicas}, Pred={use_prediction}"
        )

        state = utils.load_state()
        adaptive_threshold = state["prev_threshold"]
        rec_history = state["history"]
        last_uncertainty_time = state["last_uncertainty_time"]

        now = time.time()
        mode = "Reactive"

        if use_prediction:
            if (now - last_uncertainty_time) >= config.UNCERTAINTY_INTERVAL_SECONDS:
                if len(history_metrics) >= 60:
                    x_tensor = (
                        torch.tensor(history_metrics)
                        .float()
                        .view(1, 60, config.INPUT_SIZE)
                    )
                    model = utils.load_model()
                    adaptive_threshold = utils.get_adaptive_threshold(model, x_tensor)
                    last_uncertainty_time = now
                    mode = "Predictive (Updated)"
                else:
                    mode = "Predictive (Waiting for data)"
                    log_to_file(f"WAITING: History length is {len(history_metrics)}")
            else:
                mode = "Predictive (Cached)"
        else:
            adaptive_threshold = config.BASE_THRESHOLD
            log_to_file("FALLBACK: use_prediction was False")

        safe_threshold = adaptive_threshold if adaptive_threshold > 0 else 0.75

        raw_desired = int(np.ceil(current_replicas * (current_load / safe_threshold)))
        raw_desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, raw_desired))

        rec_history.append({"time": now, "replicas": raw_desired})
        window = [
            x["replicas"]
            for x in rec_history
            if x["time"] > (now - config.STABILIZATION_WINDOW_SECONDS)
        ]

        final_rec = raw_desired if raw_desired > current_replicas else max(window)

        utils.save_state(rec_history, adaptive_threshold, last_uncertainty_time)

        output = {
            "targetReplicas": int(final_rec),
            "logs": f"Mode: {mode}, Load: {current_load:.2f}, Thr: {adaptive_threshold:.3f}, Rec: {final_rec}",
        }

        log_to_file(f"DECISION: {json.dumps(output)}")
        print(json.dumps(output))

    except Exception as e:
        log_to_file(f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}")
        print(json.dumps({"targetReplicas": 1}))


if __name__ == "__main__":
    main()

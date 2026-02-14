import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils


def log_to_file(msg):
    try:
        with open("/tmp/cpa_debug.log", "a") as f:
            f.write(f"{time.ctime()} - {msg}\n")
    except:
        pass


def main():
    try:
        raw_input = sys.stdin.read()
        log_to_file(f"RAW INPUT RECEIVED: {raw_input[:150]}...")

        if not raw_input:
            log_to_file("ERROR: Empty input received from stdin")
            print(json.dumps({"targetReplicas": 1, "logs": "Empty input"}))
            return

        envelope = json.loads(raw_input)

        metrics_list = envelope.get("metrics", [])
        if not metrics_list:
            raise ValueError("No metrics found in CPA envelope")

        inner_json_str = metrics_list[0].get("value", "{}")

        data = json.loads(inner_json_str)

        history_metrics = data.get("metrics", [])
        use_prediction = data.get("use_prediction", False)
        current_load = float(data.get("current_load", 0.0))
        current_replicas = int(data.get("current_replicas", 1))

        log_to_file(
            f"PARSED SUCCESS: Load={current_load}, Reps={current_replicas}, Pred={use_prediction}"
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
        error_msg = f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}"
        log_to_file(error_msg)
        print(json.dumps({"targetReplicas": 1, "logs": f"Error: {str(e)}"}))


if __name__ == "__main__":
    main()

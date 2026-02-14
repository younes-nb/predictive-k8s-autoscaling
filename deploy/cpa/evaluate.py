import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            raise ValueError("Received empty input from metric.py")

        data = json.loads(raw_input)

        history_metrics = data.get("metrics", [])
        use_prediction = data.get("use_prediction", False)
        current_load = data.get("current_load", 0.0)
        current_replicas = data.get("current_replicas", 1)

        sys.stderr.write(
            f"INPUT_DATA: Load={current_load:.4f}, Replicas={current_replicas}, UsePred={use_prediction}\n"
        )

    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR (Input Parsing): {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        print(json.dumps({"targetReplicas": 1}))
        sys.exit(0)

    try:
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
            else:
                mode = "Predictive (Cached)"
        else:
            adaptive_threshold = config.BASE_THRESHOLD

        raw_desired = int(
            np.ceil(
                current_replicas * (float(current_load) / float(adaptive_threshold))
            )
        )
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
        print(json.dumps(output))

        sys.stderr.write(
            f"DEBUG: Mode={mode}, Threshold={adaptive_threshold:.4f}, Load={current_load:.4f}, Target={final_rec}\n"
        )
        sys.stderr.flush()

    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR (Logic): {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        print(json.dumps({"targetReplicas": int(current_replicas)}))
        sys.stderr.flush()


if __name__ == "__main__":
    main()

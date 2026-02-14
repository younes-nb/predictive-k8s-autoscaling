import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils
import datetime
import os

CSV_FILE = "/tmp/experiment_metrics.csv"


def get_tehran_time():
    utc_now = datetime.datetime.utcnow()
    tehran_offset = datetime.timedelta(hours=3, minutes=30)
    tehran_time = utc_now + tehran_offset
    return tehran_time.strftime("%Y-%m-%d %H:%M:%S")


def log_metrics(timestamp, curr_cpu, pred_cpu, threshold, inf_time, replicas):
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w") as f:
            f.write(
                "timestamp_tehran,current_cpu_60th,predicted_cpu_max,threshold,inference_time_s,current_replicas\n"
            )

    with open(CSV_FILE, "a") as f:
        f.write(
            f"{timestamp},{curr_cpu:.4f},{pred_cpu:.4f},{threshold:.4f},{inf_time:.4f},{replicas}\n"
        )


def main():
    t_start_eval = time.time()
    try:
        raw_input = sys.stdin.read()

        if not raw_input:
            utils.log_to_file("ERROR: Empty input received from stdin")
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

        metric_duration = float(data.get("duration_seconds", 0.0))

        state = utils.load_state()
        adaptive_threshold = state["prev_threshold"]
        rec_history = state["history"]
        last_uncertainty_time = state["last_uncertainty_time"]

        now = time.time()
        mode = "Reactive"

        predicted_load_max = 0.0

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

                    with torch.no_grad():
                        raw_preds = model(x_tensor)
                        if isinstance(raw_preds, tuple):
                            preds_tensor = raw_preds[0]
                        else:
                            preds_tensor = raw_preds

                        predicted_load_max = torch.max(preds_tensor).item()

                    last_uncertainty_time = now
                    mode = "Predictive (Updated)"
                else:
                    mode = "Predictive (Waiting for data)"
                    utils.log_to_file(
                        f"WAITING: History length is {len(history_metrics)}"
                    )
            else:
                mode = "Predictive (Cached)"
        else:
            adaptive_threshold = config.BASE_THRESHOLD
            utils.log_to_file("FALLBACK: use_prediction was False")

        safe_threshold = adaptive_threshold if adaptive_threshold > 0 else 0.75

        if mode.startswith("Predictive") and predicted_load_max > 0:
            load_to_scale_on = max(current_load, predicted_load_max)
        else:
            load_to_scale_on = current_load

        raw_desired = int(
            np.ceil(current_replicas * (load_to_scale_on / safe_threshold))
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

        t_end_eval = time.time()
        total_inference_time = metric_duration + (t_end_eval - t_start_eval)

        log_metrics(
            timestamp=get_tehran_time(),
            curr_cpu=current_load,
            pred_cpu=predicted_load_max,
            threshold=adaptive_threshold,
            inf_time=total_inference_time,
            replicas=current_replicas,
        )

        output = {
            "targetReplicas": int(final_rec),
            "logs": f"Mode: {mode}, Load: {load_to_scale_on:.2f}, PredMax: {predicted_load_max:.2f}, Thr: {adaptive_threshold:.3f}, Rec: {final_rec}",
        }

        sys.stdout.write(json.dumps(output))

    except Exception as e:
        error_msg = f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}"
        utils.log_to_file(error_msg)
        print(json.dumps({"targetReplicas": 1, "logs": f"Error: {str(e)}"}))


if __name__ == "__main__":
    main()

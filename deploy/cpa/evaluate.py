import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils


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
        model_sigma = state.get("prev_sigma", 0.0)
        rec_history = state["history"]
        last_uncertainty_time = state["last_uncertainty_time"]

        now = time.time()
        mode = "Reactive"
        predicted_load_max = 0.0

        if use_prediction and len(history_metrics) >= 60:
            x_tensor = (
                torch.tensor(history_metrics).float().view(1, 60, config.INPUT_SIZE)
            )
            model = utils.load_model()

            with torch.no_grad():
                model.eval()
                raw_preds = model(x_tensor)
                preds_tensor = (
                    raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
                )
                predicted_load_max = torch.max(preds_tensor).item()

            if (now - last_uncertainty_time) >= config.UNCERTAINTY_INTERVAL_SECONDS:
                adaptive_threshold, model_sigma = utils.get_adaptive_threshold(
                    model, x_tensor
                )
                last_uncertainty_time = now
                mode = "Predictive (New Threshold)"
            else:
                mode = "Predictive (Cached Threshold)"

        elif use_prediction:
            mode = "Predictive (Waiting for data)"
        else:
            adaptive_threshold = config.BASE_THRESHOLD
            model_sigma = 0.0

        safe_threshold = adaptive_threshold if adaptive_threshold > 0 else 0.75
        load_to_scale_on = max(current_load, predicted_load_max)

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

        utils.save_state(
            rec_history, adaptive_threshold, model_sigma, last_uncertainty_time
        )

        t_end_eval = time.time()
        total_inference_time = metric_duration + (t_end_eval - t_start_eval)

        utils.log_metrics(
            utils.get_tehran_time(),
            current_load,
            predicted_load_max,
            adaptive_threshold,
            model_sigma,
            total_inference_time,
            current_replicas,
        )

        output = {
            "targetReplicas": int(final_rec),
            "logs": f"Mode: {mode}, Load: {load_to_scale_on:.2f}, Pred: {predicted_load_max:.2f}, Thr: {adaptive_threshold:.3f}, Sigma: {model_sigma:.4f}",
        }
        sys.stdout.write(json.dumps(output))

    except Exception as e:
        utils.log_to_file(f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}")
        print(json.dumps({"targetReplicas": 1, "logs": f"Error: {str(e)}"}))


if __name__ == "__main__":
    main()

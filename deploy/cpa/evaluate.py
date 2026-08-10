import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils
import online_correction


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
        current_memory = float(data.get("current_memory", 0.0))
        current_replicas = int(data.get("current_replicas", 1))
        metric_duration = float(data.get("duration_seconds", 0.0))

        state = utils.load_state()
        rec_history = state["history"]

        now = time.time()
        mode = "Reactive"
        predicted_load_final = 0.0
        predicted_memory_final = 0.0
        delta_load = 0.0
        delta_mem = 0.0

        if config.RESIDUAL_CORRECTION:
            online_correction.finalize(state, now, current_load, current_memory)

        if use_prediction and len(history_metrics) >= config.WINDOW_SIZE:
            x_tensor = (
                torch.tensor(history_metrics)
                .float()
                .view(1, config.WINDOW_SIZE, config.INPUT_SIZE)
            )
            model = utils.load_model()

            with torch.no_grad():
                model.eval()
                raw_preds = model(x_tensor)
                preds_tensor = (
                    raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
                )
                preds_tensor = torch.clamp(preds_tensor, min=0.0, max=1.0)
                preds_tensor = torch.round(preds_tensor * 100) / 100
                if config.NUM_TARGETS > 1:
                    predicted_load_final = preds_tensor[0, -1, 0].item()
                    predicted_memory_final = preds_tensor[0, -1, 1].item()
                else:
                    predicted_load_final = preds_tensor[0, -1].item()

            if config.RESIDUAL_CORRECTION:
                online_correction.record_prediction(
                    state, now, predicted_load_final, predicted_memory_final
                )
                delta_load, delta_mem = online_correction.compute_delta(state)
                predicted_load_final = min(1.0, predicted_load_final + delta_load)
                if config.NUM_TARGETS > 1:
                    predicted_memory_final = min(
                        1.0, predicted_memory_final + delta_mem
                    )

            predicted_load_final = round(predicted_load_final, 2)
            if config.NUM_TARGETS > 1:
                predicted_memory_final = round(predicted_memory_final, 2)

            mode = "Predictive"
        elif use_prediction:
            mode = "Predictive (Waiting for data)"

        is_predicting = mode.startswith("Predictive") and predicted_load_final > 0
        safe_threshold = config.BASE_THRESHOLD

        if is_predicting:
            cpu_to_scale = predicted_load_final
            mem_to_scale = predicted_memory_final
        else:
            cpu_to_scale = current_load
            mem_to_scale = current_memory

        desired_cpu = current_replicas * (cpu_to_scale / safe_threshold)
        raw_desired = int(np.ceil(desired_cpu))

        if config.NUM_TARGETS > 1:
            desired_mem = current_replicas * (mem_to_scale / safe_threshold)
            raw_desired = max(raw_desired, int(np.ceil(desired_mem)))

        raw_desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, raw_desired))

        rec_history.append({"time": now, "replicas": raw_desired})
        window = [
            x["replicas"]
            for x in rec_history
            if x["time"] > (now - config.STABILIZATION_WINDOW_SECONDS)
        ]
        final_rec = (
            raw_desired
            if raw_desired > current_replicas
            else (max(window) if window else raw_desired)
        )

        utils.save_state(state)

        t_end_eval = time.time()
        total_inference_time = metric_duration + (t_end_eval - t_start_eval)

        utils.log_metrics(
            utils.get_tehran_time(),
            current_load,
            current_memory,
            predicted_load_final,
            predicted_memory_final,
            delta_load,
            delta_mem,
            total_inference_time,
            current_replicas,
        )

        logs = (
            f"Mode: {mode}, Load: {cpu_to_scale:.2f}, Mem: {mem_to_scale:.2f}, "
            f"PredLoad: {predicted_load_final:.2f}, PredMem: {predicted_memory_final:.2f}, "
            f"DeltaLoad: {delta_load:.4f}, DeltaMem: {delta_mem:.4f}"
        )
        output = {
            "targetReplicas": int(final_rec),
            "logs": logs,
        }
        sys.stdout.write(json.dumps(output))

    except Exception as e:
        utils.log_to_file(f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}")
        print(json.dumps({"targetReplicas": 1, "logs": f"Error: {str(e)}"}))


if __name__ == "__main__":
    main()

import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils
import model_builder
import adaptive_threshold


def _fail_conditions():
    return {"AbleToScale": False, "ScalingActive": False, "ScalingLimited": False}


def _last_target():
    try:
        return int(utils.load_state().get("last_target", 1))
    except Exception:
        return 1


def _scale_up_ceiling(current_replicas):
    """Max replicas reachable in one eval interval under the HPA default
    scaleUp policy: at most max(100% of current, SCALE_UP_MAX_PODS) pods per
    15s period."""
    rep = int(current_replicas)
    periods = max(
        1, int(config.EVAL_INTERVAL_SECONDS // config.SCALE_UP_PERIOD_SECONDS)
    )
    for _ in range(periods):
        rep = min(config.MAX_REPLICAS, rep + max(int(rep * config.SCALE_UP_MAX_PERCENT / 100), config.SCALE_UP_MAX_PODS))
    return rep


def main():
    t_start_eval = time.time()
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            utils.log_to_file("ERROR: Empty input received from stdin")
            print(
                json.dumps(
                    {
                        "targetReplicas": _last_target(),
                        "logs": "Empty input",
                        "conditions": _fail_conditions(),
                    }
                )
            )
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
        # Drop orphaned conformal keys left behind by older versions.
        state.pop("conformal", None)
        state.pop("conformal_pending", None)
        rec_history = state["history"]

        now = time.time()
        mode = "Reactive"
        predicted_load_final = 0.0
        predicted_memory_final = 0.0

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
                # Point-forecast output. Expected: (1, H, T) or (H, T).
                # Legacy 4D quantile output (1, H, T, Q) falls back to the
                # mean over the quantile dim (no q50 / conformal is used;
                # the checkpoint should be a point model).
                preds_tensor = torch.round(preds_tensor * 100) / 100
                if preds_tensor.dim() == 4:
                    point = preds_tensor[0, -1, :, :].mean(dim=-1)  # (T,)
                elif preds_tensor.dim() == 3:
                    point = preds_tensor[0, -1]  # (T,)
                elif preds_tensor.dim() == 2:
                    point = preds_tensor[-1]  # (T,)
                else:
                    point = preds_tensor.flatten()  # (T,)

                if config.NUM_TARGETS > 1:
                    predicted_load_final = float(point[0].item())
                    predicted_memory_final = float(point[1].item())
                else:
                    predicted_load_final = float(point.flatten()[0].item())
                    predicted_memory_final = 0.0

            # Scale directly on the point prediction (no conformal interval).
            cpu_to_scale = predicted_load_final
            mem_to_scale = predicted_memory_final

            mode = "Predictive"
        elif use_prediction:
            mode = "Predictive (Waiting for data)"

        is_predicting = mode.startswith("Predictive") and predicted_load_final > 0
        # Adaptive threshold: base 80% shifted by recency-weighted recent
        # model errors (last ADAPTIVE_ERROR_WINDOW errors, history from
        # Prometheus).
        safe_threshold, thresh_info = adaptive_threshold.get_adaptive_threshold()

        if is_predicting:
            cpu_to_scale = cpu_to_scale if 'cpu_to_scale' in locals() else predicted_load_final
            mem_to_scale = mem_to_scale if 'mem_to_scale' in locals() else predicted_memory_final
        else:
            cpu_to_scale = current_load
            mem_to_scale = current_memory

        # HPA-style tolerance deadband
        cpu_ratio = cpu_to_scale / safe_threshold
        if abs(cpu_ratio - 1.0) <= config.TOLERANCE:
            cpu_ratio = 1.0
        raw_desired = int(np.ceil(current_replicas * cpu_ratio))

        if config.NUM_TARGETS > 1:
            mem_ratio = mem_to_scale / safe_threshold
            if abs(mem_ratio - 1.0) <= config.TOLERANCE:
                mem_ratio = 1.0
            raw_desired = max(raw_desired, int(np.ceil(current_replicas * mem_ratio)))

        scaling_limited = raw_desired >= config.MAX_REPLICAS

        raw_desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, raw_desired))

        # HPA-style scale-up rate limit
        if raw_desired > current_replicas:
            raw_desired = min(raw_desired, _scale_up_ceiling(current_replicas))

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

        state["last_target"] = int(final_rec)
        state["last_load"] = current_load
        state["last_mem"] = current_memory
        state["last_replicas"] = current_replicas

        utils.save_state(state)

        t_end_eval = time.time()
        total_inference_time = metric_duration + (t_end_eval - t_start_eval)

        # Log metrics including the adaptive threshold.
        # These rows feed the metrics-exporter -> Prometheus, which is the
        # persistent source of truth for future adaptive-threshold errors.
        utils.log_metrics(
            utils.get_tehran_time(),
            current_load,
            current_memory,
            predicted_load_final,
            predicted_memory_final,
            safe_threshold,
            thresh_info.get("bias", 0.0),
            total_inference_time,
            current_replicas,
        )

        logs = (
            f"Mode: {mode}, Load: {cpu_to_scale:.2f}, Mem: {mem_to_scale:.2f}, "
            f"PredLoad: {predicted_load_final:.2f}, PredMem: {predicted_memory_final:.2f}, "
            f"Thresh: {safe_threshold:.2f} (base {config.BASE_THRESHOLD:.2f} "
            f"+/-{config.ADAPTIVE_THRESHOLD_RANGE:.2f}, bias {thresh_info.get('bias', 0.0):+.3f} "
            f"n={thresh_info.get('n_errors', 0)}/{thresh_info.get('n_window', 0)} "
            f"{thresh_info.get('source', '?')})"
        )
        output = {
            "targetReplicas": int(final_rec),
            "logs": logs,
            "conditions": {
                "AbleToScale": True,
                "ScalingActive": True,
                "ScalingLimited": bool(scaling_limited),
            },
        }
        sys.stdout.write(json.dumps(output))

    except Exception as e:
        utils.log_to_file(f"CRITICAL EXCEPTION: {str(e)}\n{traceback.format_exc()}")
        print(
            json.dumps(
                {
                    "targetReplicas": _last_target(),
                    "logs": f"Error: {str(e)}",
                    "conditions": _fail_conditions(),
                }
            )
        )


if __name__ == "__main__":
    main()

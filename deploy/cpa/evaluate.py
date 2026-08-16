import sys
import json
import time
import torch
import numpy as np
import traceback
import config
import utils
import model_builder
from conformal_state import ConformalManager


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
        rep += max(
            int(np.ceil(rep * config.SCALE_UP_MAX_PERCENT / 100.0)),
            config.SCALE_UP_MAX_PODS,
        )
    return rep


def _load_conformal_state():
    """Load conformal state from STATE_FILE."""
    try:
        state = utils.load_state()
        conformal_data = state.get("conformal")
        if conformal_data:
            return ConformalManager.from_dict(conformal_data)
    except Exception:
        pass
    # Return fresh state if loading fails
    return ConformalManager(
        num_targets=config.NUM_TARGETS,
        window_size=config.CONFORMAL_WINDOW,
        target_alpha=config.CONFORMAL_TARGET_ALPHA,
        eta=config.CONFORMAL_ETA,
        alpha_min=config.CONFORMAL_ALPHA_MIN,
        alpha_max=config.CONFORMAL_ALPHA_MAX,
    )


def _save_conformal_state(conformal_mgr: ConformalManager) -> None:
    """Save conformal state to STATE_FILE."""
    try:
        state = utils.load_state()
        state["conformal"] = conformal_mgr.to_dict()
        utils.save_state(state)
    except Exception as e:
        utils.log_to_file(f"Warning: Failed to save conformal state: {e}")


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
        rec_history = state["history"]

        now = time.time()
        mode = "Reactive"
        predicted_load_final = 0.0
        predicted_memory_final = 0.0

        # Load conformal state
        conformal_mgr = _load_conformal_state()

        # Pending queue for horizon-offset feedback
        pending_key = "conformal_pending"
        if pending_key not in state:
            state[pending_key] = []
        pending = state[pending_key]

        # Mature pending predictions: those where horizon time has passed
        matured = [p for p in pending if p["ts"] + config.HORIZON * 60 <= now]
        if matured:
            for p in matured:
                y_cpu = p.get("y_cpu")
                y_mem = p.get("y_mem")
                if y_cpu is not None:
                    conformal_mgr.states["cpu"].update(
                        float(y_cpu), p["q10"][0], p["q95"][0]
                    )
                if config.NUM_TARGETS > 1 and y_mem is not None:
                    conformal_mgr.states["memory"].update(
                        float(y_mem), p["q10"][1], p["q95"][1]
                    )
            pending = [p for p in pending if p["ts"] + config.HORIZON * 60 > now]
            state[pending_key] = pending

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
                # Quantile ensemble output: (1, H, T, 3) -> q10, q50, q95
                preds_tensor = torch.round(preds_tensor * 100) / 100
                if preds_tensor.dim() == 4:
                    q10 = preds_tensor[0, -1, :, 0]  # (T,)
                    q50 = preds_tensor[0, -1, :, 1]
                    q95 = preds_tensor[0, -1, :, 2]
                else:
                    # Fallback for non-quantile models
                    q50 = preds_tensor[0, -1]
                    q10 = q50
                    q95 = q50

                if config.NUM_TARGETS > 1:
                    predicted_load_final = float(q50[0].item())
                    predicted_memory_final = float(q50[1].item())
                else:
                    predicted_load_final = float(q50.item())
                    predicted_memory_final = 0.0

            # Get conformal intervals
            L, U = conformal_mgr.get_interval(q10, q95)
            if config.NUM_TARGETS > 1:
                lower_cpu, lower_mem = float(L[0]), float(L[1])
                upper_cpu, upper_mem = float(U[0]), float(U[1])
            else:
                lower_cpu, upper_cpu = float(L), float(U)
                lower_mem, upper_mem = 0.0, 0.0

            # Scale on UPPER BOUND (conservative for spike protection)
            cpu_to_scale = upper_cpu
            mem_to_scale = upper_mem

            # Store prediction in pending queue for horizon-offset feedback
            pending.append({
                "ts": time.time(),
                "q10": q10.tolist(),
                "q50": q50.tolist(),
                "q95": q95.tolist(),
            })
            # Keep only recent pending (horizon + buffer)
            max_pending = config.HORIZON + 2
            if len(pending) > max_pending:
                pending = pending[-max_pending:]
            state[pending_key] = pending

            mode = "Predictive"
        elif use_prediction:
            mode = "Predictive (Waiting for data)"

        is_predicting = mode.startswith("Predictive") and predicted_load_final > 0
        safe_threshold = config.BASE_THRESHOLD

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

        # Save conformal state
        _save_conformal_state(conformal_mgr)

        utils.save_state(state)

        t_end_eval = time.time()
        total_inference_time = metric_duration + (t_end_eval - t_start_eval)

        # Log metrics including conformal bounds
        utils.log_metrics(
            utils.get_tehran_time(),
            current_load,
            current_memory,
            predicted_load_final,
            predicted_memory_final,
            0.0,  # delta_load (no AR)
            0.0,  # delta_mem (no AR)
            total_inference_time,
            current_replicas,
        )

        # Include conformal bounds in logs
        alphas = conformal_mgr.get_alphas()
        alpha_u_cpu, alpha_l_cpu = alphas.get("cpu", (0.05, 0.05))
        alpha_u_mem, alpha_l_mem = alphas.get("memory", (0.05, 0.05))

        logs = (
            f"Mode: {mode}, Load: {cpu_to_scale:.2f}, Mem: {mem_to_scale:.2f}, "
            f"PredLoad: {predicted_load_final:.2f}, PredMem: {predicted_memory_final:.2f}, "
            f"Upper: {upper_cpu:.2f}, Lower: {lower_cpu:.2f}, "
            f"AlphaU: {alpha_u_cpu:.3f}, AlphaL: {alpha_l_cpu:.3f}"
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
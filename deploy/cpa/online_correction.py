import numpy as np
import config


def empty_state():
    p = config.AR_ORDER
    return {
        "pending": [],
        "res_cpu": [],
        "res_mem": [],
        "innov_cpu": [],
        "innov_mem": [],
        "w_cpu": [0.0] * p,
        "w_mem": [0.0] * p,
        "P_cpu": [
            [config.RLS_P0 if i == j else 0.0 for j in range(p)] for i in range(p)
        ],
        "P_mem": [
            [config.RLS_P0 if i == j else 0.0 for j in range(p)] for i in range(p)
        ],
    }


def _append(buf, value):
    buf.append(float(value))
    if len(buf) > config.RESIDUAL_WINDOW:
        del buf[: len(buf) - config.RESIDUAL_WINDOW]


def _mature_prediction(p, now):
    return p["time"] + config.HORIZON * 60 <= now


def finalize(state, now, cpu_actual, mem_actual):
    matured = [p for p in state["pending"] if _mature_prediction(p, now)]
    if not matured:
        return
    for p in matured:
        _finalize_one(state, "cpu", cpu_actual - p["cpu"])
        if config.NUM_TARGETS > 1:
            _finalize_one(state, "mem", mem_actual - p["mem"])
    state["pending"] = [p for p in state["pending"] if not _mature_prediction(p, now)]


def _finalize_one(state, target, e):
    p = config.AR_ORDER
    res_buf = state["res_" + target]
    if p > 0 and len(res_buf) >= p:
        x = np.asarray(list(reversed(res_buf[-p:])), dtype=np.float64)
        w = np.asarray(state["w_" + target], dtype=np.float64).reshape(-1, 1)
        P = np.asarray(state["P_" + target], dtype=np.float64)
        lam = config.FORGETTING_FACTOR
        r = float(e) - float(w.reshape(-1) @ x)
        _append(state["innov_" + target], r)
        Px = P @ x.reshape(-1, 1)
        denom = lam + (x.reshape(1, -1) @ Px).item()
        K = Px / denom
        w = w + K * r
        P = (P - K @ Px.T) / lam
        P = (P + P.T) / 2.0
        P = P + 1e-8 * np.eye(p)
        state["w_" + target] = w.flatten().tolist()
        state["P_" + target] = P.tolist()
    _append(res_buf, e)


def record_prediction(state, now, base_cpu, base_mem):
    state["pending"].append(
        {"time": float(now), "cpu": float(base_cpu), "mem": float(base_mem)}
    )
    max_pending = max(2, config.HORIZON + 1)
    if len(state["pending"]) > max_pending:
        del state["pending"][:-max_pending]


def compute_delta(state):
    delta_cpu = _target_delta(state, "cpu")
    delta_mem = _target_delta(state, "mem") if config.NUM_TARGETS > 1 else 0.0
    return delta_cpu, delta_mem


def _target_delta(state, target):
    p = config.AR_ORDER
    res_buf = state["res_" + target]
    ar = 0.0
    if p > 0 and len(res_buf) >= p:
        x = np.asarray(list(reversed(res_buf[-p:])), dtype=np.float64)
        w = np.asarray(state["w_" + target], dtype=np.float64)
        ar = float(w @ x)
    innov = state["innov_" + target]
    q = 0.0
    if innov:
        q = float(np.percentile(innov, config.QUANTILE_ALPHA * 100.0))
    return max(0.0, ar + q)

import json
import time
import numpy as np
import config
import utils

if config.PREPROCESS_APPROACH == "swt":
    from preprocessing.swt.config import CFG as SWT_CFG
    from preprocessing.swt.decomposition import decompose_window


def _round2(value):
    return round(float(value), 2)


def smooth_window(window_data, window_size=5):
    if not window_data or len(window_data) < window_size:
        return window_data

    arr = np.array(window_data, dtype=float)
    smoothed = np.zeros_like(arr)
    kernel = np.ones(window_size) / window_size

    for j in range(arr.shape[1]):
        col = arr[:, j]
        pad_size = window_size // 2
        padded = np.pad(col, (pad_size, pad_size), mode="edge")
        smoothed_col = np.convolve(padded, kernel, mode="valid")
        smoothed[:, j] = smoothed_col[: len(col)]

    return smoothed.tolist()


def fetch_metric_buckets(query, start_time, end_time, grid_timestamps):
    params = {"query": query, "start": start_time, "end": end_time, "step": "60s"}
    results = utils.query_prometheus(query, is_range=True, params=params)

    buckets = {i: [] for i in range(config.WINDOW_SIZE)}

    for res in results:
        for val_pair in res.get("values", []):
            ts, val = int(val_pair[0]), float(val_pair[1])
            idx = next(
                (i for i, g in enumerate(grid_timestamps) if abs(int(ts) - g) < 30),
                None,
            )
            if idx is not None:
                buckets[idx].append(float(val))
    return buckets


def get_aggregated_window():
    end_time = int(time.time())
    start_time = end_time - (config.WINDOW_SIZE * 60)
    grid_timestamps = [start_time + (i * 60) for i in range(config.WINDOW_SIZE)]

    pod_selector = f"pod=~'{config.DEPLOYMENT}-[a-z0-9]+-[a-z0-9]+'"
    ready_filter = (
        f"kube_pod_container_status_ready{{namespace='{config.NAMESPACE}', "
        f"container='server'}} == 1"
    )

    cpu_query = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', {pod_selector}, container='server'}}[1m]) and on (pod) ({ready_filter})) by (pod) / "
        f"sum(kube_pod_container_resource_requests{{namespace='{config.NAMESPACE}', {pod_selector}, container='server', resource='cpu'}} and on (pod) ({ready_filter})) by (pod)"
    )
    cpu_buckets = fetch_metric_buckets(cpu_query, start_time, end_time, grid_timestamps)

    mem_buckets = None
    if "mem" in config.FEATURE_SET:
        mem_query = (
            f"sum(container_memory_working_set_bytes{{namespace='{config.NAMESPACE}', {pod_selector}, container='server'}} and on (pod) ({ready_filter})) by (pod) / "
            f"sum(kube_pod_container_resource_requests{{namespace='{config.NAMESPACE}', {pod_selector}, container='server', resource='memory'}} and on (pod) ({ready_filter})) by (pod)"
        )
        mem_buckets = fetch_metric_buckets(
            mem_query, start_time, end_time, grid_timestamps
        )

    final_window = []
    use_prediction = True
    prev_cpu = None

    for i in range(config.WINDOW_SIZE):
        c_vals = cpu_buckets[i]

        has_cpu = bool(c_vals)
        has_mem = ("mem" not in config.FEATURE_SET) or bool(
            mem_buckets and mem_buckets[i]
        )

        if not (has_cpu and has_mem):
            use_prediction = False
            final_window.append([0.0] * config.RAW_INPUT_SIZE)
            prev_cpu = 0.0
        else:
            avg_cpu = sum(c_vals) / len(c_vals)
            avg_cpu = _round2(avg_cpu)
            cpu_diff = avg_cpu - prev_cpu if prev_cpu is not None else 0.0
            cpu_diff = _round2(cpu_diff)
            prev_cpu = avg_cpu

            row = [avg_cpu]

            if "mem" in config.FEATURE_SET:
                avg_mem = sum(mem_buckets[i]) / len(mem_buckets[i])
                avg_mem = _round2(avg_mem)
                row.append(avg_mem)

            if config.FEATURE_SET.endswith("_diff"):
                row.append(cpu_diff)

            final_window.append(row)

    if use_prediction:
        if config.PREPROCESS_APPROACH == "swt":
            final_window = apply_swt(final_window)
        else:
            final_window = smooth_window(final_window, window_size=5)
            final_window = [
                [_round2(v) for v in row] for row in final_window
            ]

    return final_window, use_prediction


def apply_swt(window_data):
    arr = np.asarray(window_data, dtype=np.float32)
    channels = []
    for col in range(arr.shape[1]):
        signal = arr[:, col]
        dec = decompose_window(signal, SWT_CFG)
        if dec is None:
            dec = np.zeros((SWT_CFG.SWT_LEVEL + 1, len(signal)), dtype=np.float32)
            dec[0] = signal
        channels.append(dec.T)
    return np.concatenate(channels, axis=1).tolist()


def main():
    t_start = time.time()
    history, use_prediction = get_aggregated_window()
    pod_selector = f"pod=~'{config.DEPLOYMENT}-[a-z0-9]+-[a-z0-9]+'"

    q_replicas = f"count(kube_pod_container_status_ready{{namespace='{config.NAMESPACE}', {pod_selector}, container='server'}} == 1)"
    res_rep = utils.query_prometheus(q_replicas)
    if res_rep:
        current_replicas = int(res_rep[0]["value"][1])
    else:
        state = utils.load_state()
        current_replicas = int(state.get("last_replicas", 1))

    q_load = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', {pod_selector}, container='server'}}[1m]) and on (pod) ({ready_filter})) / "
        f"sum(kube_pod_container_resource_requests{{namespace='{config.NAMESPACE}', {pod_selector}, container='server', resource='cpu'}} and on (pod) ({ready_filter}))"
    )
    res_load = utils.query_prometheus(q_load)

    if res_load:
        current_load = float(res_load[0]["value"][1])
        current_load = _round2(current_load)
    else:
        state = utils.load_state()
        last_load = state.get("last_load")
        if last_load is not None:
            current_load = _round2(float(last_load))
        else:
            last_point = history[-1] if history else 0.0
            current_load = last_point[0] if isinstance(last_point, list) else last_point

    q_mem = (
        f"sum(container_memory_working_set_bytes{{namespace='{config.NAMESPACE}', {pod_selector}, container='server'}} and on (pod) ({ready_filter})) / "
        f"sum(kube_pod_container_resource_requests{{namespace='{config.NAMESPACE}', {pod_selector}, container='server', resource='memory'}} and on (pod) ({ready_filter}))"
    )
    res_mem = utils.query_prometheus(q_mem)
    if res_mem:
        current_mem = float(res_mem[0]["value"][1])
    else:
        state = utils.load_state()
        last_mem = state.get("last_mem")
        current_mem = _round2(float(last_mem)) if last_mem is not None else 0.0
    current_mem = _round2(current_mem)

    t_end = time.time()

    print(
        json.dumps(
            {
                "metrics": history,
                "use_prediction": use_prediction,
                "current_load": current_load,
                "current_memory": current_mem,
                "current_replicas": current_replicas,
                "duration_seconds": t_end - t_start,
            }
        )
    )


if __name__ == "__main__":
    main()

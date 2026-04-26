import json
import time
import numpy as np
import config
import utils


def normalize_window(window_data):
    if not window_data or not isinstance(window_data[0], list):
        if isinstance(window_data, list) and len(window_data) > 0:
            vals = np.array(window_data, dtype=float)
            v_min, v_max = np.min(vals), np.max(vals)
            norm = (
                (vals - v_min) / (v_max - v_min)
                if v_max > v_min
                else np.zeros_like(vals)
            )
            return norm.tolist()
        return window_data

    arr = np.array(window_data, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        v_min, v_max = np.min(col), np.max(col)
        arr[:, j] = (col - v_min) / (v_max - v_min) if v_max > v_min else 0.0

    return arr.tolist()


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

    cpu_query = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', container!='POD'}}[1m])) by (pod) / "
        f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='cpu'}}) by (pod)"
    )
    cpu_buckets = fetch_metric_buckets(cpu_query, start_time, end_time, grid_timestamps)

    mem_buckets = None
    if config.FEATURE_SET in ["cpu_mem", "cpu_mem_traffic"]:
        mem_query = (
            f"sum(container_memory_working_set_bytes{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', container!='POD'}}) by (pod) / "
            f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='memory'}}) by (pod)"
        )
        mem_buckets = fetch_metric_buckets(
            mem_query, start_time, end_time, grid_timestamps
        )

    mcr_buckets = None
    if config.FEATURE_SET == "cpu_mem_traffic":
        mcr_query = (
            f"sum(rate(istio_requests_total{{destination_workload='{config.DEPLOYMENT}', "
            f"destination_workload_namespace='{config.NAMESPACE}', reporter='destination'}}[1m])) by (pod)"
        )
        mcr_buckets = fetch_metric_buckets(
            mcr_query, start_time, end_time, grid_timestamps
        )

    final_window = []
    use_prediction = True

    for i in range(config.WINDOW_SIZE):
        c_vals = cpu_buckets[i]

        has_cpu = bool(c_vals)
        has_mem = config.FEATURE_SET not in ["cpu_mem", "cpu_mem_traffic"] or bool(
            mem_buckets and mem_buckets[i]
        )
        has_mcr = config.FEATURE_SET != "cpu_mem_traffic" or bool(
            mcr_buckets and mcr_buckets[i]
        )

        if not (has_cpu and has_mem and has_mcr):
            use_prediction = False
            if config.FEATURE_SET == "cpu_mem_traffic":
                final_window.append([0.0, 0.0, 0.0])
            elif config.FEATURE_SET == "cpu_mem":
                final_window.append([0.0, 0.0])
            else:
                final_window.append(0.0)
        else:
            avg_cpu = sum(c_vals) / len(c_vals)
            if config.FEATURE_SET == "cpu_mem_traffic":
                avg_mem = sum(mem_buckets[i]) / len(mem_buckets[i])
                avg_mcr = sum(mcr_buckets[i]) / len(mcr_buckets[i])
                final_window.append([avg_cpu, avg_mem, avg_mcr])
            elif config.FEATURE_SET == "cpu_mem":
                avg_mem = sum(mem_buckets[i]) / len(mem_buckets[i])
                final_window.append([avg_cpu, avg_mem])
            else:
                final_window.append(avg_cpu)

    if use_prediction:
        final_window = normalize_window(final_window)

    return final_window, use_prediction


def main():
    t_start = time.time()
    history, use_prediction = get_aggregated_window()

    q_replicas = f"count(kube_pod_status_phase{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', phase='Running'}})"
    res_rep = utils.query_prometheus(q_replicas)
    current_replicas = int(res_rep[0]["value"][1]) if res_rep else 1

    q_load = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', container!='POD'}}[1m])) / "
        f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='cpu'}})"
    )
    res_load = utils.query_prometheus(q_load)

    if res_load:
        current_load = float(res_load[0]["value"][1])
    else:
        last_point = history[-1] if history else 0.0
        current_load = last_point[0] if isinstance(last_point, list) else last_point

    t_end = time.time()

    print(
        json.dumps(
            {
                "metrics": history,
                "use_prediction": use_prediction,
                "current_load": current_load,
                "current_replicas": current_replicas,
                "duration_seconds": t_end - t_start,
            }
        )
    )


if __name__ == "__main__":
    main()

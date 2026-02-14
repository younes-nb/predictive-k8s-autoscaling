import json
import time
import config
import utils


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
    if config.FEATURE_SET == "cpu_mem":
        mem_query = (
            f"sum(container_memory_working_set_bytes{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', container!='POD'}}) by (pod) / "
            f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='memory'}}) by (pod)"
        )
        mem_buckets = fetch_metric_buckets(
            mem_query, start_time, end_time, grid_timestamps
        )

    final_window = []
    use_prediction = True

    for i in range(config.WINDOW_SIZE):
        c_vals = cpu_buckets[i]
        if not c_vals or (
            config.FEATURE_SET == "cpu_mem" and (not mem_buckets or not mem_buckets[i])
        ):
            use_prediction = False
            row = [0.0, 0.0] if config.FEATURE_SET == "cpu_mem" else 0.0
            final_window.append(row)
        else:
            avg_cpu = sum(c_vals) / len(c_vals)
            if config.FEATURE_SET == "cpu_mem":
                m_vals = mem_buckets[i]
                avg_mem = sum(m_vals) / len(m_vals)
                final_window.append([avg_cpu, avg_mem])
            else:
                final_window.append(avg_cpu)

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

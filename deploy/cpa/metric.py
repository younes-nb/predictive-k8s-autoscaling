import json
import time
import config
import utils


def get_aggregated_window():
    end_time = int(time.time())
    start_time = end_time - (config.WINDOW_SIZE * 60)
    grid_timestamps = [start_time + (i * 60) for i in range(config.WINDOW_SIZE)]
    buckets = {i: [] for i in range(config.WINDOW_SIZE)}

    query = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*'}}[1m])) by (pod) / "
        f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='cpu'}}) by (pod)"
    )

    params = {"query": query, "start": start_time, "end": end_time, "step": "60s"}
    results = utils.query_prometheus(query, is_range=True, params=params)

    for res in results:
        for ts, val in res.get("values", []):
            idx = next(
                (i for i, g in enumerate(grid_timestamps) if abs(int(ts) - g) < 30),
                None,
            )
            if idx is not None:
                buckets[idx].append(float(val))

    final_window = []
    use_prediction = True
    for i in range(config.WINDOW_SIZE):
        if not buckets[i]:
            use_prediction = False
            final_window.append(0.0)
        else:
            final_window.append(sum(buckets[i]) / len(buckets[i]))
    return final_window, use_prediction


def main():
    history, use_prediction = get_aggregated_window()

    q_replicas = f"count(kube_pod_status_phase{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', phase='Running'}})"
    res_rep = utils.query_prometheus(q_replicas)
    current_replicas = int(res_rep[0]["value"][1]) if res_rep else 1

    q_load = (
        f"sum(rate(container_cpu_usage_seconds_total{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*'}}[1m])) / "
        f"sum(kube_pod_container_resource_limits{{namespace='{config.NAMESPACE}', pod=~'{config.DEPLOYMENT}-.*', resource='cpu'}})"
    )
    res_load = utils.query_prometheus(q_load)
    current_load = (
        float(res_load[0]["value"][1])
        if res_load
        else (history[-1] if history else 0.0)
    )

    print(
        json.dumps(
            {
                "metrics": history,
                "use_prediction": use_prediction,
                "current_load": current_load,
                "current_replicas": current_replicas,
            }
        )
    )


if __name__ == "__main__":
    main()

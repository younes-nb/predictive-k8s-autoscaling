"""Generic, runtime-agnostic feature set.

Nothing here assumes JVM, Spring, Go or any framework: every feature comes from
cAdvisor, kube-state-metrics or the Istio sidecar, all of which exist for any
workload scheduled on this cluster.

Each ``MetricQuery`` is one range query that returns *all* services at once
(``sum by (container)`` / ``sum by (destination_workload)``), so a control cycle
costs one query per family rather than one per service.
"""

from dataclasses import dataclass


@dataclass
class MetricQuery:
    name: str
    group: str
    promql: str          # "{ns}" is substituted with the namespace
    kind: str            # "rate" (already a rate), "gauge", "hist"
    required: bool = True
    label_key: str = "container"   # which label identifies the service


# --------------------------------------------------------------------------
# Queries. Order here is NOT the model input order; FEATURE_ORDER below is.
# --------------------------------------------------------------------------
def queries(ns):
    """One range query per metric family.

    Every query preserves the labels needed to attribute a series to a
    Deployment: ``namespace``, ``pod`` and, where the metric is per-container,
    ``container``. Aggregating by container name alone was wrong -- many
    Deployments use the same container name (e.g. ``app``), so a container-level
    ``sum by (container)`` merges unrelated services into one series. Pods are
    mapped to their owning Deployment after the query (``telemetry.mimir``).

    CPU and memory are aggregated to the *pod* boundary, which matches the scope
    the HPA uses (a Deployment's pods), so numerator (usage) and denominator
    (request) refer to the same thing.
    """
    sel = f'namespace="{ns}",container!="",container!="istio-proxy"'
    k8s = f'kube_deployment_spec_replicas{{namespace="{ns}"}}'
    ist = (f'istio_requests_total{{reporter="destination",'
           f'destination_workload_namespace="{ns}"}}')
    hist = (f'istio_request_duration_milliseconds_bucket{{reporter="destination",'
            f'destination_workload_namespace="{ns}"}}')
    return [
        # ---- workload / performance (Istio, destination reporter) --------
        MetricQuery("req_rate", "workload",
                    f'sum by (destination_workload) (rate({ist}[1m]))', "rate",
                    label_key="destination_workload"),
        MetricQuery("conn_fail", "workload",
                    f'sum by (destination_workload) (rate(istio_requests_total{{reporter="destination",'
                    f'destination_workload_namespace="{ns}",response_code="0"}}[1m]))', "rate",
                    required=False, label_key="destination_workload"),
        MetricQuery("err_rate", "workload",
                    f'sum by (destination_workload) (rate(istio_requests_total{{reporter="destination",'
                    f'destination_workload_namespace="{ns}",response_code=~"5.."}}[1m]))', "rate",
                    label_key="destination_workload"),
        MetricQuery("lat_p50", "workload",
                    f'histogram_quantile(0.50, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist",
                    label_key="destination_workload"),
        MetricQuery("lat_p90", "workload",
                    f'histogram_quantile(0.90, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist",
                    label_key="destination_workload"),
        MetricQuery("lat_p99", "workload",
                    f'histogram_quantile(0.99, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist",
                    label_key="destination_workload"),
        # ---- CPU (pod scope; HPA semantics) ------------------------------
        MetricQuery("cpu_cores", "cpu",
                    f'sum by (pod) (rate(container_cpu_usage_seconds_total'
                    f'{{{sel}}}[1m]))', "rate", label_key="pod"),
        MetricQuery("cpu_throttled", "cpu",
                    f'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total'
                    f'{{{sel}}}[1m]))', "rate", label_key="pod"),
        MetricQuery("cpu_periods", "cpu",
                    f'sum by (pod) (rate(container_cpu_cfs_periods_total'
                    f'{{{sel}}}[1m]))', "rate", label_key="pod"),
        # ---- memory ------------------------------------------------------
        MetricQuery("mem_working_set", "memory",
                    f'sum by (pod) (container_memory_working_set_bytes{{{sel}}})',
                    "gauge", label_key="pod"),
        MetricQuery("cpu_request", "cpu",
                    f'sum by (pod) (kube_pod_container_resource_requests'
                    f'{{namespace="{ns}",resource="cpu",container!="",container!="istio-proxy"}})',
                    "gauge", label_key="pod"),
        MetricQuery("mem_request", "memory",
                    f'sum by (pod) (kube_pod_container_resource_requests'
                    f'{{namespace="{ns}",resource="memory",container!="",container!="istio-proxy"}})',
                    "gauge", label_key="pod"),
        MetricQuery("mem_limit", "memory",
                    f'sum by (pod) (kube_pod_container_resource_limits'
                    f'{{namespace="{ns}",resource="memory",container!="",container!="istio-proxy"}})',
                    "gauge", label_key="pod"),
        MetricQuery("mem_page_faults", "memory",
                    f'sum by (pod) (rate(container_memory_failures_total{{{sel}}}[1m]))',
                    "rate", required=False, label_key="pod"),
        MetricQuery("oom_events", "memory",
                    f'sum by (pod) (container_oom_events_total{{{sel}}})',
                    "gauge", required=False, label_key="pod"),
        # ---- network -----------------------------------------------------
        MetricQuery("net_rx", "network",
                    f'sum by (pod) (rate(container_network_receive_bytes_total'
                    f'{{namespace="{ns}"}}[1m]))',
                    "rate", required=False, label_key="pod"),
        MetricQuery("net_tx", "network",
                    f'sum by (pod) (rate(container_network_transmit_bytes_total'
                    f'{{namespace="{ns}"}}[1m]))',
                    "rate", required=False, label_key="pod"),
        # ---- storage / IO ------------------------------------------------
        MetricQuery("fs_read", "storage",
                    f'sum by (pod) (rate(container_fs_reads_bytes_total{{{sel}}}[1m]))',
                    "rate", required=False, label_key="pod"),
        MetricQuery("fs_write", "storage",
                    f'sum by (pod) (rate(container_fs_writes_bytes_total{{{sel}}}[1m]))',
                    "rate", required=False, label_key="pod"),
        MetricQuery("io_time", "storage",
                    f'sum by (pod) (rate(container_blkio_device_usage_total{{{sel}}}[1m]))',
                    "rate", required=False, label_key="pod"),
        # ---- kubernetes lifecycle ---------------------------------------
        MetricQuery("replicas", "k8s", k8s, "gauge", label_key="deployment"),
        MetricQuery("ready", "k8s",
                    f'kube_deployment_status_replicas_ready{{namespace="{ns}"}}',
                    "gauge", label_key="deployment"),
        MetricQuery("available", "k8s",
                    f'kube_deployment_status_replicas_available{{namespace="{ns}"}}',
                    "gauge", label_key="deployment"),
        MetricQuery("pod_start_time", "k8s",
                    f'max by (pod) (kube_pod_start_time{{namespace="{ns}"}})',
                    "gauge", required=False, label_key="pod"),
    ]


# --------------------------------------------------------------------------
# Model input order. Derived features (ratios) are appended after the raw ones.
# --------------------------------------------------------------------------
RAW_FEATURES = [
    "req_rate", "err_rate", "conn_fail", "lat_p50", "lat_p90", "lat_p99",
    "cpu_cores", "cpu_throttled", "cpu_periods",
    "mem_working_set", "mem_limit", "mem_page_faults", "oom_events",
    "net_rx", "net_tx",
    "fs_read", "fs_write", "io_time",
    "cpu_request", "mem_request", "replicas", "ready", "available", "pod_age_s",
]

#   *_pct  = a saturation ratio -- a pressure indicator
#   completion_ratio / unresolved_ratio = request lifecycle, not pressure
#   inflight / inflight_per_replica = concurrency PROXY from Little's law on the
#                                     P50 latency, not a measured in-flight count
DERIVED_FEATURES = [
    "success_rate",          # 1 - err/req  (5xx are completed-but-bad responses)
    "cpu_throttled_frac",    # throttled / periods  -> CPU pressure
    "mem_limit_pct",         # working_set / mem_limit
    "mem_req_pct",           # working_set / mem_request
    "mem_headroom",          # 1 - mem_limit_pct
    "cpu_per_request",       # cpu_cores / req_rate (efficiency, scale-free)
    "lat_ratio_p90_p50",     # queueing indicator
    "inflight",              # concurrency proxy: req_rate * lat_p50 / 1000
    "inflight_per_replica",  # concurrency proxy per replica
    "pending",               # replicas - ready
    "cpu_per_replica",       # cpu_cores / replicas
    "req_per_replica",       # req_rate / replicas
    "completion_ratio",      # completed responses / requests (5xx ARE complete)
    "unresolved_ratio",      # timeout/connection-failure share of requests
    "cpu_request_utilization",  # cpu_cores / cpu_request (HPA convention, uncapped)
]

FEATURE_ORDER = RAW_FEATURES + DERIVED_FEATURES
N_FEATURES = len(FEATURE_ORDER)

# Groups that can be silently absent (cAdvisor does not expose them for every
# pod). Their mask bits will simply be 0.
OPTIONAL_FEATURES = {"mem_page_faults", "oom_events", "net_rx", "net_tx",
                     "fs_read", "fs_write", "io_time", "pod_start_time"}

# Per-edge feature order for the directed invocation edges. Kept next to the
# node schema so a bundle can validate both against one source of truth.
EDGE_FEATURE_ORDER = ["rate", "err", "p50", "p90", "p99"]


def derive(f):
    """Compute derived features from a raw feature dict.

    Three families, deliberately kept apart:

    * **pressure** ratios (``cpu_request_utilization``, ``mem_limit_pct``,
      ``cpu_throttled_frac``) describe how close a resource is to its own
      ceiling. ``cpu_request_utilization`` is the HPA convention
      (usage / request) and is intentionally uncapped -- it can exceed 1.0.
    * **request lifecycle** ratios (``completion_ratio``, ``success_rate``,
      ``unresolved_ratio``) describe whether requests finished and how. A 503 is
      a *completed* but unsuccessful response, so it lowers success_rate and
      leaves completion_ratio alone; only timeouts and connection failures
      (response_code 0) lower completion_ratio. A non-zero unresolved ratio
      means some requests have not completed and the latency histogram cannot
      see them yet.
    * **concurrency proxies** (``inflight``, ``inflight_per_replica``). These
      come from Little's law on the P50 latency and are approximations, not
      measured in-flight counters.

    Any ratio whose inputs are missing returns None, so a missing denominator
    can never masquerade as a healthy zero.
    """
    def g(k):
        return f.get(k)

    def div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    out = {}

    req, err, cf = g("req_rate"), g("err_rate"), g("conn_fail")
    # success: 5xx is a completed-but-bad response; connection failures are not
    # responses at all. They are reported separately as the unresolved ratio.
    out["success_rate"] = None if (req is None or err is None) else (
        1.0 - (err / req) if req else 1.0)
    if req is None or req <= 0:
        out["completion_ratio"] = None
        out["unresolved_ratio"] = None
    else:
        # Completed responses = requests that produced a response. A 5xx did
        # complete, so it stays in the numerator; connection failures did not.
        out["completion_ratio"] = max(0.0, min(1.0, 1.0 - (cf or 0.0) / req))
        out["unresolved_ratio"] = max(0.0, min(1.0, (cf or 0.0) / req))

    out["mem_limit_pct"] = div(g("mem_working_set"), g("mem_limit"))
    # FRACTION in [0,1], not a percentage. The capacity estimator
    # consumes it directly; dividing by 100 here would understate
    # throttling 100-fold (78% read as 0.78%).
    out["cpu_throttled_frac"] = div(g("cpu_throttled"), g("cpu_periods"))
    out["cpu_per_request"] = div(g("cpu_cores"), req)
    out["lat_ratio_p90_p50"] = div(g("lat_p90"), g("lat_p50"))
    lat = g("lat_p50")
    # Concurrency PROXY (Little's law with the P50 latency, not the mean
    # residence time). It is not a measured in-flight request count.
    out["inflight"] = None if (req is None or lat is None) else req * lat / 1000.0
    rep, rdy = g("replicas"), g("ready")
    out["pending"] = None if (rep is None or rdy is None) else max(0.0, rep - rdy)
    out["cpu_per_replica"] = div(g("cpu_cores"), rep)
    out["req_per_replica"] = div(req, rep)
    out["inflight_per_replica"] = div(out.get("inflight"), rep)

    mlp = out.get("mem_limit_pct")
    out["cpu_request_utilization"] = div(g("cpu_cores"), g("cpu_request"))
    out["mem_req_pct"] = div(g("mem_working_set"), g("mem_request"))
    out["mem_headroom"] = None if mlp is None else max(0.0, 1.0 - mlp)
    return out

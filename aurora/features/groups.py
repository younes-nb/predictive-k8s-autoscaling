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
    sel = f'namespace="{ns}",container!="",container!="istio-proxy"'
    k8s = f'kube_deployment_spec_replicas{{namespace="{ns}"}}'
    ist = (f'istio_requests_total{{reporter="destination",'
           f'destination_workload_namespace="{ns}"}}')
    hist = (f'istio_request_duration_milliseconds_bucket{{reporter="destination",'
            f'destination_workload_namespace="{ns}"}}')
    return [
        # ---- workload / performance (Istio, destination reporter) --------
        MetricQuery("req_rate", "workload",
                    f'sum by (destination_workload) (rate({ist}[1m]))', "rate"),
        MetricQuery("err_rate", "workload",
                    f'sum by (destination_workload) (rate(istio_requests_total{{reporter="destination",'
                    f'destination_workload_namespace="{ns}",response_code=~"5.."}}[1m]))', "rate"),
        MetricQuery("lat_p50", "workload",
                    f'histogram_quantile(0.50, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist"),
        MetricQuery("lat_p90", "workload",
                    f'histogram_quantile(0.90, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist"),
        MetricQuery("lat_p99", "workload",
                    f'histogram_quantile(0.99, sum by (destination_workload, le) '
                    f'(rate({hist}[1m])))', "hist"),
        # ---- CPU ---------------------------------------------------------
        MetricQuery("cpu_cores", "cpu",
                    f'sum by (container) (rate(container_cpu_usage_seconds_total'
                    f'{{{sel}}}[1m]))', "rate"),
        MetricQuery("cpu_limit", "cpu",
                    f'sum by (container) (kube_pod_container_resource_limits'
                    f'{{namespace="{ns}",resource="cpu",container!=""}})', "gauge"),
        MetricQuery("cpu_throttled", "cpu",
                    f'sum by (container) (rate(container_cpu_cfs_throttled_periods_total'
                    f'{{{sel}}}[1m]))', "rate"),
        MetricQuery("cpu_periods", "cpu",
                    f'sum by (container) (rate(container_cpu_cfs_periods_total'
                    f'{{{sel}}}[1m]))', "rate"),
        # ---- memory ------------------------------------------------------
        MetricQuery("mem_working_set", "memory",
                    f'sum by (container) (container_memory_working_set_bytes{{{sel}}})',
                    "gauge"),
        MetricQuery("mem_limit", "memory",
                    f'sum by (container) (kube_pod_container_resource_limits'
                    f'{{namespace="{ns}",resource="memory",container!=""}})', "gauge"),
        MetricQuery("mem_page_faults", "memory",
                    f'sum by (container) (rate(container_memory_failures_total{{{sel}}}[1m]))',
                    "rate", required=False),
        MetricQuery("oom_events", "memory",
                    f'sum by (container) (container_oom_events_total{{{sel}}})',
                    "gauge", required=False),
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
                    f'sum by (container) (rate(container_fs_reads_bytes_total{{{sel}}}[1m]))',
                    "rate", required=False),
        MetricQuery("fs_write", "storage",
                    f'sum by (container) (rate(container_fs_writes_bytes_total{{{sel}}}[1m]))',
                    "rate", required=False),
        MetricQuery("io_time", "storage",
                    f'sum by (container) (rate(container_blkio_device_usage_total{{{sel}}}[1m]))',
                    "rate", required=False),
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
    "req_rate", "err_rate", "lat_p50", "lat_p90", "lat_p99",
    "cpu_cores", "cpu_limit", "cpu_throttled", "cpu_periods",
    "mem_working_set", "mem_limit", "mem_page_faults", "oom_events",
    "net_rx", "net_tx",
    "fs_read", "fs_write", "io_time",
    "replicas", "ready", "available", "pod_start_time",
]

DERIVED_FEATURES = [
    "success_rate",          # 1 - err/req
    "cpu_limit_pct",         # cpu_cores / cpu_limit
    "cpu_throttled_pct",     # throttled / periods
    "mem_limit_pct",         # working_set / mem_limit
    "cpu_per_request",       # cpu_cores / req_rate  (efficiency, scale-free)
    "lat_ratio_p90_p50",     # queueing indicator
    "inflight",              # Little's law: req_rate * lat_p50 / 1000
    "pending",               # replicas - ready
    "cpu_per_replica",       # cpu_cores / replicas
    "req_per_replica",       # req_rate / replicas
]

FEATURE_ORDER = RAW_FEATURES + DERIVED_FEATURES
N_FEATURES = len(FEATURE_ORDER)

# Groups that can be silently absent (cAdvisor does not expose them for every
# pod). Their mask bits will simply be 0.
OPTIONAL_FEATURES = {"mem_page_faults", "oom_events", "net_rx", "net_tx",
                     "fs_read", "fs_write", "io_time", "pod_start_time"}


def derive(f):
    """Compute derived features from a raw feature dict.

    Returns a new dict. A ratio is unavailable (None) unless both inputs are,
    so a missing denominator never becomes a fake zero.
    """
    def g(k):
        return f.get(k)

    def div(a, b):
        if a is None or b is None:
            return None
        if b == 0:
            return None
        return a / b

    out = {}
    req, err = g("req_rate"), g("err_rate")
    out["success_rate"] = None if (req is None or err is None) else (
        1.0 - (err / req) if req else 1.0)
    out["cpu_limit_pct"] = div(g("cpu_cores"), g("cpu_limit"))
    out["cpu_throttled_pct"] = div(g("cpu_throttled"), g("cpu_periods"))
    out["mem_limit_pct"] = div(g("mem_working_set"), g("mem_limit"))
    out["cpu_per_request"] = div(g("cpu_cores"), req)
    out["lat_ratio_p90_p50"] = div(g("lat_p90"), g("lat_p50"))
    lat = g("lat_p50")
    out["inflight"] = None if (req is None or lat is None) else req * lat / 1000.0
    rep, rdy = g("replicas"), g("ready")
    out["pending"] = None if (rep is None or rdy is None) else max(0.0, rep - rdy)
    out["cpu_per_replica"] = div(g("cpu_cores"), rep)
    out["req_per_replica"] = div(req, rep)
    return out
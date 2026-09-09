from typing import Dict, Any, List, Set, Optional

import numpy as np

FEATURES: Dict[str, Dict[str, str]] = {
    "cpu_utilization": {"table": "msresource", "column": "cpu_utilization"},
    "memory_utilization": {"table": "msresource", "column": "memory_utilization"},
    "replicas": {"table": "msresource", "column": "replicas"},
    # Infra-native columns from analytics/export_hpa.py (istio/envoy/
    # kube-state/node telemetry, no app logs). Served from the HPA-logs CSV
    # (--csv_path); the feature_set using them is csv/export-native.
    "desired_replicas": {"table": "msresource", "column": "desired_replicas"},
    "unavailable": {"table": "msresource", "column": "unavailable"},
    "restart_rate": {"table": "msresource", "column": "restart_rate"},
    "n_callers": {"table": "msrtmcre", "column": "n_callers"},
    "req_byte_rate": {"table": "msrtmcre", "column": "req_byte_rate"},
    "resp_byte_rate": {"table": "msrtmcre", "column": "resp_byte_rate"},
    "req_bytes_per_req": {"table": "msrtmcre", "column": "req_bytes_per_req"},
    "resp_bytes_per_req": {"table": "msrtmcre", "column": "resp_bytes_per_req"},
    "slow_frac": {"table": "msrtmcre", "column": "slow_frac"},
    "req_msg_rate": {"table": "msrtmcre", "column": "req_msg_rate"},
    "resp_msg_rate": {"table": "msrtmcre", "column": "resp_msg_rate"},
    "msgs_per_req": {"table": "msrtmcre", "column": "msgs_per_req"},
    "err_rate": {"table": "msrtmcre", "column": "err_rate"},
    "err_frac": {"table": "msrtmcre", "column": "err_frac"},
    "flag_rate": {"table": "msrtmcre", "column": "flag_rate"},
    "flag_frac": {"table": "msrtmcre", "column": "flag_frac"},
    "queue_in": {"table": "msrtmcre", "column": "queue_in"},
    "queue_for": {"table": "msrtmcre", "column": "queue_for"},
    "active_in": {"table": "msrtmcre", "column": "active_in"},
    "active_for": {"table": "msrtmcre", "column": "active_for"},
    "tod_sin": {"table": "time", "column": "tod_sin"},
    "tod_cos": {"table": "time", "column": "tod_cos"},
    "neigh_cpu_mean": {"table": "msrtmcre", "column": "neigh_cpu_mean"},
    "neigh_cpu_slope3": {"table": "msrtmcre", "column": "neigh_cpu_slope3"},
    "neigh_rps_z30_mean": {"table": "msrtmcre", "column": "neigh_rps_z30_mean"},
    "neigh_rps_slope5_mean": {"table": "msrtmcre", "column": "neigh_rps_slope5_mean"},
    "node_load_mean": {"table": "node", "column": "node_load_mean"},
    "node_load_max": {"table": "node", "column": "node_load_max"},
    "node_cpu_mean": {"table": "node", "column": "node_cpu_mean"},
    # Older export columns (analytics/export_hpa.py causal/graph features).
    "p99_latency": {"table": "msrtmcre", "column": "p99_latency"},
    "throttle_ratio": {"table": "msresource", "column": "throttle_ratio"},
    "net_rx": {"table": "msresource", "column": "net_rx"},
    "rps_total": {"table": "msrtmcre", "column": "rps_total"},
    "caller_rps_max": {"table": "msrtmcre", "column": "caller_rps_max"},
    "upstream_rps_sum": {"table": "msrtmcre", "column": "upstream_rps_sum"},
    "root_rps": {"table": "msrtmcre", "column": "root_rps"},
    "rps_z30": {"table": "msrtmcre", "column": "rps_z30"},
    "rps_slope5": {"table": "msrtmcre", "column": "rps_slope5"},
    "ewma_gap": {"table": "msrtmcre", "column": "ewma_gap"},
    "cpu_slope3": {"table": "msresource", "column": "cpu_slope3"},
    "mem_delta5": {"table": "msresource", "column": "mem_delta5"},
    "node_cpu_utilization": {"table": "node", "column": "cpu_utilization"},
    "node_memory_utilization": {"table": "node", "column": "memory_utilization"},
    "providerrpc_rt": {"table": "msrtmcre", "column": "providerrpc_rt"},
    "providerrpc_mcr": {"table": "msrtmcre", "column": "providerrpc_mcr"},
    "consumerrpc_mcr": {"table": "msrtmcre", "column": "consumerrpc_mcr"},
    "http_rt": {"table": "msrtmcre", "column": "http_rt"},
    "http_mcr": {"table": "msrtmcre", "column": "http_mcr"},
    "providermq_rt": {"table": "msrtmcre", "column": "providermq_rt"},
    "providermq_mcr": {"table": "msrtmcre", "column": "providermq_mcr"},
    "consumermq_mcr": {"table": "msrtmcre", "column": "consumermq_mcr"},
    "writemc_mcr": {"table": "msrtmcre", "column": "writemc_mcr"},
    "readmc_mcr": {"table": "msrtmcre", "column": "readmc_mcr"},
    "writedb_mcr": {"table": "msrtmcre", "column": "writedb_mcr"},
    "readdb_mcr": {"table": "msrtmcre", "column": "readdb_mcr"},
    # Calendar features derived from the (relative) trace timestamp, not from
    # any parquet column. minute/hour wrap on wall clock (t=0 is 00:00);
    # day is the day-index since trace start (timestamps are relative, so
    # no calendar date exists). Values are exact integers (no normalization).
    "minute": {"table": "time", "column": "minute", "derived": True},
    "hour": {"table": "time", "column": "hour", "derived": True},
    "day": {"table": "time", "column": "day", "derived": True},
    "um": {"table": "mscallgraph", "column": "um"},
    "dm": {"table": "mscallgraph", "column": "dm"},
    "rpctype": {"table": "mscallgraph", "column": "rpctype"},
    "rt": {"table": "mscallgraph", "column": "rt"},
    "traceid": {"table": "mscallgraph", "column": "traceid"},
    "rpc_id": {"table": "mscallgraph", "column": "rpc_id"},
    "service": {"table": "mscallgraph", "column": "service"},
    "interface": {"table": "mscallgraph", "column": "interface"},
    "uminstanceid": {"table": "mscallgraph", "column": "uminstanceid"},
    "dminstanceid": {"table": "mscallgraph", "column": "dminstanceid"},
}


FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "features": ["cpu_utilization"],
        "target": "cpu_utilization",
        "base_table": "msresource",
    },
    "cpu_mem": {
        "features": ["cpu_utilization", "memory_utilization"],
        "target": "cpu_utilization",
        "base_table": "msresource",
    },
    "node_cpu_mem": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "node_cpu_utilization",
            "node_memory_utilization",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "join_keys": {"msresource": ["nodeid"], "node": ["nodeid"]},
    },
    "cpu_mem_mcr": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "providerrpc_mcr",
            "consumerrpc_mcr",
            "providermq_mcr",
            "consumermq_mcr",
            "http_mcr",
            "writemc_mcr",
            "readmc_mcr",
            "writedb_mcr",
            "readdb_mcr",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "join_keys": {
            "msresource": ["msname"],
            "msrtmcre": ["msname"],
        },
    },
    "threshold_analysis": {
        "features": [
            "cpu_utilization",
            "providerrpc_rt",
            "providerrpc_mcr",
            "http_rt",
            "http_mcr",
            "providermq_rt",
            "providermq_mcr",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "join_keys": {
            "msresource": ["msname", "msinstanceid"],
            "msrtmcre": ["msname", "msinstanceid"],
        },
    },
    "cpu_mem_both": {
        "features": ["cpu_utilization", "memory_utilization"],
        "targets": ["cpu_utilization", "memory_utilization"],
        "base_table": "msresource",
    },
    "cpu_mem_http_rpc": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "http_mcr",
            "providerrpc_mcr",
        ],
        "targets": ["cpu_utilization", "memory_utilization"],
        "base_table": "msresource",
        "join_keys": {
            "msresource": ["msname"],
            "msrtmcre": ["msname"],
        },
    },
    "mcr_http": {
        "features": ["http_mcr"],
        "target": "http_mcr",
        "base_table": "msrtmcre",
    },
    "cpu_mem_rpc": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "providerrpc_mcr",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "join_keys": {
            "msresource": ["msname"],
            "msrtmcre": ["msname"],
        },
    },
    "cpu_mem_http": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "http_mcr",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "join_keys": {
            "msresource": ["msname"],
            "msrtmcre": ["msname"],
        },
    },
    "http_time": {
        "features": [
            "http_mcr",
            "minute",
            "hour",
            "day",
        ],
        "target": "cpu_utilization",
        "base_table": "msrtmcre",
        "join_keys": {
            "msresource": ["msname"],
            "msrtmcre": ["msname"],
        },
    },
    "cpu_mem_http_rpc_replicas": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "http_mcr",
            "providerrpc_mcr",
            "replicas",
        ],
        "targets": ["cpu_utilization", "memory_utilization"],
        "base_table": "msresource",
    },
    "callgraph": {
        "features": [
            "um",
            "dm",
            "rpctype",
            "rt",
            "traceid",
            "rpc_id",
            "service",
            "interface",
            "uminstanceid",
            "dminstanceid",
        ],
        "target": "um",
        "base_table": "mscallgraph",
    },
    # Full infra-native MS feature set for next-minute cpu_utilization:
    # mesh/call-graph load + per-request cost proxies + envoy queues +
    # control-plane + node pressure. Served from the HPA-logs CSV
    # (--csv_path); csv/export-native (no parquet tables back the
    # istio/envoy/node columns).
    "cpu_ms_infra": {
        "features": [
            "cpu_utilization",
            "memory_utilization",
            "replicas",
            "desired_replicas",
            "unavailable",
            "restart_rate",
            "http_mcr",
            "providerrpc_mcr",
            "rps_total",
            "caller_rps_max",
            "n_callers",
            "upstream_rps_sum",
            "root_rps",
            "req_byte_rate",
            "resp_byte_rate",
            "req_bytes_per_req",
            "resp_bytes_per_req",
            "slow_frac",
            "req_msg_rate",
            "resp_msg_rate",
            "msgs_per_req",
            "err_rate",
            "err_frac",
            "flag_rate",
            "flag_frac",
            "queue_in",
            "queue_for",
            "active_in",
            "active_for",
            "p99_latency",
            "throttle_ratio",
            "net_rx",
            "rps_z30",
            "rps_slope5",
            "ewma_gap",
            "cpu_slope3",
            "mem_delta5",
            "tod_sin",
            "tod_cos",
            "neigh_cpu_mean",
            "neigh_cpu_slope3",
            "neigh_rps_z30_mean",
            "neigh_rps_slope5_mean",
            "node_load_mean",
            "node_load_max",
            "node_cpu_mean",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
    },
}


def get_feature_set(name: str) -> Dict[str, Any]:
    if name not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature_set='{name}'. Available: {list(FEATURE_SETS.keys())}"
        )
    spec = dict(FEATURE_SETS[name])
    feats = list(spec["features"])

    if "targets" in spec:
        target_feats = list(spec["targets"])
        spec["target"] = target_feats[0]
    elif "target" in spec:
        target_feats = [str(spec["target"])]
        spec["targets"] = target_feats
    else:
        raise KeyError(
            f"feature_set='{name}' must define 'target' or 'targets'"
        )

    for tf in target_feats:
        if tf not in FEATURES:
            raise KeyError(
                f"feature_set='{name}': unknown target '{tf}' "
                f"(must be defined in FEATURES)"
            )
        # NOTE: a target may live outside the input features (e.g. http_time
        # predicts cpu_utilization from http+time inputs). Consumers that need
        # target data (build_windows aggregation, simulator actuals) source
        # such target-only columns explicitly; model inputs stay as listed.
    for f in feats:
        if f not in FEATURES:
            raise KeyError(
                f"feature_set='{name}': feature '{f}' not defined in FEATURES"
            )
    return spec


def feature_names_for_feature_set(feature_set: str) -> List[str]:
    return list(get_feature_set(feature_set)["features"])


def target_feature_for_feature_set(feature_set: str) -> str:
    return str(get_feature_set(feature_set)["target"])


def target_features_for_feature_set(feature_set: str) -> List[str]:
    return list(get_feature_set(feature_set)["targets"])


def is_derived_feature(feature_name: str) -> bool:
    """Whether a feature is synthesized from the timestamp (minute/hour/day)
    rather than read from a parquet column."""
    return bool(FEATURES.get(feature_name, {}).get("derived", False))


def derived_time_value(feature_name: str, minute_index: int) -> int:
    """Exact calendar component for a relative minute index (t=0 is 00:00 of
    day 0): minute-of-hour (0-59), hour-of-day (0-23), day-index (0-N)."""
    if feature_name == "minute":
        return minute_index % 60
    if feature_name == "hour":
        return (minute_index // 60) % 24
    if feature_name == "day":
        return minute_index // 1440
    raise KeyError(f"Unknown derived time feature '{feature_name}'")


def _sourced_names(spec: Dict[str, Any]) -> List[str]:
    """Input features plus target-only extras (targets outside the inputs,
    e.g. cpu_utilization for http_time) that still need table sourcing."""
    names = list(spec["features"])
    for tf in spec.get("targets", []):
        if tf not in names:
            names.append(tf)
    return names


def tables_for_feature_set(feature_set: str) -> Set[str]:
    spec = get_feature_set(feature_set)
    return {FEATURES[f]["table"]
            for f in _sourced_names(spec) if not is_derived_feature(f)}


def table_to_raw_columns(feature_set: str) -> Dict[str, List[str]]:
    spec = get_feature_set(feature_set)
    out: Dict[str, List[str]] = {}
    for feat_name in _sourced_names(spec):
        if is_derived_feature(feat_name):
            continue
        meta = FEATURES[feat_name]
        t = meta["table"]
        c = meta["column"]
        out.setdefault(t, [])
        if c not in out[t]:
            out[t].append(c)
    return out


def table_to_feature_exprs(feature_set: str) -> Dict[str, List[tuple]]:
    spec = get_feature_set(feature_set)
    out: Dict[str, List[tuple]] = {}
    for feat_name in _sourced_names(spec):
        if is_derived_feature(feat_name):
            continue
        meta = FEATURES[feat_name]
        t = meta["table"]
        c = meta["column"]
        out.setdefault(t, [])
        out[t].append((feat_name, c))
    return out


def is_mcr_feature(feature_name: str) -> bool:
    """Whether a feature is an MCR-family column (rpc/http/mcr rate columns
    from msrtmcre). Single source of truth for the --normalize_mcr column rule
    shared by build_windows and the simulator."""
    n = feature_name.lower()
    return "mcr" in n or "rpc" in n or "http" in n


def mcr_column_indices(feature_names: List[str]) -> List[int]:
    """Channel positions of MCR-family features within an ordered feature list."""
    return [i for i, f in enumerate(feature_names) if is_mcr_feature(f)]


def normalize_mcr_array(arr: np.ndarray, cols: List[int],
                        lo_hi: Optional[Dict[int, tuple]] = None) -> Dict[int, tuple]:
    """In-place per-column [0,1] min-max normalization of MCR channels.

    With lo_hi=None the bounds are computed over arr (per-service scope when
    arr holds one service's full timeline); otherwise the given global bounds
    are applied. Zero-range columns become 0.0. Returns {col: (lo, hi)}.
    """
    out: Dict[int, tuple] = {}
    for j in cols:
        if lo_hi is not None:
            lo, hi = lo_hi[j]
        else:
            col = arr[:, j]
            lo, hi = float(col.min()), float(col.max())
        if hi - lo > 1e-12:
            arr[:, j] = (arr[:, j] - lo) / (hi - lo)
        else:
            arr[:, j] = 0.0
        out[j] = (lo, hi)
    return out

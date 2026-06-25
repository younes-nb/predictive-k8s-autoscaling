from typing import Dict, Any, List, Set

FEATURES: Dict[str, Dict[str, str]] = {
    "cpu_utilization": {"table": "msresource", "column": "cpu_utilization"},
    "memory_utilization": {"table": "msresource", "column": "memory_utilization"},
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
    "cpu_delta_upstream": {
        "features": [
            "cpu_utilization",
            "cpu_utilization_delta",
            "upstream_cpu_utilization_mean",
            "upstream_cpu_utilization_delta_mean",
        ],
        "target": "cpu_utilization",
        "base_table": "msresource",
        "requires_callgraph": True,
        "requires_delta": True,
    },
    "cpu_mem_both": {
        "features": ["cpu_utilization", "memory_utilization"],
        "targets": ["cpu_utilization", "memory_utilization"],
        "base_table": "msresource",
    },
}


DERIVED_FEATURES = {
    "cpu_utilization_delta",
    "upstream_cpu_utilization_mean",
    "upstream_cpu_utilization_delta_mean",
}


def is_derived_feature(feat_name: str) -> bool:
    return feat_name in DERIVED_FEATURES


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
        if tf not in feats:
            raise ValueError(
                f"feature_set='{name}': target='{tf}' must be included in features={feats}"
            )
    for f in feats:
        if f not in FEATURES and not is_derived_feature(f):
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


def tables_for_feature_set(feature_set: str) -> Set[str]:
    feats = feature_names_for_feature_set(feature_set)
    return {FEATURES[f]["table"] for f in feats if not is_derived_feature(f)}


def table_to_raw_columns(feature_set: str) -> Dict[str, List[str]]:
    spec = get_feature_set(feature_set)
    out: Dict[str, List[str]] = {}
    for feat_name in spec["features"]:
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
    for feat_name in spec["features"]:
        if is_derived_feature(feat_name):
            continue
        meta = FEATURES[feat_name]
        t = meta["table"]
        c = meta["column"]
        out.setdefault(t, [])
        out[t].append((feat_name, c))
    return out

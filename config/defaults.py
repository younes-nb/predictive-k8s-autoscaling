from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Set
import os


@dataclass(frozen=True)
class Paths:
    RAW_ROOT: str = "/dataset/raw"
    PARQUET_ROOT: str = "/dataset/parquet"
    RAW_MSRESOURCE: str = "/dataset/raw/msresource"
    PARQUET_MSRESOURCE: str = "/dataset/parquet/msresource"
    RAW_NODE: str = "/dataset/raw/node"
    PARQUET_NODE: str = "/dataset/parquet/node"
    RAW_MSRTMCRE: str = "/dataset/raw/msrtmcre"
    PARQUET_MSRTMCRE: str = "/dataset/parquet/msrtmcre"
    WINDOWS_DIR: str = "/dataset/windows"
    MODELS_DIR: str = "/dataset/models"
    LOGS_DIR: str = "/dataset/logs"


PATHS = Paths()

DEFAULT_CHECKPOINT_PATH = os.path.join(PATHS.MODELS_DIR, "model.pt")

DATASET_TABLES: Dict[str, Dict[str, Any]] = {
    "msresource": {
        "prefix": "MSMetricsUpdate/MSMetricsUpdate",
        "ratio_min": 30,
        "raw_dir": PATHS.RAW_MSRESOURCE,
        "parquet_dir": PATHS.PARQUET_MSRESOURCE,
        "key_cols": ["msname", "msinstanceid", "nodeid"],
    },
    "node": {
        "prefix": "NodeMetricsUpdate/NodeMetricsUpdate",
        "ratio_min": 30,
        "raw_dir": PATHS.RAW_NODE,
        "parquet_dir": PATHS.PARQUET_NODE,
        "key_cols": ["nodeid"],
    },
    "msrtmcre": {
        "prefix": "MCRRTUpdate/MCRRTUpdate",
        "ratio_min": 30,
        "raw_dir": PATHS.RAW_MSRTMCRE,
        "parquet_dir": PATHS.PARQUET_MSRTMCRE,
        "key_cols": ["msname", "msinstanceid"],
    },
}


FEATURES: Dict[str, Dict[str, str]] = {
    "cpu_utilization": {"table": "msresource", "column": "cpu_utilization"},
    "memory_utilization": {"table": "msresource", "column": "memory_utilization"},
    "node_cpu_utilization": {"table": "node", "column": "cpu_utilization"},
    "node_memory_utilization": {"table": "node", "column": "memory_utilization"},
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
}


def get_feature_set(name: str) -> Dict[str, Any]:
    if name not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature_set='{name}'. Available: {list(FEATURE_SETS.keys())}"
        )
    spec = FEATURE_SETS[name]
    feats = list(spec["features"])
    target = str(spec["target"])

    if target not in feats:
        raise ValueError(
            f"feature_set='{name}': target='{target}' must be included in features={feats}"
        )
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


def tables_for_feature_set(feature_set: str) -> Set[str]:
    feats = feature_names_for_feature_set(feature_set)
    return {FEATURES[f]["table"] for f in feats}


def table_to_raw_columns(feature_set: str) -> Dict[str, List[str]]:
    spec = get_feature_set(feature_set)
    out: Dict[str, List[str]] = {}
    for feat_name in spec["features"]:
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
        meta = FEATURES[feat_name]
        t = meta["table"]
        c = meta["column"]
        out.setdefault(t, [])
        out[t].append((feat_name, c))
    return out


@dataclass(frozen=True)
class PreprocessingDefaults:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 10
    TRAIN_FRAC: float = 0.7
    VAL_FRAC: float = 0.1
    SMOOTHING_WINDOW: int = 5
    REPARTITION: int = 4
    TIME_COL: str = "timestamp_dt"
    ID_COLS: Tuple[str, ...] = ("msname", "msinstanceid")
    SERVICE_COL: str = "msname"
    FREQ: str = "1m"
    MAX_SERVICES: int = 2000
    SUBSET_SEED: int = 42
    FEATURE_SET: str = "cpu"


@dataclass(frozen=True)
class TrainingDefaults:
    HIDDEN_SIZE: int = 96
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.15
    BATCH_SIZE: int = 512
    EPOCHS: int = 10
    LR: float = 1e-3
    GRAD_CLIP: float = 1.0
    NUM_WORKERS: int = 8
    SEED: int = 42
    LOG_INTERVAL: int = 2000
    USE_WEIGHTS: bool = True
    MC_REPEATS: int = 25
    TAU_BASE: float = 0.80
    K_UNCERTAINTY: float = 2.0
    GAMMA: float = 6.0
    DELTA: float = 0.05
    THETA_MIN: float = 0.60
    THETA_MAX: float = 0.90
    INFERENCE_REPEATS: int = 100


PREPROCESSING = PreprocessingDefaults()
TRAINING = TrainingDefaults()

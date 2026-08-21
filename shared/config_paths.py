from dataclasses import dataclass
from typing import Dict, Any
import os

from shared.config_env import get_env


@dataclass(frozen=True)
class Paths:
    RAW_ROOT: str = get_env("RAW_ROOT", "/dataset/raw")
    PARQUET_ROOT: str = get_env("PARQUET_ROOT", "/dataset/parquet")
    RAW_MSRESOURCE: str = get_env("RAW_MSRESOURCE", "/dataset/raw/msresource")
    PARQUET_MSRESOURCE: str = get_env("PARQUET_MSRESOURCE", "/dataset/parquet/msresource")
    SERVICE_CACHE_FILE: str = get_env("SERVICE_CACHE_FILE", "/dataset/parquet/.service_cache.json")
    RAW_NODE: str = get_env("RAW_NODE", "/dataset/raw/node")
    PARQUET_NODE: str = get_env("PARQUET_NODE", "/dataset/parquet/node")
    RAW_MSRTMCRE: str = get_env("RAW_MSRTMCRE", "/dataset/raw/msrtmcre")
    PARQUET_MSRTMCRE: str = get_env("PARQUET_MSRTMCRE", "/dataset/parquet/msrtmcre")
    RAW_MSCALLGRAPH: str = get_env("RAW_MSCALLGRAPH", "/dataset/raw/mscallgraph")
    PARQUET_MSCALLGRAPH: str = get_env("PARQUET_MSCALLGRAPH", "/dataset/parquet/mscallgraph")
    PARQUET_THRESHOLD_MSRESOURCE: str = get_env("PARQUET_THRESHOLD_MSRESOURCE", "/dataset/threshold/msresource")
    PARQUET_THRESHOLD_MSRTMCRE: str = get_env("PARQUET_THRESHOLD_MSRTMCRE", "/dataset/threshold/msrtmcre")
    WINDOWS_DIR: str = get_env("WINDOWS_DIR", "/dataset/windows")
    MODELS_DIR: str = get_env("MODELS_DIR", "/proj/k8sautoscaledl-PG0/models")
    CHECKPOINT_PATH: str = os.path.join(MODELS_DIR, "model.pt")
    LOGS_DIR: str = get_env("LOGS_DIR", "/proj/k8sautoscaledl-PG0/logs")
    ANALYTICS_OUT_DIR: str = get_env("ANALYTICS_OUT_DIR", "/proj/k8sautoscaledl-PG0/analytics_out")
    RESUME_STATE_FILE: str = get_env("RESUME_STATE_FILE", "/proj/k8sautoscaledl-PG0/train_resume_state.pt")


PATHS = Paths()

DEFAULT_CHECKPOINT_PATH = PATHS.CHECKPOINT_PATH

DATASET_TABLES: Dict[str, Dict[str, Any]] = {
    "msresource": {
        "prefix": "MSMetricsUpdate/MSMetricsUpdate",
        "ratio_min": 30,
        "raw_dir": PATHS.RAW_MSRESOURCE,
        "parquet_dir": PATHS.PARQUET_MSRESOURCE,
        "key_cols": ["msname", "msinstanceid"],
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
        "ratio_min": 3,
        "raw_dir": PATHS.RAW_MSRTMCRE,
        "parquet_dir": PATHS.PARQUET_MSRTMCRE,
        "key_cols": ["msname", "msinstanceid"],
    },
    "mscallgraph": {
        "prefix": "CallGraph/CallGraph",
        "ratio_min": 30,
        "raw_dir": PATHS.RAW_MSCALLGRAPH,
        "parquet_dir": PATHS.PARQUET_MSCALLGRAPH,
        "key_cols": ["traceid", "rpc_id"],
    },
}

from dataclasses import dataclass
from typing import Dict, Any
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
    RAW_MSCALLGRAPH: str = "/dataset/raw/mscallgraph"
    PARQUET_MSCALLGRAPH: str = "/dataset/parquet/mscallgraph"
    PARQUET_THRESHOLD_MSRESOURCE: str = "/dataset/threshold/msresource"
    PARQUET_THRESHOLD_MSRTMCRE: str = "/dataset/threshold/msrtmcre"
    WINDOWS_DIR: str = "/dataset/windows"
    MODELS_DIR: str = "/proj/k8sautoscaledl-PG0/models"
    CHECKPOINT_PATH: str = os.path.join(MODELS_DIR, "model_global.pt")
    LOGS_DIR: str = "/proj/k8sautoscaledl-PG0/logs"
    RESUME_STATE_FILE: str = "/proj/k8sautoscaledl-PG0/train_resume_state.pt"


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
        "ratio_min": 30,
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

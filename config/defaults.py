from dataclasses import dataclass
from typing import Tuple, Dict, Any
import os


@dataclass(frozen=True)
class Paths:
    RAW_MSRESOURCE: str = "/dataset/raw/msresource"
    PARQUET_MSRESOURCE: str = "/dataset/parquet/msresource"
    WINDOWS_DIR: str = "/dataset/windows"
    MODELS_DIR: str = "/dataset/models"
    LOGS_DIR: str = "/dataset/logs"


FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "feature_cols": ["cpu_utilization"],
        "target_col": "cpu_utilization",
    },
    "cpu_mem": {
        "feature_cols": ["cpu_utilization", "mem_utilization"],
        "target_col": "cpu_utilization",
    },
}


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
    FREQ: str = "1m"
    SERVICE_COL: str = "msname"
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
    INFERENCE_REPEATS: int = 100


PATHS = Paths()
PREPROCESSING = PreprocessingDefaults()
TRAINING = TrainingDefaults()

DEFAULT_CHECKPOINT_PATH = os.path.join(PATHS.MODELS_DIR, "model.pt")

DATASET_TABLES = {
    "msresource": {
        "prefix": "MSMetricsUpdate/MSMetricsUpdate",
        "ratio_min": 30,
    },
}

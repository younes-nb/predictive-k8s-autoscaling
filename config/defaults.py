from dataclasses import dataclass
from typing import Tuple
import os


@dataclass(frozen=True)
class Paths:
    RAW_MSRESOURCE: str = "/dataset/raw/msresource"
    PARQUET_MSRESOURCE: str = "/dataset/parquet/msresource"
    WINDOWS_DIR: str = "/dataset/windows"
    MODELS_DIR: str = "/dataset/models"
    LOGS_DIR: str = "/dataset/logs"


@dataclass(frozen=True)
class PreprocessingDefaults:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 10
    TRAIN_FRAC: float = 0.6
    VAL_FRAC: float = 0.2
    SMOOTHING_WINDOW: int = 5
    REPARTITION: int = 4
    TIME_COL: str = "timestamp_dt"
    TARGET_COL: str = "cpu_utilization"
    ID_COLS: Tuple[str, ...] = ("msname", "msinstanceid")
    FREQ: str = "1m"
    SERVICE_COL: str = "msname"
    MAX_SERVICES: int = 2000
    SUBSET_SEED: int = 42


@dataclass(frozen=True)
class TrainingDefaults:
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.1
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

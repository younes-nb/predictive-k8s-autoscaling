from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class PreprocessingDefaults:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 15
    STRIDE: int = 10
    TRAIN_FRAC: float = 0.7
    VAL_FRAC: float = 0.1
    SMOOTHING_WINDOW: int = 5
    REPARTITION: int = 4
    TIME_COL: str = "timestamp_dt"
    ID_COLS: Tuple[str, ...] = ("msname", "msinstanceid")
    SERVICE_COL: str = "msname"
    FREQ: str = "1m"
    MAX_SERVICES: Optional[int] = None
    SUBSET_SEED: int = 42
    FEATURE_SET: str = "cpu_mem_both"
    SMOTE_TOMEK: bool = False


PREPROCESSING = PreprocessingDefaults()

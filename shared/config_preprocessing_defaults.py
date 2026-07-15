from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class PreprocessingDefaults:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5
    TRAIN_FRAC: float = 0.7
    VAL_FRAC: float = 0.1
    REPARTITION: int = 4
    TIME_COL: str = "timestamp_dt"
    ID_COLS: Tuple[str, ...] = ("msname", "msinstanceid")
    SERVICE_COL: str = "msname"
    FREQ: str = "1m"
    MAX_SERVICES: Optional[int] = None
    SUBSET_SEED: int = 42
    FEATURE_SET: str = "cpu_mem_both"


PREPROCESSING = PreprocessingDefaults()

from dataclasses import dataclass
from typing import Tuple, Optional

from shared.config_env import get_env


@dataclass(frozen=True)
class PreprocessingDefaults:
    INPUT_LEN: int = get_env("INPUT_LEN", 128, int)
    PRED_HORIZON: int = get_env("PRED_HORIZON", 5, int)
    STRIDE: int = get_env("STRIDE", 5, int)
    TRAIN_FRAC: float = get_env("TRAIN_FRAC", 0.7, float)
    VAL_FRAC: float = get_env("VAL_FRAC", 0.1, float)
    REPARTITION: int = get_env("REPARTITION", 4, int)
    TIME_COL: str = get_env("TIME_COL", "timestamp_dt")
    ID_COLS: Tuple[str, ...] = tuple(get_env("ID_COLS", ["msname", "msinstanceid"], is_list=True))
    SERVICE_COL: str = get_env("SERVICE_COL", "msname")
    FREQ: str = get_env("FREQ", "1m")
    MAX_SERVICES: Optional[int] = get_env("MAX_SERVICES", None, int)
    SUBSET_SEED: int = get_env("SUBSET_SEED", 42, int)
    FEATURE_SET: str = get_env("FEATURE_SET", "cpu_mem_both")


PREPROCESSING = PreprocessingDefaults()

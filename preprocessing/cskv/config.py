
import random

import numpy as np
import torch
from dataclasses import dataclass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class CskvConfig:
    KERNEL_SIZES: tuple = (2, 4, 8)
    CONV1_OUT_CH: int = 32
    CONV2_OUT_CH: int = 64

    BILSTM_HIDDEN: tuple = (32, 64, 128)

    CEEMDAN_TRIALS: int = 100
    CEEMDAN_EPSILON: float = 0.005

    SE_M: int = 2
    SE_R_FRAC: float = 0.2
    SE_MAX_SAMPLES: int = 1000

    N_CLUSTERS: int = 3
    NO_CLUSTERING: bool = False

    VMD_K: int = 10
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7

CFG = CskvConfig()

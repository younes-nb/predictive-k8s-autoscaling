
from dataclasses import dataclass

@dataclass(frozen=True)

class CvcbmConfig:

    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5

    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 2048
    EPOCHS: int = 100

    KERNEL_SIZES: tuple = (2, 4, 8)
    CONV1_OUT_CH: int = 32
    CONV2_OUT_CH: int = 64

    BILSTM_HIDDEN: tuple = (32, 64, 128)

    CEEMDAN_TRIALS: int = 30
    CEEMDAN_EPSILON: float = 0.005

    SE_M: int = 2
    SE_R_FRAC: float = 0.2
    SE_MAX_SAMPLES: int = 1000

    N_CLUSTERS: int = 3

    VMD_K: int = 5
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10
    STRIDE: int = 5

CFG = CvcbmConfig()

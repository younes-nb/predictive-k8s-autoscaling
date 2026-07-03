from dataclasses import dataclass


@dataclass(frozen=True)
class CshvtbConfig:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5

    LEARNING_RATE: float = 0.0003
    BATCH_SIZE: int = 4096
    EPOCHS: int = 100
    WEIGHT_INIT_LOW: float = 0.0
    WEIGHT_INIT_HIGH: float = 0.0015

    CEEMDAN_TRIALS: int = 30
    CEEMDAN_EPSILON: float = 0.005

    SE_M: int = 2
    SE_R_FRAC: float = 0.2
    SE_MAX_SAMPLES: int = 1000

    HDBSCAN_MIN_CLUSTER_SIZE: int = 2
    HDBSCAN_MIN_SAMPLES: int = 1

    VMD_K: int = 5
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7

    N_LOWFREQ_CHANNELS: int = 1
    TOTAL_CHANNELS: int = 6

    TCN_KERNEL_SIZE: int = 3
    TCN_FILTERS: tuple = (256, 256)
    TCN_DILATIONS: tuple = (1, 2, 4)
    TCN_DROPOUT: float = 0.2

    BIGRU_HIDDEN: tuple = (64, 128)

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10


CFG = CshvtbConfig()

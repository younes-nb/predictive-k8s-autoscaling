from dataclasses import dataclass


@dataclass(frozen=True)
class SdtnetConfig:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10

    # --- SVMD (primary decomposition) ---
    SVMD_MAX_MODES: int = 8
    SVMD_ENERGY_RATIO_TOL: float = 0.005
    SVMD_ALPHA: int = 2000
    SVMD_TAU: float = 0.0
    SVMD_TOL: float = 1e-7

    # --- Dispersion Entropy ---
    DE_CLASSES: int = 6
    DE_EMBED_DIM: int = 2
    DE_TIME_DELAY: int = 1

    # --- Quantile banding ---
    DE_HIGH_QUANTILE: float = 0.60

    # --- SVMD (secondary decomposition) ---
    SVMD2_MAX_MODES: int = 5
    SVMD2_ENERGY_RATIO_TOL: float = 0.01
    SVMD2_ALPHA: int = 2000
    SVMD2_TAU: float = 0.0
    SVMD2_TOL: float = 1e-7

    # Fixed channel contract
    TOTAL_CHANNELS: int = 6

    # --- TCN ---
    TCN_NUM_FILTERS: int = 64
    TCN_KERNEL_SIZE: int = 3
    TCN_DILATIONS: tuple = (1, 2, 4, 8)
    TCN_DROPOUT: float = 0.1
    RESIDUAL_PREDICTION: bool = True

    # --- Loss ---
    DELTA_LOSS_WEIGHT: float = 0.2
    DIRECTION_LOSS_WEIGHT: float = 0.01

    # --- LR Schedule ---
    USE_LR_SCHEDULER: bool = True

    # --- Training ---
    LEARNING_RATE: float = 0.0005
    BATCH_SIZE: int = 4096  # can raise toward 8192 if no CUDA OOM
    EPOCHS: int = 100
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0


CFG = SdtnetConfig()

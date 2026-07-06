from dataclasses import dataclass


@dataclass(frozen=True)
class SdtnetConfig:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10

    # --- SVMD (primary decomposition) ---
    SVMD_MAX_MODES: int = 20
    SVMD_ENERGY_RATIO_TOL: float = 0.005
    SVMD_ALPHA: int = 2000
    SVMD_TAU: float = 0.0
    SVMD_TOL: float = 1e-7

    # --- Dispersion Entropy ---
    DE_CLASSES: int = 6
    DE_EMBED_DIM: int = 2
    DE_TIME_DELAY: int = 1

    # --- Quantile banding ---
    DE_HIGH_QUANTILE: float = 0.40

    # --- SVMD (secondary decomposition) ---
    SVMD2_MAX_MODES: int = 12
    SVMD2_ENERGY_RATIO_TOL: float = 0.01
    SVMD2_ALPHA: int = 2000
    SVMD2_TAU: float = 0.0
    SVMD2_TOL: float = 1e-7

    # Fixed channel contract (SVMD2_MAX_MODES + lowfreq + residual)
    TOTAL_CHANNELS: int = 14

    # --- CNN+BiLSTM+Attention ---
    KERNEL_SIZES: tuple = (2, 4, 8)
    CONV1_OUT_CH: int = 32
    CONV2_OUT_CH: int = 64
    BILSTM_HIDDEN: tuple = (32, 64, 128, 128)
    DROPOUT: float = 0.2

    # --- Training ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 2048
    EPOCHS: int = 100
    WEIGHT_DECAY: float = 0.0
    GRAD_CLIP_NORM: float = 1.0


CFG = SdtnetConfig()

from dataclasses import dataclass


@dataclass(frozen=True)
class TsdpConfig:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10

    # --- SVMD (adaptive decomposition with center-frequency progression) ---
    SVMD_ALPHA: int = 2000
    SVMD_TAU: float = 0.0
    SVMD_TOL: float = 1e-7

    # --- Dispersion Entropy ---
    DE_CLASSES: int = 6
    DE_EMBED_DIM: int = 2
    DE_TIME_DELAY: int = 1

    # MODWT (sym4, level=3) = 4 channels + low composite = 5 total
    TOTAL_CHANNELS: int = 5

    # --- Shared Training ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 2048
    EPOCHS: int = 100
    WEIGHT_DECAY: float = 0.0
    GRAD_CLIP_NORM: float = 1.0


CFG = TsdpConfig()

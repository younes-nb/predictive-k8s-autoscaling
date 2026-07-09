from dataclasses import dataclass


@dataclass(frozen=True)
class TsdpConfig:
    INPUT_LEN: int = 60
    PRED_HORIZON: int = 5
    STRIDE: int = 5

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10

    # --- VMD (applied to D1 after MODWT) ---
    VMD_K: int = 10
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7

    # MODWT (sym4, level=3) on signal: [D1, D2, D3, A3],
    # then VMD(K=10) on D1 → 10 VMD modes, total = 10 + 3 = 13
    TOTAL_CHANNELS: int = 13

    # --- Shared Training ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 16384
    EPOCHS: int = 1000
    WEIGHT_DECAY: float = 0.0
    GRAD_CLIP_NORM: float = 1.0
    LOSS_CHANGE_THRESHOLD: float = 1e-5
    NO_CHANGE_EPOCHS_LIMIT: int = 50


CFG = TsdpConfig()

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TrainingDefaults:
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.3
    BATCH_SIZE: int = 512
    EPOCHS: int = 100
    LR: float = 0.00127
    HIDDEN_SIZE_OPTIONS: Tuple[int, ...] = (32, 64, 128, 256)
    NUM_LAYERS_OPTIONS: Tuple[int, ...] = (1, 2, 3, 4)
    DROPOUT_RANGE: Tuple[float, float] = (0.1, 0.5)
    LR_RANGE: Tuple[float, float] = (5e-4, 5e-3)
    HYPERPARAM_SAMPLE_ATTEMPTS: int = 5000
    HYPERPARAM_CHECK_INTERVAL: int = 50
    LOSS_CHANGE_THRESHOLD: float = 1e-4
    GRAD_CLIP: float = 1.0
    WEIGHT_DECAY: float = 1e-4
    UNDER_PENALTY: float = 8.0
    SEED: int = 42
    USE_WEIGHTS: bool = True
    GAMMA: float = 20.0
    DELTA: float = 0.08
    THETA_MODE: str = "adaptive"
    THETA_BASE: float = 0.75
    THETA_MIN: float = 0.60
    BIDIRECTIONAL: bool = False
    PROBABILISTIC_TRAINING: bool = False
    QUANTILES: Tuple[float, ...] = (0.5, 0.9, 0.95)
    HYPERPARAM_OPTIMIZER: str = "sfoa"
    SFOA_POPULATION: int = 20
    SFOA_ITERATIONS: int = 10
    SFOA_EVAL_EPOCHS: int = 10
    SFOA_GP: float = 0.5
    SFOA_EVALUATION_PARALLEL: bool = True
    NO_CHANGE_EPOCHS_LIMIT: int = 100


TRAINING = TrainingDefaults()

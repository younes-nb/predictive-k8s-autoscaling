from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TrainingDefaults:
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.5
    BATCH_SIZE: int = 512
    EPOCHS: int = 1000
    LR: float = 0.0005
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
    USE_WEIGHTS: bool = False
    GAMMA: float = 20.0
    DELTA: float = 0.08
    THETA_MODE: str = "adaptive"
    THETA_BASE: float = 0.75
    THETA_MIN: float = 0.60
    BIDIRECTIONAL: bool = True
    PROBABILISTIC_TRAINING: bool = False
    QUANTILES: Tuple[float, ...] = (0.5, 0.9, 0.95)
    HYPERPARAM_OPTIMIZER: str = "none"
    SFOA_POPULATION: int = 10
    SFOA_ITERATIONS: int = 5
    SFOA_EVAL_EPOCHS: int = 5
    SFOA_GP: float = 0.5
    SFOA_EVALUATION_PARALLEL: bool = True
    SFOA_TRAIN_PCT: float = 10.0
    SFOA_VAL_PCT: float = 10.0
    SFOA_NUM_WORKERS: int = 40
    TRAIN_PCT: float = 100.0
    VAL_PCT: float = 100.0
    TEST_PCT: float = 100.0
    NO_CHANGE_EPOCHS_LIMIT: int = 50


TRAINING = TrainingDefaults()

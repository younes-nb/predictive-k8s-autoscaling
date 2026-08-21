from dataclasses import dataclass
from typing import Tuple

from shared.config_env import get_env


@dataclass(frozen=True)
class TrainingDefaults:
    HIDDEN_SIZE: int = get_env("HIDDEN_SIZE", 64, int)
    NUM_LAYERS: int = get_env("NUM_LAYERS", 3, int)
    DROPOUT: float = get_env("DROPOUT", 0.5, float)
    BATCH_SIZE: int = get_env("BATCH_SIZE", 4096, int)
    NUM_WORKERS: int = get_env("NUM_WORKERS", 12, int)
    EPOCHS: int = get_env("EPOCHS", 1000, int)
    LR: float = get_env("LR", 0.0005, float)
    HIDDEN_SIZE_OPTIONS: Tuple[int, ...] = tuple(get_env("HIDDEN_SIZE_OPTIONS", [32, 64, 128, 256], is_list=True))
    NUM_LAYERS_OPTIONS: Tuple[int, ...] = tuple(get_env("NUM_LAYERS_OPTIONS", [1, 2, 3, 4], is_list=True))
    DROPOUT_RANGE: Tuple[float, float] = tuple(get_env("DROPOUT_RANGE", [0.1, 0.5], is_list=True))
    LR_RANGE: Tuple[float, float] = tuple(get_env("LR_RANGE", [5e-4, 5e-3], is_list=True))
    HYPERPARAM_SAMPLE_ATTEMPTS: int = get_env("HYPERPARAM_SAMPLE_ATTEMPTS", 5000, int)
    HYPERPARAM_CHECK_INTERVAL: int = get_env("HYPERPARAM_CHECK_INTERVAL", 50, int)
    LOSS_CHANGE_THRESHOLD: float = get_env("LOSS_CHANGE_THRESHOLD", 1e-5, float)
    EARLY_STOP_PATIENCE: int = get_env("EARLY_STOP_PATIENCE", 20, int)
    EARLY_STOP_MIN_DELTA: float = get_env("EARLY_STOP_MIN_DELTA", 1e-6, float)
    LR_SCHEDULER: str = get_env("LR_SCHEDULER", "ReduceLROnPlateau")
    LR_REDUCE_PATIENCE: int = get_env("LR_REDUCE_PATIENCE", 10, int)
    LR_REDUCE_FACTOR: float = get_env("LR_REDUCE_FACTOR", 0.5, float)
    LR_MIN: float = get_env("LR_MIN", 1e-6, float)
    GRAD_CLIP: float = get_env("GRAD_CLIP", 1.0, float)
    WEIGHT_DECAY: float = get_env("WEIGHT_DECAY", 1e-4, float)
    SEED: int = get_env("SEED", 42, int)
    BIDIRECTIONAL: bool = get_env("BIDIRECTIONAL", False, bool)
    PROBABILISTIC_TRAINING: bool = get_env("PROBABILISTIC_TRAINING", False, bool)
    QUANTILES: Tuple[float, ...] = tuple(get_env("QUANTILES", [0.5, 0.9, 0.95], is_list=True))
    HYPERPARAM_OPTIMIZER: str = get_env("HYPERPARAM_OPTIMIZER", "none")
    SFOA_POPULATION: int = get_env("SFOA_POPULATION", 10, int)
    SFOA_ITERATIONS: int = get_env("SFOA_ITERATIONS", 5, int)
    SFOA_EVAL_EPOCHS: int = get_env("SFOA_EVAL_EPOCHS", 10, int)
    SFOA_GP: float = get_env("SFOA_GP", 0.5, float)
    SFOA_EVALUATION_PARALLEL: bool = get_env("SFOA_EVALUATION_PARALLEL", True, bool)
    SFOA_TRAIN_PCT: float = get_env("SFOA_TRAIN_PCT", 20.0, float)
    SFOA_VAL_PCT: float = get_env("SFOA_VAL_PCT", 20.0, float)
    SFOA_NUM_WORKERS: int = get_env("SFOA_NUM_WORKERS", 4, int)
    TRAIN_PCT: float = get_env("TRAIN_PCT", 100.0, float)
    VAL_PCT: float = get_env("VAL_PCT", 100.0, float)
    TEST_PCT: float = get_env("TEST_PCT", 100.0, float)


TRAINING = TrainingDefaults()

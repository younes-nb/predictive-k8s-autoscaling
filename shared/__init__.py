import importlib

__all__ = [
    "PATHS",
    "DATASET_TABLES",
    "DEFAULT_CHECKPOINT_PATH",
    "Paths",
    "PREPROCESSING",
    "PreprocessingDefaults",
    "TRAINING",
    "TrainingDefaults",
    "FEATURES",
    "FEATURE_SETS",
    "get_feature_set",
    "feature_names_for_feature_set",
    "target_feature_for_feature_set",
    "target_features_for_feature_set",
    "tables_for_feature_set",
    "table_to_raw_columns",
    "table_to_feature_exprs",
    "setup_logging",
    "log_configs",
    "run",
]

_LAZY = {
    "PATHS": "config_paths",
    "DATASET_TABLES": "config_paths",
    "DEFAULT_CHECKPOINT_PATH": "config_paths",
    "Paths": "config_paths",
    "PREPROCESSING": "config_preprocessing_defaults",
    "PreprocessingDefaults": "config_preprocessing_defaults",
    "TRAINING": "config_training_defaults",
    "TrainingDefaults": "config_training_defaults",
    "FEATURES": "features",
    "FEATURE_SETS": "features",
    "get_feature_set": "features",
    "feature_names_for_feature_set": "features",
    "target_feature_for_feature_set": "features",
    "target_features_for_feature_set": "features",
    "tables_for_feature_set": "features",
    "table_to_raw_columns": "features",
    "table_to_feature_exprs": "features",
    "setup_logging": "logging_utils",
    "log_configs": "logging_utils",
    "run": "subprocess_utils",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY.keys()))

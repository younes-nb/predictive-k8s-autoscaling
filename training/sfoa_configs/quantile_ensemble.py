from types import SimpleNamespace
from core.architectures.ensemble import QuantileEnsembleForecaster


SEARCH_SPACE = [
    {"name": "dropout", "type": "float", "low": 0.1, "high": 0.4, "log": False},
    {"name": "lr", "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    {"name": "hidden_size", "type": "int", "low": 64, "high": 256, "log": False},
    {"name": "ensemble_size", "type": "int", "low": 3, "high": 7, "log": False},
]

DEFAULTS = {
    "dropout": 0.2,
    "lr": 1e-3,
    "hidden_size": 128,
    "ensemble_size": 5,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    quantiles = [0.05, 0.5, 0.95]
    model = QuantileEnsembleForecaster(
        input_size=input_size,
        hidden_size=hyperparams.get("hidden_size", DEFAULTS["hidden_size"]),
        num_layers=3,
        dropout=hyperparams.get("dropout", DEFAULTS["dropout"]),
        horizon=args.pred_horizon,
        num_targets=num_targets,
        quantiles=quantiles,
        ensemble_size=hyperparams.get("ensemble_size", DEFAULTS["ensemble_size"]),
    ).to(device)
    return model
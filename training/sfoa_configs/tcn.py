SEARCH_SPACE = [
    {"name": "hidden_size", "type": "categorical", "options": [32, 64, 128, 256]},
    {"name": "num_layers", "type": "categorical", "options": [2, 3, 4, 5, 6]},
    {"name": "dropout", "type": "continuous", "low": 0.0, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 4,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.tcn import TCNForecaster
    return TCNForecaster(
        input_size=input_size,
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
        horizon=args.pred_horizon,
        num_targets=num_targets,
    ).to(device)


def build_dual_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.tcn import TCNDualHead
    return TCNDualHead(
        input_size=input_size,
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
        horizon=args.pred_horizon,
        num_targets=num_targets,
    ).to(device)

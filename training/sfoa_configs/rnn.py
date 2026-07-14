SEARCH_SPACE = [
    {"name": "hidden_size", "type": "categorical", "options": [32, 64, 128, 256]},
    {"name": "num_layers", "type": "categorical", "options": [1, 2, 3, 4]},
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.models import RNNForecaster
    model_type = args.model_type
    bidirectional = model_type in ("bilstm", "bigrue")
    rnn_type = "lstm" if model_type in ("lstm", "bilstm") else "gru"
    return RNNForecaster(
        input_size=input_size,
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
        horizon=args.pred_horizon,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        num_targets=num_targets,
    ).to(device)

SEARCH_SPACE = [
    {"name": "bilstm_hidden0", "type": "categorical", "options": [32, 64, 128]},
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "bilstm_hidden0": 32,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.models import RNNForecaster
    return RNNForecaster(
        input_size=input_size,
        hidden_size=hyperparams["bilstm_hidden0"],
        num_layers=2,
        dropout=hyperparams["dropout"],
        horizon=args.pred_horizon,
        rnn_type="lstm",
        bidirectional=True,
        num_targets=num_targets,
    ).to(device)

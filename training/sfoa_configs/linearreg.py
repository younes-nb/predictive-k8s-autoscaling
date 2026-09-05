SEARCH_SPACE = [
    {"name": "lr", "type": "log", "low": 1e-4, "high": 1e-2},
]

DEFAULTS = {
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.linearreg import LinearRegression
    return LinearRegression(
        input_size=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        num_targets=num_targets,
    ).to(device)

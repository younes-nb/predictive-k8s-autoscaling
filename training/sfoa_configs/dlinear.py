SEARCH_SPACE = [
    {"name": "kernel_size", "type": "categorical", "options": [7, 13, 25, 33]},
    {"name": "individual", "type": "categorical", "options": [False, True]},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "kernel_size": 25,
    "individual": False,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.dlinear import DLinear
    return DLinear(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        kernel_size=hyperparams.get("kernel_size", 25),
        individual=hyperparams.get("individual", False),
        num_targets=num_targets,
    ).to(device)

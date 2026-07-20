SEARCH_SPACE = [
    {"name": "tcn_filters_base", "type": "categorical", "options": [32, 64, 128]},
    {"name": "bigru_hidden0", "type": "categorical", "options": [32, 64, 128]},
    {"name": "tcn_dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "tcn_filters_base": 64,
    "bigru_hidden0": 64,
    "tcn_dropout": 0.2,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.tcn_bigru import TcnBiGru
    base = hyperparams["tcn_filters_base"]
    h0 = hyperparams["bigru_hidden0"]
    return TcnBiGru(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        tcn_kernel_size=3,
        tcn_filters=(base, base, base * 2, base * 2, base * 4),
        tcn_dilations=(1, 2, 4, 8, 16),
        tcn_dropout=hyperparams["tcn_dropout"],
        bigru_hidden=(h0, h0 * 2),
        num_targets=num_targets,
    ).to(device)

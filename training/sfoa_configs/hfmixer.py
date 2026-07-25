SEARCH_SPACE = [
    {"name": "d_group", "type": "categorical", "options": [32, 64, 128]},
    {"name": "d_time", "type": "categorical", "options": [64, 128, 256]},
    {"name": "d_feat", "type": "categorical", "options": [32, 64, 128]},
    {"name": "dropout", "type": "continuous", "low": 0.05, "high": 0.3},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "d_group": 64,
    "d_time": 128,
    "d_feat": 64,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.hfmixer import HFMixer
    return HFMixer(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        num_targets=num_targets,
        d_group=hyperparams["d_group"],
        d_time=hyperparams["d_time"],
        d_feat=hyperparams["d_feat"],
        dropout=hyperparams["dropout"],
    ).to(device)

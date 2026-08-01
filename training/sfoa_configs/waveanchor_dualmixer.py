SEARCH_SPACE = [
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.waveanchor_dualmixer import WaveAnchorDualMixer
    return WaveAnchorDualMixer(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_channels=5,
        mem_channels=3,
        dropout=hyperparams.get("dropout", 0.1),
        num_targets=num_targets,
    ).to(device)

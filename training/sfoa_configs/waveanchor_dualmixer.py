SEARCH_SPACE = [
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from preprocessing.sv.config import CFG as SV_CFG
    from core.architectures.waveanchor_dualmixer import WaveAnchorDualMixer
    swt_level = getattr(args, "swt_level", SV_CFG.SWT_LEVEL)
    mem_swt_level = getattr(args, "mem_swt_level", SV_CFG.MEM_SWT_LEVEL)
    return WaveAnchorDualMixer(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_channels=swt_level + 1,
        mem_channels=mem_swt_level + 1,
        dropout=hyperparams.get("dropout", 0.1),
        num_targets=num_targets,
    ).to(device)

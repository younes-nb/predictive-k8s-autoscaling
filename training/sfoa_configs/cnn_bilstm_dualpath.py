SEARCH_SPACE = [
    {"name": "d_group", "type": "categorical", "options": [32, 64, 128]},
    {"name": "d_time", "type": "categorical", "options": [64, 128, 256]},
    {"name": "d_feat", "type": "categorical", "options": [32, 64, 128]},
    {"name": "n_group_blocks", "type": "categorical", "options": [1, 2, 3]},
    {"name": "n_cross_blocks", "type": "categorical", "options": [1, 2, 3]},
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "d_group": 64,
    "d_time": 128,
    "d_feat": 64,
    "n_group_blocks": 2,
    "n_cross_blocks": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "mix_style": "linear_fused",
    "anchor_kind": "linear",
    "slope_span": 8,
    "head_style": "pooled_mlp",
    "pool_head_dim": 128,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.cnn_bilstm_dualpath import CnnBiLSTM_DualPath
    return CnnBiLSTM_DualPath(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_channels=5,
        mem_channels=3,
        d_group=hyperparams["d_group"],
        d_time=hyperparams["d_time"],
        d_feat=hyperparams["d_feat"],
        n_group_blocks=hyperparams["n_group_blocks"],
        n_cross_blocks=hyperparams["n_cross_blocks"],
        dropout=hyperparams["dropout"],
        mix_style=hyperparams.get("mix_style", "gelu_bottleneck"),
        anchor_kind=hyperparams.get("anchor_kind", "linear"),
        slope_span=hyperparams.get("slope_span", 8),
        head_style=hyperparams.get("head_style", "linear"),
        pool_head_dim=hyperparams.get("pool_head_dim", 128),
        num_targets=num_targets,
    ).to(device)

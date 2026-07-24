SEARCH_SPACE = [
    {"name": "osc_out", "type": "categorical", "options": [16, 32, 48]},
    {"name": "det_out", "type": "categorical", "options": [8, 16, 32]},
    {"name": "trd_out", "type": "categorical", "options": [4, 8, 16]},
    {"name": "d_msc", "type": "categorical", "options": [16, 32, 64]},
    {"name": "se_reduction", "type": "categorical", "options": [2, 4, 8]},
    {"name": "n_res_blocks", "type": "categorical", "options": [1, 2]},
    {"name": "dropout", "type": "continuous", "low": 0.05, "high": 0.3},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "osc_out": 32,
    "det_out": 16,
    "trd_out": 8,
    "d_msc": 32,
    "se_reduction": 4,
    "n_res_blocks": 1,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.farms_cnn import FARMS_CNN
    return FARMS_CNN(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        num_targets=num_targets,
        osc_out=hyperparams["osc_out"],
        det_out=hyperparams["det_out"],
        trd_out=hyperparams["trd_out"],
        d_msc=hyperparams["d_msc"],
        se_reduction=hyperparams["se_reduction"],
        n_res_blocks=hyperparams["n_res_blocks"],
        dropout=hyperparams["dropout"],
    ).to(device)

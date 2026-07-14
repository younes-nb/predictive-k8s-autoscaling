SEARCH_SPACE = [
    {"name": "conv1_out_ch", "type": "categorical", "options": [16, 32, 64]},
    {"name": "conv2_out_ch", "type": "categorical", "options": [32, 64, 128]},
    {"name": "bilstm_hidden0", "type": "categorical", "options": [32, 64, 128]},
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "conv1_out_ch": 32,
    "conv2_out_ch": 64,
    "bilstm_hidden0": 32,
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.cnn_bilstm import CnnBiLSTM
    h0 = hyperparams["bilstm_hidden0"]
    return CnnBiLSTM(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        kernel_sizes=(2, 4, 8),
        conv1_out_ch=hyperparams["conv1_out_ch"],
        conv2_out_ch=hyperparams["conv2_out_ch"],
        bilstm_hidden=(h0, h0 * 2, h0 * 4),
    ).to(device)

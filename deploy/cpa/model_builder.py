import config

RNN_MODEL_TYPES = ("lstm", "gru", "bilstm", "bigrue")


def _get(ckpt_args, hyperparams, key, default):
    if key in hyperparams:
        return hyperparams[key]
    if key in ckpt_args:
        return ckpt_args[key]
    return default


def _input_size(checkpoint, ckpt_args, hyperparams):
    return _get(ckpt_args, hyperparams, "input_size",
                checkpoint.get("input_size") or config.INPUT_SIZE)


def build_rnn(checkpoint, model_type):
    from core.models import RNNForecaster

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    rnn_type = "lstm" if model_type in ("lstm", "bilstm") else "gru"
    bidirectional = model_type in ("bilstm", "bigrue")

    return RNNForecaster(
        input_size=_input_size(checkpoint, ckpt_args, hyperparams),
        hidden_size=_get(ckpt_args, hyperparams, "hidden_size", config.HIDDEN_SIZE),
        num_layers=_get(ckpt_args, hyperparams, "num_layers", config.NUM_LAYERS),
        dropout=_get(ckpt_args, hyperparams, "dropout", config.DROPOUT),
        horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        num_targets=num_targets,
    )


def build_cnn_bilstm(checkpoint, model_type):
    from core.architectures.cnn_bilstm import CnnBiLSTM

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    conv1_out_ch = _get(ckpt_args, hyperparams, "conv1_out_ch", 32)
    conv2_out_ch = _get(ckpt_args, hyperparams, "conv2_out_ch", 64)
    h0 = _get(ckpt_args, hyperparams, "bilstm_hidden0", 32)

    return CnnBiLSTM(
        in_channels=_input_size(checkpoint, ckpt_args, hyperparams),
        input_len=_get(ckpt_args, hyperparams, "input_len", config.WINDOW_SIZE),
        pred_horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        kernel_sizes=(2, 4, 8),
        conv1_out_ch=conv1_out_ch,
        conv2_out_ch=conv2_out_ch,
        bilstm_hidden=(h0, h0 * 2, h0 * 4),
        num_targets=num_targets,
    )


def build_dpam(checkpoint, model_type):
    from core.architectures.waveanchor_dualmixer import (
        DualPathAnchorMixer,
        N_GROUP_BLOCKS,
        D_GROUP,
        MEM_D_GROUP,
        POOL_HEAD_DIM,
    )

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    swt_level = _get(ckpt_args, hyperparams, "swt_level", None)
    mem_swt_level = _get(ckpt_args, hyperparams, "mem_swt_level", None)
    if swt_level is None or mem_swt_level is None:
        from preprocessing.swt.config import CFG as SWT_CFG
        if swt_level is None:
            swt_level = SWT_CFG.SWT_LEVEL
        if mem_swt_level is None:
            mem_swt_level = SWT_CFG.MEM_SWT_LEVEL
    kernels = _get(ckpt_args, hyperparams, "dpam_cnn_kernels", None)
    if kernels is None:
        kernels = (3, 5)
    n_group_blocks = _get(ckpt_args, hyperparams, "dpam_group_blocks", None)
    if n_group_blocks is None:
        n_group_blocks = N_GROUP_BLOCKS
    d_group = _get(ckpt_args, hyperparams, "dpam_d_group", None)
    if d_group is None:
        d_group = D_GROUP
    mem_d_group = _get(ckpt_args, hyperparams, "dpam_mem_d_group", None)
    if mem_d_group is None:
        mem_d_group = MEM_D_GROUP
    pool_head_dim = _get(ckpt_args, hyperparams, "dpam_pool_head_dim", None)
    if pool_head_dim is None:
        pool_head_dim = POOL_HEAD_DIM
    cpu_recon = _get(ckpt_args, hyperparams, "dpam_cpu_recon", None)
    if cpu_recon is None:
        cpu_recon = True

    return DualPathAnchorMixer(
        in_channels=_input_size(checkpoint, ckpt_args, hyperparams),
        input_len=_get(ckpt_args, hyperparams, "input_len", config.WINDOW_SIZE),
        pred_horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        cpu_channels=swt_level + 1,
        mem_channels=mem_swt_level + 1,
        dropout=_get(ckpt_args, hyperparams, "dropout", config.DROPOUT),
        num_targets=num_targets,
        group_cnn_kernels=tuple(kernels),
        n_group_blocks=n_group_blocks,
        d_group=d_group,
        mem_d_group=mem_d_group,
        pool_head_dim=pool_head_dim,
        cpu_recon=cpu_recon,
    )


def build_tcn(checkpoint, model_type):
    from core.architectures.tcn import TCNForecaster

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    num_channels = _get(ckpt_args, hyperparams, "num_channels", [64, 64, 64, 64])
    kernel_size = _get(ckpt_args, hyperparams, "kernel_size", 3)
    dropout = _get(ckpt_args, hyperparams, "dropout", config.DROPOUT)

    return TCNForecaster(
        input_size=_input_size(checkpoint, ckpt_args, hyperparams),
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        num_targets=num_targets,
    )


def build_tcn_dual(checkpoint, model_type):
    from core.architectures.tcn import DualPathTCN

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    num_channels = _get(ckpt_args, hyperparams, "num_channels", [128, 128, 128, 128])
    kernel_size = _get(ckpt_args, hyperparams, "kernel_size", 3)
    dropout = _get(ckpt_args, hyperparams, "dropout", config.DROPOUT)

    return DualPathTCN(
        input_size=_input_size(checkpoint, ckpt_args, hyperparams),
        cpu_channels=6,  # SWT level 5 => 6 channels
        mem_channels=6,  # SWT level 5 => 6 channels
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        num_targets=num_targets,
    )


def build_quantile_ensemble(checkpoint, model_type):
    from core.architectures.ensemble import QuantileEnsembleForecaster

    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)

    return QuantileEnsembleForecaster(
        input_size=_input_size(checkpoint, ckpt_args, hyperparams),
        hidden_size=_get(ckpt_args, hyperparams, "hidden_size", config.HIDDEN_SIZE),
        num_layers=_get(ckpt_args, hyperparams, "num_layers", config.NUM_LAYERS),
        dropout=_get(ckpt_args, hyperparams, "dropout", config.DROPOUT),
        horizon=_get(ckpt_args, hyperparams, "pred_horizon", config.HORIZON),
        num_targets=num_targets,
        quantiles=_get(ckpt_args, hyperparams, "quantiles", [0.10, 0.50, 0.95]),
        ensemble_size=_get(ckpt_args, hyperparams, "ensemble_size", 5),
    )


def load_cqr_calibrators(checkpoint) -> dict:
    """Load CQR calibrators from checkpoint if available."""
    cqr_calibrators = checkpoint.get("cqr_calibrators")
    if cqr_calibrators:
        # Convert to simple dict with q_conf per target
        return {t_idx: cal.get("q_conf", 0.0) for t_idx, cal in cqr_calibrators.items()}
    return {}


BUILDERS = {
    "lstm": build_rnn,
    "gru": build_rnn,
    "bilstm": build_rnn,
    "bigrue": build_rnn,
    "cnn_bilstm": build_cnn_bilstm,
    "dpam": build_dpam,
    "tcn": build_tcn,
    "tcn_dual": build_tcn_dual,
    "quantile_ensemble": build_quantile_ensemble,
}


def build_model(checkpoint, model_type):
    builder = BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(
            f"Unknown model_type='{model_type}'. Available: {sorted(BUILDERS.keys())}"
        )
    model = builder(checkpoint, model_type)
    ckpt_args = checkpoint.get("args", {}) or {}
    if ckpt_args.get("change_head", False) or ckpt_args.get("change_head_mem", False):
        from core.models import ChangeHeadForecaster
        num_targets = ckpt_args.get("num_targets", config.NUM_TARGETS)
        inject_mask = None
        if ckpt_args.get("change_head_mem", False):
            inject_mask = [False] * num_targets
            if num_targets > 1:
                inject_mask[-1] = True
            else:
                inject_mask[0] = True
        model = ChangeHeadForecaster(model, inject_mask)
    return model

SEARCH_SPACE = [
    {"name": "dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from preprocessing.swt.config import CFG as SWT_CFG
    from core.architectures.waveanchor_dualmixer import (
        DualPathAnchorMixer, N_GROUP_BLOCKS, D_GROUP, MEM_D_GROUP, POOL_HEAD_DIM,
    )
    swt_level = getattr(args, "swt_level", SWT_CFG.SWT_LEVEL)
    mem_swt_level = getattr(args, "mem_swt_level", SWT_CFG.MEM_SWT_LEVEL)
    kernels = getattr(args, "dpam_cnn_kernels", None)
    if kernels is None:
        kernels = (3, 5)
    n_group_blocks = getattr(args, "dpam_group_blocks", None)
    if n_group_blocks is None:
        n_group_blocks = N_GROUP_BLOCKS
    d_group = getattr(args, "dpam_d_group", None)
    if d_group is None:
        d_group = D_GROUP
    mem_d_group = getattr(args, "dpam_mem_d_group", None)
    if mem_d_group is None:
        mem_d_group = MEM_D_GROUP
    pool_head_dim = getattr(args, "dpam_pool_head_dim", None)
    if pool_head_dim is None:
        pool_head_dim = POOL_HEAD_DIM
    cpu_recon = getattr(args, "dpam_cpu_recon", None)
    if cpu_recon is None:
        cpu_recon = True
    if input_size <= 2:
        cpu_channels, mem_channels = 1, input_size - 1
    else:
        cpu_channels = swt_level + 1
        mem_channels = mem_swt_level + 1
    mem_disable_drift = getattr(args, "dpam_mem_disable_drift", True)
    return DualPathAnchorMixer(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_channels=cpu_channels,
        mem_channels=mem_channels,
        dropout=hyperparams.get("dropout", 0.1),
        num_targets=num_targets,
        group_cnn_kernels=tuple(kernels),
        n_group_blocks=n_group_blocks,
        d_group=d_group,
        mem_d_group=mem_d_group,
        pool_head_dim=pool_head_dim,
        cpu_recon=cpu_recon,
        mem_disable_drift=mem_disable_drift,
    ).to(device)

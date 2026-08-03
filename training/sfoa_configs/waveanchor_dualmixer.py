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
    from core.architectures.waveanchor_dualmixer import (
        WaveAnchorDualMixer, N_GROUP_BLOCKS, D_GROUP, POOL_HEAD_DIM,
    )
    swt_level = getattr(args, "swt_level", SV_CFG.SWT_LEVEL)
    mem_swt_level = getattr(args, "mem_swt_level", SV_CFG.MEM_SWT_LEVEL)
    kernels = getattr(args, "wadm_cnn_kernels", None)
    if kernels is None:
        kernels = (3, 5)
    n_group_blocks = getattr(args, "wadm_group_blocks", None)
    if n_group_blocks is None:
        n_group_blocks = N_GROUP_BLOCKS
    d_group = getattr(args, "wadm_d_group", None)
    if d_group is None:
        d_group = D_GROUP
    pool_head_dim = getattr(args, "wadm_pool_head_dim", None)
    if pool_head_dim is None:
        pool_head_dim = POOL_HEAD_DIM
    cpu_anchor = getattr(args, "wadm_cpu_anchor", None) or "trend"
    mem_anchor = getattr(args, "wadm_mem_anchor", None) or "trend"
    mem_gate_init = None if getattr(args, "wadm_no_mem_gate", False) else 0.01
    grouping = getattr(args, "wadm_grouping", None) or "default"
    return WaveAnchorDualMixer(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_channels=swt_level + 1,
        mem_channels=mem_swt_level + 1,
        dropout=hyperparams.get("dropout", 0.1),
        num_targets=num_targets,
        group_cnn_kernels=tuple(kernels),
        n_group_blocks=n_group_blocks,
        d_group=d_group,
        pool_head_dim=pool_head_dim,
        cpu_anchor_mode=cpu_anchor,
        mem_anchor_mode=mem_anchor,
        mem_residual_gate_init=mem_gate_init,
        grouping=grouping,
    ).to(device)

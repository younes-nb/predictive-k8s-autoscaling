SEARCH_SPACE = [
    {"name": "tcn_dropout", "type": "continuous", "low": 0.1, "high": 0.5},
    {"name": "lr", "type": "log", "low": 5e-4, "high": 5e-3},
]

DEFAULTS = {
    "tcn_dropout": 0.1,
    "lr": 1e-3,
}


def build_model(hyperparams, input_size, args, num_targets, device):
    from core.architectures.tcn import DualPathwayTcn
    from preprocessing.sv.config import CFG as SV_CFG

    cpu_n_vmd = SV_CFG.VMD_K
    cpu_n_swt = SV_CFG.SWT_LEVEL
    mem_n_vmd = SV_CFG.VMD_K if num_targets > 1 else 0
    mem_n_swt = SV_CFG.MEM_SWT_LEVEL if num_targets > 1 else 0

    expected = cpu_n_vmd + cpu_n_swt + mem_n_vmd + mem_n_swt
    if input_size != expected:
        raise ValueError(
            f"DualPathwayTcn with sv expects input_size={expected} "
            f"(cpu_vmd={cpu_n_vmd}+cpu_swt={cpu_n_swt}"
            f"+mem_vmd={mem_n_vmd}+mem_swt={mem_n_swt}), got {input_size}"
        )

    return DualPathwayTcn(
        in_channels=input_size,
        input_len=args.input_len,
        pred_horizon=args.pred_horizon,
        cpu_n_vmd=cpu_n_vmd,
        cpu_n_swt=cpu_n_swt,
        mem_n_vmd=mem_n_vmd,
        mem_n_swt=mem_n_swt,
        dropout=hyperparams["tcn_dropout"],
        num_targets=num_targets,
    ).to(device)

import warnings
from typing import Optional

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")

D_GROUP = 64
MEM_D_GROUP = 24
N_GROUP_BLOCKS = 2
POOL_HEAD_DIM = 128
TREND_HIDDEN = 32


class GroupCNN(nn.Module):

    def __init__(self, in_ch, d_out, kernels=(3, 5), dropout=0.1):
        super().__init__()
        per_ch = d_out // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, per_ch, k, padding="same"),
                nn.GELU(),
            )
            for k in kernels
        ])
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        xc = x.permute(0, 2, 1)
        h = torch.cat([conv(xc) for conv in self.convs], dim=1)
        return self.norm(h.permute(0, 2, 1))


class MixerBlock(nn.Module):

    def __init__(self, T, C, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.temporal_mix = nn.Linear(T, T)
        self.norm2 = nn.LayerNorm(C)
        self.feature_mix = nn.Linear(C, C)

    def forward(self, x):
        h = self.norm1(x)
        h = h.permute(0, 2, 1)
        h = self.temporal_mix(h)
        h = h.permute(0, 2, 1)
        x = x + h

        h = self.norm2(x)
        h = self.feature_mix(h)
        x = x + h
        return x


class TrendExtrapolator(nn.Module):

    def __init__(self, input_len: int, pred_horizon: int, d_hidden: int = TREND_HIDDEN, dropout: float = 0.0):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.drift = nn.Sequential(
            nn.Linear(input_len, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, pred_horizon),
        )
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)

    def forward(self, a_l):
        base = a_l[:, -1:].expand(-1, self.pred_horizon)
        return base + self.drift(a_l)


class AnchorMixer(nn.Module):

    def __init__(
        self,
        in_channels: int = 6,
        input_len: int = 64,
        pred_horizon: int = 5,
        d_group: int = D_GROUP,
        dropout: float = 0.1,
        residual_gate_init: Optional[float] = None,
        group_cnn_kernels=(3, 5),
        n_group_blocks: int = N_GROUP_BLOCKS,
        pool_head_dim: int = POOL_HEAD_DIM,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self._osc_idx = torch.tensor(list(range(1, in_channels)), dtype=torch.long)

        self.cnn_osc = GroupCNN(in_channels - 1, d_group, kernels=group_cnn_kernels, dropout=dropout)
        self.group_mix_osc = nn.ModuleList([
            MixerBlock(input_len, d_group, dropout)
            for _ in range(n_group_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_group)

        self.trend_extrapolator = TrendExtrapolator(input_len, pred_horizon, dropout=dropout)

        self.residual_gate = None
        if residual_gate_init is not None:
            self.residual_gate = nn.Parameter(
                torch.full((pred_horizon,), float(residual_gate_init))
            )

        self.attn_proj = nn.Linear(d_group, 1)
        self.pool_head = nn.Sequential(
            nn.Linear(d_group + pred_horizon, pool_head_dim),
            nn.GELU(),
            nn.Linear(pool_head_dim, pred_horizon),
        )
        nn.init.zeros_(self.pool_head[2].weight)
        nn.init.zeros_(self.pool_head[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        dev = x.device
        osc = self._osc_idx.to(dev)

        base_trend = self.trend_extrapolator(x[:, :, 0])

        h = self.cnn_osc(x[:, :, osc])
        for blk in self.group_mix_osc:
            h = blk(h)
        h = self.norm_out(h)

        logits = self.attn_proj(h).squeeze(-1)
        w = torch.softmax(logits, dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        head_in = torch.cat([pooled, base_trend], dim=1)
        delta = self.pool_head(head_in)

        out = base_trend + delta
        if self.residual_gate is not None:
            out = base_trend + delta * self.residual_gate

        return out


class WaveAnchorDualMixer(nn.Module):

    def __init__(
        self,
        in_channels: int = 12,
        input_len: int = 64,
        pred_horizon: int = 5,
        cpu_channels: int = 6,
        mem_channels: int = 6,
        dropout: float = 0.1,
        num_targets: int = 1,
        group_cnn_kernels=(3, 5),
        n_group_blocks: int = N_GROUP_BLOCKS,
        d_group: int = D_GROUP,
        mem_d_group: int = MEM_D_GROUP,
        pool_head_dim: int = POOL_HEAD_DIM,
        mem_residual_gate_init: Optional[float] = 0.01,
    ):
        super().__init__()
        assert in_channels == cpu_channels + mem_channels
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.cpu_channels = cpu_channels
        self.mem_channels = mem_channels

        self.cpu_mixer = AnchorMixer(
            in_channels=cpu_channels,
            input_len=input_len,
            pred_horizon=pred_horizon,
            d_group=d_group,
            dropout=dropout,
            group_cnn_kernels=group_cnn_kernels,
            n_group_blocks=n_group_blocks,
            pool_head_dim=pool_head_dim,
        )

        self.mem_mixer = AnchorMixer(
            in_channels=mem_channels,
            input_len=input_len,
            pred_horizon=pred_horizon,
            d_group=mem_d_group,
            dropout=dropout,
            residual_gate_init=mem_residual_gate_init,
            group_cnn_kernels=group_cnn_kernels,
            n_group_blocks=n_group_blocks,
            pool_head_dim=pool_head_dim,
        )

    def memory_gate(self):
        return self.mem_mixer.residual_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        xc = x.permute(0, 2, 1)
        x_cpu = xc[:, :self.cpu_channels, :]
        x_mem = xc[:, self.cpu_channels:, :]

        cpu_out = self.cpu_mixer(x_cpu.permute(0, 2, 1))
        mem_out = self.mem_mixer(x_mem.permute(0, 2, 1))

        return torch.stack([cpu_out, mem_out], dim=-1)

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")

D_GROUP = 64
MEM_D_GROUP = 24
D_TIME = 128
D_FEAT = 64
N_GROUP_BLOCKS = 2
N_CROSS_BLOCKS = 2
MIX_STYLE = "linear_fused"
ANCHOR_KIND = "linear"
SLOPE_SPAN = 8
HEAD_STYLE = "pooled_mlp"
POOL_HEAD_DIM = 128


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

    def __init__(self, T, C, d_time, d_feat, dropout, mix_style="gelu_bottleneck"):
        super().__init__()
        self.mix_style = mix_style
        self.norm1 = nn.LayerNorm(C)
        if mix_style == "gelu_bottleneck":
            self.temporal_mix = nn.Sequential(
                nn.Linear(T, d_time),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_time, T),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(C)
            self.feature_mix = nn.Sequential(
                nn.Linear(C, d_feat),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_feat, C),
                nn.Dropout(dropout),
            )
        elif mix_style == "linear_bottleneck":
            self.temporal_mix = nn.Sequential(
                nn.Linear(T, d_time),
                nn.Linear(d_time, T),
            )
            self.norm2 = nn.LayerNorm(C)
            self.feature_mix = nn.Sequential(
                nn.Linear(C, d_feat),
                nn.Linear(d_feat, C),
            )
        elif mix_style in ("linear_fused", "linear_fused_gelu"):
            self.temporal_mix = nn.Linear(T, T)
            self.norm2 = nn.LayerNorm(C)
            self.feature_mix = nn.Linear(C, C)
        else:
            raise ValueError(f"Unknown mix_style: {mix_style}")

    def forward(self, x):
        h = self.norm1(x)
        h = h.permute(0, 2, 1)
        h = self.temporal_mix(h)
        if self.mix_style == "linear_fused_gelu":
            h = torch.nn.functional.gelu(h)
        h = h.permute(0, 2, 1)
        x = x + h

        h = self.norm2(x)
        h = self.feature_mix(h)
        if self.mix_style == "linear_fused_gelu":
            h = torch.nn.functional.gelu(h)
        x = x + h
        return x


class AnchorMixer(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        input_len: int = 64,
        pred_horizon: int = 5,
        num_targets: int = 1,
        d_group: int = D_GROUP,
        dropout: float = 0.1,
        osc_channels=None,
        det_channels=None,
        trd_channels=None,
        anchor_mode: str = "trend",
        residual_gate_init: Optional[float] = None,
    ):
        super().__init__()
        T = input_len
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.slope_span = SLOPE_SPAN
        self.anchor_mode = anchor_mode

        osc = list(osc_channels if osc_channels is not None else [1, 2, 3])
        det = list(det_channels if det_channels is not None else [4])
        trd = list(trd_channels if trd_channels is not None else [0])
        self._osc_idx = torch.tensor(osc, dtype=torch.long)
        self._det_idx = torch.tensor(det, dtype=torch.long)
        self._trd_idx = torch.tensor(trd, dtype=torch.long)

        self.has_osc = len(osc) > 0
        self.has_det = len(det) > 0
        self.has_trd = len(trd) > 0

        self.cnn_osc = GroupCNN(len(osc), d_group, dropout=dropout) if self.has_osc else None
        self.cnn_det = GroupCNN(len(det), d_group, dropout=dropout) if self.has_det else None
        self.cnn_trd = GroupCNN(len(trd), d_group, dropout=dropout) if self.has_trd else None

        self.group_mix_osc = (
            nn.ModuleList([
                MixerBlock(T, d_group, D_TIME, D_FEAT, dropout, MIX_STYLE)
                for _ in range(N_GROUP_BLOCKS)
            ]) if self.has_osc else None
        )
        self.group_mix_det = (
            nn.ModuleList([
                MixerBlock(T, d_group, D_TIME, D_FEAT, dropout, MIX_STYLE)
                for _ in range(N_GROUP_BLOCKS)
            ]) if self.has_det else None
        )
        self.group_mix_trd = (
            nn.ModuleList([
                MixerBlock(T, d_group, D_TIME, D_FEAT, dropout, MIX_STYLE)
                for _ in range(N_GROUP_BLOCKS)
            ]) if self.has_trd else None
        )

        cross_c = d_group * (self.has_osc + self.has_det + self.has_trd)
        self.cross_blocks = nn.ModuleList([
            MixerBlock(T, cross_c, D_TIME, D_FEAT, dropout, MIX_STYLE)
            for _ in range(N_CROSS_BLOCKS)
        ])

        self.norm_out = nn.LayerNorm(cross_c)

        self.register_buffer(
            "trend_w", self._build_trend_w(input_len, in_channels, pred_horizon, anchor_mode)
        )

        self.residual_gate = None
        if residual_gate_init is not None:
            self.residual_gate = nn.Parameter(
                torch.full((pred_horizon,), float(residual_gate_init))
            )

        self.head_style = HEAD_STYLE
        if HEAD_STYLE == "linear":
            self.output_head = nn.Linear(input_len * cross_c, pred_horizon)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
        elif HEAD_STYLE == "pooled_mlp":
            self.pool_head = nn.Sequential(
                nn.Linear(2 * cross_c + pred_horizon, POOL_HEAD_DIM),
                nn.GELU(),
                nn.Linear(POOL_HEAD_DIM, pred_horizon),
            )
            nn.init.zeros_(self.pool_head[2].weight)
            nn.init.zeros_(self.pool_head[2].bias)
        else:
            raise ValueError(f"Unknown head_style: {HEAD_STYLE}")

    @staticmethod
    def _build_trend_w(
        input_len: int, n_bands: int, pred_horizon: int, anchor_mode: str = "trend"
    ) -> torch.Tensor:
        w = AnchorMixer._swt_reconstruction_weights(input_len, n_bands, position=-1)
        trend_w = np.zeros((n_bands * input_len, pred_horizon), dtype=np.float64)
        if anchor_mode == "persistence":
            for s in range(1, pred_horizon + 1):
                trend_w[:, s - 1] = w.reshape(-1)
            return torch.from_numpy(trend_w).float()
        w8 = AnchorMixer._swt_reconstruction_weights(input_len, n_bands, position=input_len - 1 - SLOPE_SPAN)
        if ANCHOR_KIND == "quadratic":
            w16 = AnchorMixer._swt_reconstruction_weights(input_len, n_bands, position=input_len - 1 - 2 * SLOPE_SPAN)
            c1 = (3.0 * w - 4.0 * w8 + w16) / (2.0 * SLOPE_SPAN)
            c2 = (w - 2.0 * w8 + w16) / (2.0 * SLOPE_SPAN * SLOPE_SPAN)
            for s in range(1, pred_horizon + 1):
                trend_w[:, s - 1] = (w + c1 * s + c2 * s * s).reshape(-1)
        else:
            for s in range(1, pred_horizon + 1):
                coef = w + (w - w8) * (s / SLOPE_SPAN)
                trend_w[:, s - 1] = coef.reshape(-1)
        return torch.from_numpy(trend_w).float()

    @staticmethod
    def _swt_reconstruction_weights(input_len: int, n_bands: int, position: int = -1) -> np.ndarray:
        import pywt
        w = np.zeros((n_bands, input_len), dtype=np.float64)
        for b in range(n_bands):
            for i in range(input_len):
                coeffs = [np.zeros(input_len) for _ in range(n_bands)]
                coeffs[b][i] = 1.0
                recon = pywt.iswt(coeffs, "sym4", norm=True)
                w[b, i] = recon[position]
        return w

    def _mix_group(self, x, blocks):
        for blk in blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        dev = x.device
        osc = self._osc_idx.to(dev)
        det = self._det_idx.to(dev)
        trd = self._trd_idx.to(dev)

        base_trend = x.permute(0, 2, 1).flatten(1, 2) @ self.trend_w

        h_parts = []
        if self.has_osc:
            h_parts.append(self._mix_group(self.cnn_osc(x[:, :, osc]), self.group_mix_osc))
        if self.has_det:
            h_parts.append(self._mix_group(self.cnn_det(x[:, :, det]), self.group_mix_det))
        if self.has_trd:
            h_parts.append(self._mix_group(self.cnn_trd(x[:, :, trd]), self.group_mix_trd))
        h = torch.cat(h_parts, dim=2)

        for blk in self.cross_blocks:
            h = blk(h)

        h = self.norm_out(h)

        if self.head_style == "linear":
            delta = self.output_head(h.flatten(1))
        else:
            delta = self.pool_head(torch.cat([h[:, -1, :], h.mean(dim=1), base_trend], dim=1))

        out = base_trend + delta
        if self.residual_gate is not None:
            out = base_trend + delta * self.residual_gate

        if self.num_targets > 1:
            return out.view(-1, self.pred_horizon, self.num_targets)
        return out


def _swt_groups(n_channels: int):
    if n_channels < 2:
        raise ValueError(
            f"SWT level 0 gives a single band; at least 2 channels "
            f"(A level + D1) are required, got {n_channels}"
        )
    if n_channels == 2:
        return [0], [], [1]
    return [0], list(range(1, n_channels - 1)), [n_channels - 1]


class WaveAnchorDualMixer(nn.Module):

    def __init__(
        self,
        in_channels: int = 8,
        input_len: int = 64,
        pred_horizon: int = 5,
        cpu_channels: int = 5,
        mem_channels: int = 3,
        dropout: float = 0.1,
        num_targets: int = 1,
        osc_channels=None,
        det_channels=None,
        trd_channels=None,
    ):
        super().__init__()
        assert in_channels == cpu_channels + mem_channels
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.cpu_channels = cpu_channels
        self.mem_channels = mem_channels

        if osc_channels is not None or det_channels is not None or trd_channels is not None:
            cpu_osc, cpu_det, cpu_trd = osc_channels, det_channels, trd_channels
        else:
            cpu_osc, cpu_det, cpu_trd = _swt_groups(cpu_channels)
        mem_osc, mem_det, mem_trd = _swt_groups(mem_channels)

        self.cpu_mixer = AnchorMixer(
            in_channels=cpu_channels,
            input_len=input_len,
            pred_horizon=pred_horizon,
            num_targets=1,
            d_group=D_GROUP,
            dropout=dropout,
            osc_channels=cpu_osc,
            det_channels=cpu_det,
            trd_channels=cpu_trd,
        )

        self.mem_mixer = AnchorMixer(
            in_channels=mem_channels,
            input_len=input_len,
            pred_horizon=pred_horizon,
            num_targets=1,
            d_group=MEM_D_GROUP,
            dropout=dropout,
            osc_channels=mem_osc,
            det_channels=mem_det,
            trd_channels=mem_trd,
            anchor_mode="persistence",
            residual_gate_init=0.01,
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

        out = torch.stack([cpu_out, mem_out], dim=-1)

        if self.num_targets > 1:
            return out
        return out

import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")


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
    """Temporal + feature mixing over a (B, T, C) block.

    ``mix_style`` controls the internal mixing stack (fewer modules == lower
    single-sample latency, since dispatch overhead dominates at batch 1):
    * ``gelu_bottleneck`` (default, legacy): Linear(T,d_time)+GELU+Dropout+
      Linear(d_time,T) for temporal mixing, mirror for feature mixing.
    * ``linear_bottleneck``: same widths, activations/dropout removed.
    * ``linear_fused``: single Linear(T,T) / Linear(C,C) per branch (canonical
      TSMixer mixing).
    * ``linear_fused_gelu``: same single-Linear mixing as ``linear_fused``
      with one GELU after each mix, restoring block nonlinearity for ~2
      extra module dispatches per block.
    """

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


class FreqTSMixer(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        input_len: int = 64,
        pred_horizon: int = 5,
        num_targets: int = 1,
        d_group: int = 64,
        n_group_blocks: int = 2,
        n_cross_blocks: int = 2,
        d_time: int = 128,
        d_feat: int = 64,
        dropout: float = 0.1,
        osc_channels=None,
        det_channels=None,
        trd_channels=None,
    ):
        super().__init__()
        T = input_len
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets

        osc = list(osc_channels if osc_channels is not None else [1, 2, 3])
        det = list(det_channels if det_channels is not None else [4])
        trd = list(trd_channels if trd_channels is not None else [0])
        self._osc_idx = torch.tensor(osc, dtype=torch.long)
        self._det_idx = torch.tensor(det, dtype=torch.long)
        self._trd_idx = torch.tensor(trd, dtype=torch.long)

        self.cnn_osc = GroupCNN(len(osc), d_group, dropout=dropout)
        self.cnn_det = GroupCNN(len(det), d_group, dropout=dropout)
        self.cnn_trd = GroupCNN(len(trd), d_group, dropout=dropout)

        self.group_mix_osc = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout)
            for _ in range(n_group_blocks)
        ])
        self.group_mix_det = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout)
            for _ in range(n_group_blocks)
        ])
        self.group_mix_trd = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout)
            for _ in range(n_group_blocks)
        ])

        cross_c = d_group * 3
        self.cross_blocks = nn.ModuleList([
            MixerBlock(T, cross_c, d_time, d_feat, dropout)
            for _ in range(n_cross_blocks)
        ])

        self.norm_out = nn.LayerNorm(cross_c)
        self.output_head = nn.Linear(T * cross_c, pred_horizon * num_targets)

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

        h_osc = self._mix_group(self.cnn_osc(x[:, :, osc]), self.group_mix_osc)
        h_det = self._mix_group(self.cnn_det(x[:, :, det]), self.group_mix_det)
        h_trd = self._mix_group(self.cnn_trd(x[:, :, trd]), self.group_mix_trd)

        h = torch.cat([h_osc, h_det, h_trd], dim=2)

        for blk in self.cross_blocks:
            h = blk(h)

        h = self.norm_out(h)
        out = self.output_head(h.flatten(1))

        if self.num_targets > 1:
            return out.view(-1, self.pred_horizon, self.num_targets)
        return out


class ResMixerCPU(nn.Module):
    """Frequency-grouped TSMixer CPU encoder with a trend-anchored head.

    Identical encoder to ``FreqTSMixer`` (per-group convs + per-group and
    cross MixerBlocks over the SWT band groups). The head differs:

    * ``base_trend`` is an extrapolation of the SWT reconstruction (last
      observed value + recent slope, optionally quadratic), computed from
      the raw bands via a single precomputed weight matrix; it catches
      rising windows that a persistence baseline misses;
    * a single Linear over the full flattened encoder output (the same
      kernel shape as the baseline head) learns the residual correction,
      zero-initialized so training starts exactly at the trend extension.

    Output = base_trend + delta with ``delta`` initialized to zero, so
    training starts at the trend-extension forecast and only learns
    deviations. The head keeps the baseline's single-Linear structure plus
    one small anchor matmul to minimize single-sample latency.
    """

    def __init__(
        self,
        in_channels: int = 5,
        input_len: int = 64,
        pred_horizon: int = 5,
        num_targets: int = 1,
        d_group: int = 64,
        n_group_blocks: int = 2,
        n_cross_blocks: int = 2,
        d_time: int = 128,
        d_feat: int = 64,
        dropout: float = 0.1,
        mix_style: str = "gelu_bottleneck",
        anchor_kind: str = "linear",
        slope_span: int = 8,
        head_style: str = "linear",
        pool_head_dim: int = 128,
        osc_channels=None,
        det_channels=None,
        trd_channels=None,
    ):
        super().__init__()
        T = input_len
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.slope_span = slope_span

        osc = list(osc_channels if osc_channels is not None else [1, 2, 3])
        det = list(det_channels if det_channels is not None else [4])
        trd = list(trd_channels if trd_channels is not None else [0])
        self._osc_idx = torch.tensor(osc, dtype=torch.long)
        self._det_idx = torch.tensor(det, dtype=torch.long)
        self._trd_idx = torch.tensor(trd, dtype=torch.long)

        self.cnn_osc = GroupCNN(len(osc), d_group, dropout=dropout)
        self.cnn_det = GroupCNN(len(det), d_group, dropout=dropout)
        self.cnn_trd = GroupCNN(len(trd), d_group, dropout=dropout)

        self.group_mix_osc = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout, mix_style)
            for _ in range(n_group_blocks)
        ])
        self.group_mix_det = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout, mix_style)
            for _ in range(n_group_blocks)
        ])
        self.group_mix_trd = nn.ModuleList([
            MixerBlock(T, d_group, d_time, d_feat, dropout, mix_style)
            for _ in range(n_group_blocks)
        ])

        cross_c = d_group * 3
        self.cross_blocks = nn.ModuleList([
            MixerBlock(T, cross_c, d_time, d_feat, dropout, mix_style)
            for _ in range(n_cross_blocks)
        ])

        self.norm_out = nn.LayerNorm(cross_c)

        self.register_buffer("trend_w", self._build_trend_w(input_len, in_channels, pred_horizon, anchor_kind, slope_span))

        self.head_style = head_style
        if head_style == "linear":
            self.output_head = nn.Linear(input_len * cross_c, pred_horizon)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
        elif head_style == "pooled_mlp":
            self.pool_head = nn.Sequential(
                nn.Linear(2 * cross_c + pred_horizon, pool_head_dim),
                nn.GELU(),
                nn.Linear(pool_head_dim, pred_horizon),
            )
            nn.init.zeros_(self.pool_head[2].weight)
            nn.init.zeros_(self.pool_head[2].bias)
        else:
            raise ValueError(f"Unknown head_style: {head_style}")

    @staticmethod
    def _build_trend_w(input_len: int, n_bands: int, pred_horizon: int,
                       anchor_kind: str, slope_span: int) -> torch.Tensor:
        w = ResMixerCPU._swt_reconstruction_weights(input_len, n_bands, position=-1)
        w8 = ResMixerCPU._swt_reconstruction_weights(input_len, n_bands, position=input_len - 1 - slope_span)
        trend_w = np.zeros((n_bands * input_len, pred_horizon), dtype=np.float64)
        if anchor_kind == "quadratic":
            w16 = ResMixerCPU._swt_reconstruction_weights(input_len, n_bands, position=input_len - 1 - 2 * slope_span)
            c1 = (3.0 * w - 4.0 * w8 + w16) / (2.0 * slope_span)
            c2 = (w - 2.0 * w8 + w16) / (2.0 * slope_span * slope_span)
            for s in range(1, pred_horizon + 1):
                trend_w[:, s - 1] = (w + c1 * s + c2 * s * s).reshape(-1)
        else:
            for s in range(1, pred_horizon + 1):
                coef = w + (w - w8) * (s / slope_span)
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

        h_osc = self._mix_group(self.cnn_osc(x[:, :, osc]), self.group_mix_osc)
        h_det = self._mix_group(self.cnn_det(x[:, :, det]), self.group_mix_det)
        h_trd = self._mix_group(self.cnn_trd(x[:, :, trd]), self.group_mix_trd)

        h = torch.cat([h_osc, h_det, h_trd], dim=2)

        for blk in self.cross_blocks:
            h = blk(h)

        h = self.norm_out(h)

        if self.head_style == "linear":
            delta = self.output_head(h.flatten(1))
        else:
            delta = self.pool_head(torch.cat([h[:, -1, :], h.mean(dim=1), base_trend], dim=1))

        out = base_trend + delta

        if self.num_targets > 1:
            return out.view(-1, self.pred_horizon, self.num_targets)
        return out


class CnnBiLSTM_DualPath(nn.Module):
    """Dual-path CPU/memory forecaster.

    CPU: frequency-grouped TSMixer encoder (osc/det/trd SWT bands) with a
    trend-anchored head regressing future CPU values.
    Memory: SWT-persistence baseline (reconstruction of the last value)
    plus a small learned MLP that corrects the per-step prediction from
    window statistics (deviation from trend, slopes, detail energy, level).
    The correction is trained end-to-end with per-target L1 loss.
    """

    def __init__(
        self,
        in_channels: int = 8,
        input_len: int = 64,
        pred_horizon: int = 5,
        cpu_channels: int = 5,
        mem_channels: int = 3,
        d_group: int = 64,
        n_group_blocks: int = 2,
        n_cross_blocks: int = 2,
        d_time: int = 128,
        d_feat: int = 64,
        dropout: float = 0.1,
        mix_style: str = "gelu_bottleneck",
        anchor_kind: str = "linear",
        slope_span: int = 8,
        head_style: str = "linear",
        pool_head_dim: int = 128,
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

        w = self._swt_reconstruction_weights(input_len, mem_channels)
        self.register_buffer("mem_recon_w", torch.from_numpy(w).float())

        self.cpu_mixer = ResMixerCPU(
            in_channels=cpu_channels,
            input_len=input_len,
            pred_horizon=pred_horizon,
            num_targets=1,
            d_group=d_group,
            n_group_blocks=n_group_blocks,
            n_cross_blocks=n_cross_blocks,
            d_time=d_time,
            d_feat=d_feat,
            dropout=dropout,
            mix_style=mix_style,
            anchor_kind=anchor_kind,
            slope_span=slope_span,
            head_style=head_style,
            pool_head_dim=pool_head_dim,
            osc_channels=osc_channels,
            det_channels=det_channels,
            trd_channels=trd_channels,
        )

        self.mem_bias = nn.Parameter(torch.zeros(pred_horizon))
        self.mem_nn_head = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, pred_horizon),
        )
        nn.init.zeros_(self.mem_nn_head[-1].weight)
        nn.init.zeros_(self.mem_nn_head[-1].bias)

    @staticmethod
    def _swt_reconstruction_weights(input_len: int, n_bands: int) -> np.ndarray:
        import pywt
        w = np.zeros((n_bands, input_len), dtype=np.float64)
        for b in range(n_bands):
            for i in range(input_len):
                coeffs = [np.zeros(input_len) for _ in range(n_bands)]
                coeffs[b][i] = 1.0
                recon = pywt.iswt(coeffs, "sym4", norm=True)
                w[b, i] = recon[-1]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        xc = x.permute(0, 2, 1)
        x_cpu = xc[:, :self.cpu_channels, :]
        x_mem = xc[:, self.cpu_channels:, :]

        cpu_out = self.cpu_mixer(x_cpu.permute(0, 2, 1))

        mem_bands = x_mem
        naive_mem = (mem_bands * self.mem_recon_w.unsqueeze(0)).sum(dim=(1, 2))
        rawdev = naive_mem - mem_bands[:, 0, -1]
        a2_last = mem_bands[:, 0, -1]
        a2_slope5 = a2_last - mem_bands[:, 0, -6]
        a2_slope20 = a2_last - mem_bands[:, 0, -21]
        d2_energy = (mem_bands[:, 2, -6:] ** 2).mean(dim=1)
        feats = torch.stack(
            [rawdev, rawdev ** 2, a2_slope5, a2_slope20, d2_energy, naive_mem],
            dim=1,
        )
        mem_out = naive_mem.unsqueeze(1) + self.mem_bias + self.mem_nn_head(feats)

        out = torch.stack([cpu_out, mem_out], dim=-1)

        if self.num_targets > 1:
            return out
        return out

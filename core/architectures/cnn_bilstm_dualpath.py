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

    def __init__(self, T, C, d_time, d_feat, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
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


class CnnBiLSTM_DualPath(nn.Module):
    """Dual-path CPU/memory forecaster.

    CPU: frequency-grouped TSMixer encoder (osc/det/trd SWT bands) regressing
    future CPU values directly.
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

        import pywt
        w = self._swt_reconstruction_weights(input_len, mem_channels)
        self.register_buffer("mem_recon_w", torch.from_numpy(w).float())

        self.cpu_mixer = FreqTSMixer(
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

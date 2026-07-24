import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")


OSC_CHANNELS = list(range(0, 9)) + list(range(13, 22))
DET_CHANNELS = [9, 10, 11, 22]
TRD_CHANNELS = [12, 23]


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
        in_channels: int = 24,
        input_len: int = 64,
        pred_horizon: int = 5,
        num_targets: int = 2,
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

        osc = list(osc_channels or OSC_CHANNELS)
        det = list(det_channels or DET_CHANNELS)
        trd = list(trd_channels or TRD_CHANNELS)
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

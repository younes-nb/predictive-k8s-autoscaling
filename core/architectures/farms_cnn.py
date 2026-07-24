import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")


OSC_CHANNELS = list(range(0, 9)) + list(range(13, 22))
DET_CHANNELS = [9, 10, 11, 22]
TRD_CHANNELS = [12, 23]


class SEGate(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.permute(0, 2, 1))
        return x * w.unsqueeze(1)


class ResidualMultiscaleCNN(nn.Module):

    def __init__(self, in_ch, d_msc, kernels=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, d_msc, k, padding="same")
            for k in kernels
        ])
        out_ch = d_msc * len(kernels)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        xc = x.permute(0, 2, 1)
        branch_out = torch.cat([b(xc) for b in self.branches], dim=1)
        branch_out = branch_out.permute(0, 2, 1)
        res = self.shortcut(xc).permute(0, 2, 1)
        h = self.norm(branch_out + res)
        h = torch.nn.functional.gelu(h)
        return self.dropout(h)


class ResBlock(nn.Module):

    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding="same")
        self.conv2 = nn.Conv1d(ch, ch, 3, padding="same")
        self.norm = nn.LayerNorm(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        xc = x.permute(0, 2, 1)
        h = torch.nn.functional.gelu(self.conv1(xc))
        h = self.conv2(h)
        h = h.permute(0, 2, 1)
        h = self.norm(h + x)
        return self.dropout(torch.nn.functional.gelu(h))


class FARMS_CNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 24,
        input_len: int = 64,
        pred_horizon: int = 5,
        num_targets: int = 2,
        osc_out: int = 32,
        det_out: int = 16,
        trd_out: int = 8,
        d_msc: int = 32,
        se_reduction: int = 4,
        n_res_blocks: int = 1,
        dropout: float = 0.1,
        osc_channels=None,
        det_channels=None,
        trd_channels=None,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets

        osc = list(osc_channels or OSC_CHANNELS)
        det = list(det_channels or DET_CHANNELS)
        trd = list(trd_channels or TRD_CHANNELS)
        self._osc_idx = torch.tensor(osc, dtype=torch.long)
        self._det_idx = torch.tensor(det, dtype=torch.long)
        self._trd_idx = torch.tensor(trd, dtype=torch.long)

        self.se_osc = SEGate(len(osc), se_reduction)
        self.se_det = SEGate(len(det), se_reduction)
        self.se_trd = SEGate(len(trd), se_reduction)

        self.reduce_osc = nn.Conv1d(len(osc), osc_out, 1)
        self.reduce_det = nn.Conv1d(len(det), det_out, 1)
        self.reduce_trd = nn.Conv1d(len(trd), trd_out, 1)

        concat_ch = osc_out + det_out + trd_out

        self.msc = ResidualMultiscaleCNN(concat_ch, d_msc, dropout=dropout)
        msc_out = d_msc * 3

        self.res_blocks = nn.Sequential(*[
            ResBlock(msc_out, dropout) for _ in range(n_res_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(msc_out),
            nn.Linear(msc_out, msc_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(msc_out, pred_horizon * num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        dev = x.device
        osc = self._osc_idx.to(dev)
        det = self._det_idx.to(dev)
        trd = self._trd_idx.to(dev)

        h_osc = self.se_osc(x[:, :, osc])
        h_det = self.se_det(x[:, :, det])
        h_trd = self.se_trd(x[:, :, trd])

        h_osc = self.reduce_osc(h_osc.permute(0, 2, 1)).permute(0, 2, 1)
        h_det = self.reduce_det(h_det.permute(0, 2, 1)).permute(0, 2, 1)
        h_trd = self.reduce_trd(h_trd.permute(0, 2, 1)).permute(0, 2, 1)

        h = torch.cat([h_osc, h_det, h_trd], dim=2)

        h = self.msc(h)
        h = self.res_blocks(h)

        h = h.mean(dim=1)
        out = self.head(h)

        if self.num_targets > 1:
            return out.view(-1, self.pred_horizon, self.num_targets)
        return out

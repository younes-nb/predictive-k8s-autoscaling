import os
import sys

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return self.relu(out + self.skip(x))


class TCNBranch(nn.Module):
    def __init__(self, in_channels, filters, dilations, kernel_size, dropout):
        super().__init__()
        n_blocks = len(dilations)
        filters_expanded = list(filters)
        while len(filters_expanded) < n_blocks:
            filters_expanded.append(filters_expanded[-1])

        self.blocks = nn.ModuleList()
        prev_out = in_channels
        for i in range(n_blocks):
            out_ch = filters_expanded[i]
            self.blocks.append(
                TCNResidualBlock(prev_out, out_ch, kernel_size, dilations[i], dropout)
            )
            prev_out = out_ch

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]


class BiGRUBranch(nn.Module):
    def __init__(self, in_channels, hidden_sizes):
        super().__init__()
        self.gru1 = nn.GRU(in_channels, hidden_sizes[0], batch_first=True,
                          bidirectional=True)
        self.gru2 = nn.GRU(hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        return out2[:, -1, :]


def init_weights(model, cfg):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.uniform_(m.weight, cfg.WEIGHT_INIT_LOW, cfg.WEIGHT_INIT_HIGH)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.uniform_(param, cfg.WEIGHT_INIT_LOW, cfg.WEIGHT_INIT_HIGH)


class CshvtbModel(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.cfg = cfg

        self.tcn = TCNBranch(
            in_channels=in_channels,
            filters=cfg.TCN_FILTERS,
            dilations=cfg.TCN_DILATIONS,
            kernel_size=cfg.TCN_KERNEL_SIZE,
            dropout=cfg.TCN_DROPOUT,
        )

        self.bigru = BiGRUBranch(
            in_channels=in_channels,
            hidden_sizes=cfg.BIGRU_HIDDEN,
        )

        self.fc = nn.Linear(256 + 256, cfg.PRED_HORIZON)

        init_weights(self, cfg)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == self.cfg.INPUT_LEN:
            x_tcn = x.permute(0, 2, 1)
        else:
            x_tcn = x

        tcn_out = self.tcn(x_tcn)
        bigru_out = self.bigru(x)

        combined = torch.cat([tcn_out, bigru_out], dim=1)
        output = self.fc(combined)

        return output


if __name__ == "__main__":
    from experiments.cshvtb.config import CFG

    model = CshvtbModel(in_channels=6, cfg=CFG)
    x = torch.randn(4, 60, 6)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Model parameters: {total_params}")
    print("CshvtbModel smoke test passed")

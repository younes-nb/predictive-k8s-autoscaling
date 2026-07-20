import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")


class CnnBiLSTM(nn.Module):

    def __init__(
        self,
        in_channels: int = 7,
        input_len: int = 60,
        pred_horizon: int = 5,
        kernel_sizes: tuple = (2, 4, 8),
        conv1_out_ch: int = 32,
        conv2_out_ch: int = 64,
        bilstm_hidden: tuple = (32, 64, 128),
        num_targets: int = 1,
    ):
        super().__init__()
        del input_len
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        K = len(kernel_sizes)

        self.conv_set1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, conv1_out_ch, ks, padding="same"),
                nn.ReLU(),
            )
            for ks in kernel_sizes
        ])

        in_ch2 = K * conv1_out_ch
        self.conv_set2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch2, conv2_out_ch, ks, padding="same"),
                nn.ReLU(),
            )
            for ks in kernel_sizes
        ])

        lstm_in = K * conv2_out_ch
        h = bilstm_hidden
        self.bilstm1 = nn.LSTM(lstm_in, h[0], batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(h[0] * 2, h[1], batch_first=True, bidirectional=True)
        self.bilstm3 = nn.LSTM(h[1] * 2, h[2], batch_first=True, bidirectional=True)

        self.fc = nn.Linear(h[2] * 2, pred_horizon * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        xc = x.permute(0, 2, 1)

        out1 = torch.cat([conv(xc) for conv in self.conv_set1], dim=1)
        out2 = torch.cat([conv(out1) for conv in self.conv_set2], dim=1)

        seq = out2.permute(0, 2, 1)

        o1, _ = self.bilstm1(seq)
        o2, _ = self.bilstm2(o1)
        o3, _ = self.bilstm3(o2)

        last = o3[:, -1, :]
        out = self.fc(last)
        if self.num_targets > 1:
            return out.view(-1, self.pred_horizon, self.num_targets)
        return out

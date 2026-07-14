import torch
import torch.nn as nn
from torch.nn.utils import parametrizations


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )
        self.downsample = parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, 1)) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TcnBiGru(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_len: int,
        pred_horizon: int,
        tcn_kernel_size: int = 3,
        tcn_filters: tuple = (256, 256),
        tcn_dilations: tuple = (1, 2, 4),
        tcn_dropout: float = 0.2,
        bigru_hidden: tuple = (64, 128),
    ):
        super().__init__()

        tcn_layers = []
        tcn_in = in_channels
        for i, filters in enumerate(tcn_filters):
            d = tcn_dilations[i] if i < len(tcn_dilations) else tcn_dilations[-1]
            tcn_layers.append(TemporalBlock(tcn_in, filters, tcn_kernel_size, d, tcn_dropout))
            tcn_in = filters
        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_output_len = input_len

        for m in self.tcn.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.uniform_(m.weight, a=0.0, b=0.0015)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        if len(bigru_hidden) == 1:
            self.bigru = nn.GRU(
                in_channels, bigru_hidden[0],
                batch_first=True, bidirectional=True,
            )
            gru_out = bigru_hidden[0] * 2
        else:
            self.bigru = nn.GRU(
                in_channels, bigru_hidden[0],
                batch_first=True, bidirectional=True,
                num_layers=2, dropout=tcn_dropout if len(bigru_hidden) > 2 else 0,
            )
            self.bigru2 = nn.GRU(
                bigru_hidden[0] * 2, bigru_hidden[1] if len(bigru_hidden) > 1 else bigru_hidden[0],
                batch_first=True, bidirectional=True,
            )
            gru_out = (bigru_hidden[1] if len(bigru_hidden) > 1 else bigru_hidden[0]) * 2

        self.fc = nn.Linear(gru_out + tcn_filters[-1], pred_horizon)

    def forward(self, x):
        tcn_feat = self.tcn(x.transpose(1, 2))[:, :, -1]

        if hasattr(self, "bigru2"):
            o1, _ = self.bigru(x)
            o2, _ = self.bigru2(o1)
            gru_feat = o2[:, -1, :]
        else:
            o, _ = self.bigru(x)
            gru_feat = o[:, -1, :]

        return self.fc(torch.cat([tcn_feat, gru_feat], dim=1))

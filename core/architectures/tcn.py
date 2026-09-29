import torch
import torch.nn as nn


class CausalBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=0, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        xp = nn.functional.pad(x, (self.pad, 0))
        out = self.drop(self.relu(self.conv(xp)))
        return out + x


class TCNDualHead(nn.Module):

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        horizon: int = 5,
        num_targets: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.in_proj = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden_size, kernel_size, dilation=2 ** i,
                        dropout=dropout)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon * num_targets)
        self.event_fc = nn.Linear(hidden_size, horizon * 2)
    def _embed(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.in_proj(x.permute(0, 2, 1))
        for blk in self.blocks:
            h = blk(h)
        return self.dropout_layer(h.permute(0, 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = self._embed(x)[:, -1, :]
        out = self.fc(last)
        if self.num_targets > 1:
            return out.view(out.size(0), self.horizon, self.num_targets)
        return out

    def event_logits(self, x: torch.Tensor) -> torch.Tensor:
        last = self._embed(x)[:, -1, :]
        return self.event_fc(last).view(last.size(0), self.horizon, 2)


class TCNForecaster(nn.Module):

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        horizon: int = 5,
        num_targets: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.in_proj = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden_size, kernel_size, dilation=2 ** i,
                        dropout=dropout)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.in_proj(x.permute(0, 2, 1))
        for blk in self.blocks:
            h = blk(h)
        last = self.dropout_layer(h[:, :, -1])
        out = self.fc(last)
        if self.num_targets > 1:
            return out.view(out.size(0), self.horizon, self.num_targets)
        return out


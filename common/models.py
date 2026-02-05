import torch
from torch import nn


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()

        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)

        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.dropout_layer(last)
        return self.fc(last)

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
  
    def __init__(self, hidden_size, bidirectional=False):
        super(Attention, self).__init__()
        input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, encoder_outputs):
        energy = self.fc(encoder_outputs)
        weights = F.softmax(energy, dim=1)

        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        num_targets: int = 1,
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_targets = num_targets

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        self.dropout_layer = nn.Dropout(dropout)
        fc_input_dim = hidden_size * 2 if self.bidirectional else hidden_size
        fc_out = horizon * self.num_targets
        self.fc = nn.Linear(fc_input_dim, fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        last_step = self.dropout_layer(last_step)
        out = self.fc(last_step)
        if self.num_targets > 1:
            return out.view(out.size(0), self.horizon, self.num_targets)
        return out


class ChangeHeadForecaster(nn.Module):
    """Residual / change (difference) wrapper.

    Injects the last observed target values directly into the forecast so the
    base network only learns the CHANGE (delta) to the horizon, rather than
    echoing the current load level. On flat targets (e.g. memory) this removes
    the added noise that made the model strictly worse than persistence.

    pred = base(x) + x[:, -1, :num_targets]  (broadcast over the horizon)
    """

    def __init__(self, base, inject_mask=None):
        super().__init__()
        self.base = base
        self.register_buffer("inject_mask", torch.tensor(inject_mask, dtype=torch.bool)
                             if inject_mask is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.base(x)
        single = preds.dim() == 2
        if single:
            preds = preds.unsqueeze(-1)
        num_targets = preds.shape[-1]
        last = x[..., -1, :num_targets].unsqueeze(1)
        if self.inject_mask is not None:
            last = last * self.inject_mask.to(last)
        out = preds + last
        if single:
            out = out.squeeze(-1)
        return out

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


class StochasticAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(StochasticAttention, self).__init__()
        input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc_a = nn.Linear(input_dim, input_dim)
        self.fc_mu = nn.Linear(input_dim, 1)
        self.fc_sigma = nn.Linear(input_dim, 1)

    def forward(self, encoder_outputs):
        a = torch.tanh(self.fc_a(encoder_outputs))
        mu = self.fc_mu(a)
        log_sigma = self.fc_sigma(a)
        sigma = torch.exp(log_sigma)
        
        if self.training:
            epsilon = torch.randn_like(mu)
            energy = mu + sigma * epsilon
        else:
            energy = mu
            
        weights = F.softmax(energy, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights


class UncertaintyAwareForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        horizon: int = 5,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.encoder = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = StochasticAttention(hidden_size, bidirectional=True)

        decoder_hidden_dim = hidden_size * 2
        if self.rnn_type == "lstm":
            self.decoder_cell = nn.LSTMCell(input_size, decoder_hidden_dim)
        else:
            self.decoder_cell = nn.GRUCell(input_size, decoder_hidden_dim)

        self.fc = nn.Linear(decoder_hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        enc_out, _ = self.encoder(x)

        context, _ = self.attention(enc_out)

        hx = context
        cx = torch.zeros_like(hx) if self.rnn_type == "lstm" else None

        curr_input = x[:, -1, :]

        outputs = []
        for _ in range(self.horizon):
            if self.rnn_type == "lstm":
                hx, cx = self.decoder_cell(curr_input, (hx, cx))
            else:
                hx = self.decoder_cell(curr_input, hx)

            out = self.fc(hx)
            outputs.append(out)

            curr_input = torch.zeros((batch_size, x.size(-1))).to(x.device)

        return torch.stack(outputs, dim=1)


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
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

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
        self.fc = nn.Linear(fc_input_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        last_step = self.dropout_layer(last_step)
        return self.fc(last_step)

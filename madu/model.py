"""Madu Seq2Seq workload model (§4.1).

Encoder: LSTM over X^E (n steps of {CPM, t_m, t_h, t_w}).
Decoder: autoregressive LSTM for m steps. Each step outputs a mean
mu = f(X^E) and std sigma = softplus(g(X^E)) (Eq.2), trained with the
Gaussian NLL loss (Eq.1).

Attention (§4.1.4, Eq.3-6): a_tj = tanh(W_a h_{t-1} + U_a h^E_j);
stochastic mode samples s_tj ~ N(mu_tj, sigma_tj) with
mu_tj = W_mu a_tj + b_mu, sigma_tj = softplus(W_sig a_tj + b_sig)
via the reparameterization trick (sampled in training, mean used
in eval). Note: the paper writes sigma_tj as a raw linear map, which can
go negative; softplus (+eps) is the well-defined implementation of the
same Gaussian-sampling rule. 'deterministic' uses s_tj = W a_tj + b;
'none' uses the last encoder hidden state as context. The decoder step
takes [prev_cpm, cv] as input: an implementation of Eq.(4) in which the
previous sigma is carried by the recurrent state rather than fed
explicitly. Final prediction Y_hat = mu + sigma
(Eq.8). With uncertainty=False the model is the paper's deterministic
Seq2Seq baseline (single mu head, MSE).
"""
import torch
from torch import nn


class MaduSeq2Seq(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1,
                 horizon=5, attention="stochastic", uncertainty=True,
                 dropout=0.0, sigma_eps=1e-3):
        super().__init__()
        if attention not in ("stochastic", "deterministic", "none"):
            raise ValueError(f"unknown attention='{attention}'")
        self.horizon = horizon
        self.attention = attention
        self.uncertainty = uncertainty
        self.sigma_eps = sigma_eps

        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_dim = hidden_size
        self.W_a = nn.Linear(hidden_size, self.attn_dim)
        self.U_a = nn.Linear(hidden_size, self.attn_dim)
        if attention == "stochastic":
            self.W_mu = nn.Linear(self.attn_dim, 1)
            self.b_mu = nn.Parameter(torch.zeros(1))
            self.W_sig = nn.Linear(self.attn_dim, 1)
            self.b_sig = nn.Parameter(torch.zeros(1))
        else:
            self.W_s = nn.Linear(self.attn_dim, 1)

        # decoder input: [prev_cpm (1) + context (hidden)]
        self.decoder_cell = nn.LSTMCell(1 + hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, 1)
        if uncertainty:
            self.sig_head = nn.Linear(hidden_size, 1)

    def _attend(self, dec_h, enc_out):
        # dec_h: (B, H); enc_out: (B, n, H)
        a = torch.tanh(
            self.W_a(dec_h).unsqueeze(1) + self.U_a(enc_out)
        )  # (B, n, A)
        if self.attention == "stochastic":
            mu = self.W_mu(a).squeeze(-1) + self.b_mu  # (B, n)
            sig = nn.functional.softplus(
                self.W_sig(a).squeeze(-1) + self.b_sig
            ) + self.sigma_eps
            if self.training:
                s = mu + sig * torch.randn_like(sig)
            else:
                s = mu
        else:  # deterministic
            s = self.W_s(a).squeeze(-1)
        alpha = torch.softmax(s, dim=1)  # (B, n) Eq.5
        cv = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # (B, H)
        return cv

    def forward(self, x, y_teacher=None):
        # x: (B, n, F); y_teacher: (B, m) normalized CPM or None
        enc_out, (h_n, c_n) = self.encoder(x)
        h, c = h_n[-1], c_n[-1]
        prev = x[:, -1, 0:1]  # Start token: last observed CPM (Fig.7)
        mus, sigs = [], []
        for k in range(self.horizon):
            if self.attention == "none":
                cv = enc_out[:, -1, :]
            else:
                cv = self._attend(h, enc_out)
            h, c = self.decoder_cell(torch.cat([prev, cv], dim=1), (h, c))
            mu = self.mu_head(h)  # (B, 1)
            if self.uncertainty:
                sig = nn.functional.softplus(self.sig_head(h)) + self.sigma_eps
            else:
                sig = torch.zeros_like(mu)
            mus.append(mu)
            sigs.append(sig)
            if y_teacher is not None and self.training:
                prev = y_teacher[:, k:k + 1]  # teacher forcing
            else:
                prev = mu.detach() if not self.uncertainty else mu
        mu_seq = torch.cat(mus, dim=1)
        sig_seq = torch.cat(sigs, dim=1)
        return mu_seq, sig_seq

    def predict(self, x):
        """Final workload prediction Y_hat = mu + sigma (Eq.8)."""
        self.eval()
        with torch.no_grad():
            mu, sig = self.forward(x)
        return (mu + sig).clamp_min(0.0), mu.clamp_min(0.0), sig

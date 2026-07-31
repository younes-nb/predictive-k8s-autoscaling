import numpy as np
import torch
import torch.nn as nn


class _PathEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        kernel_sizes: tuple,
        conv1_out_ch: int,
        conv2_out_ch: int,
        bilstm_hidden: tuple,
    ):
        super().__init__()
        K = len(kernel_sizes)
        h = bilstm_hidden

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
        self.bilstm1 = nn.LSTM(lstm_in, h[0], batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(h[0] * 2, h[1], batch_first=True, bidirectional=True)
        self.bilstm3 = nn.LSTM(h[1] * 2, h[2], batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = torch.cat([conv(x) for conv in self.conv_set1], dim=1)
        out2 = torch.cat([conv(out1) for conv in self.conv_set2], dim=1)
        seq = out2.permute(0, 2, 1)
        o1, _ = self.bilstm1(seq)
        o2, _ = self.bilstm2(o1)
        o3, _ = self.bilstm3(o2)
        return o3


class CnnBiLSTM_DualPath(nn.Module):
    """Dual-path CPU/memory forecaster.

    CPU: conv + BiLSTM encoder regressing future CPU values directly.
    Memory: SWT-persistence baseline (reconstruction of the last value)
    plus a small learned MLP that corrects the per-step prediction from
    window statistics (deviation from trend, slopes, detail energy, level).
    The correction is trained end-to-end with per-target L1 loss.
    """

    def __init__(
        self,
        in_channels: int = 8,
        input_len: int = 60,
        pred_horizon: int = 5,
        kernel_sizes: tuple = (2, 4, 8),
        conv1_out_ch: int = 32,
        conv2_out_ch: int = 64,
        bilstm_hidden: tuple = (32, 64, 128),
        cpu_channels: int = 5,
        mem_channels: int = 3,
        dropout: float = 0.1,
        num_targets: int = 1,
    ):
        super().__init__()
        assert in_channels == cpu_channels + mem_channels
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.cpu_channels = cpu_channels
        self.mem_channels = mem_channels

        import pywt
        w = self._swt_reconstruction_weights(input_len, mem_channels)
        self.register_buffer("mem_recon_w", torch.from_numpy(w).float())

        h_cpu = bilstm_hidden
        self.enc_cpu = _PathEncoder(
            cpu_channels, kernel_sizes, conv1_out_ch, conv2_out_ch, h_cpu
        )

        self.dropout = nn.Dropout(dropout)

        n_cpu = h_cpu[2] * 2
        self.fc_cpu = nn.Linear(n_cpu * 2, pred_horizon)

        self.mem_bias = nn.Parameter(torch.zeros(pred_horizon))
        self.mem_nn_head = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, pred_horizon),
        )
        nn.init.zeros_(self.mem_nn_head[-1].weight)
        nn.init.zeros_(self.mem_nn_head[-1].bias)

    @staticmethod
    def _swt_reconstruction_weights(input_len: int, n_bands: int) -> np.ndarray:
        import pywt
        w = np.zeros((n_bands, input_len), dtype=np.float64)
        for b in range(n_bands):
            for i in range(input_len):
                coeffs = [np.zeros(input_len) for _ in range(n_bands)]
                coeffs[b][i] = 1.0
                recon = pywt.iswt(coeffs, "sym4", norm=True)
                w[b, i] = recon[-1]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        xc = x.permute(0, 2, 1)

        x_cpu = xc[:, :self.cpu_channels, :]
        x_mem = xc[:, self.cpu_channels:, :]

        o_cpu = self.enc_cpu(x_cpu)
        o_cpu = self.dropout(o_cpu)
        cpu_out = self.fc_cpu(torch.cat([o_cpu[:, -1, :], o_cpu.mean(dim=1)], dim=1))

        mem_bands = x_mem
        naive_mem = (mem_bands * self.mem_recon_w.unsqueeze(0)).sum(dim=(1, 2))
        rawdev = naive_mem - mem_bands[:, 0, -1]
        a2_last = mem_bands[:, 0, -1]
        a2_slope5 = a2_last - mem_bands[:, 0, -6]
        a2_slope20 = a2_last - mem_bands[:, 0, -21]
        d2_energy = (mem_bands[:, 2, -6:] ** 2).mean(dim=1)
        feats = torch.stack(
            [rawdev, rawdev ** 2, a2_slope5, a2_slope20, d2_energy, naive_mem],
            dim=1,
        )
        mem_out = naive_mem.unsqueeze(1) + self.mem_bias + self.mem_nn_head(feats)

        out = torch.stack([cpu_out, mem_out], dim=-1)

        if self.num_targets > 1:
            return out
        return out

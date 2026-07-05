import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TimesBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ):
        super().__init__()
        self.input_len = input_len
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels

        kernel_sizes = [1, 3, 5, 7][:num_kernels]
        self.conv_blocks = nn.ModuleList()
        for ks in kernel_sizes:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(d_model, d_ff, kernel_size=ks, padding="same"),
                nn.GELU(),
                nn.Conv2d(d_ff, d_model, kernel_size=ks, padding="same"),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        x_avg = x.mean(dim=-1)
        spectrum = torch.fft.rfft(x_avg, dim=1)
        amplitude = spectrum.abs().mean(dim=0)
        amplitude[0] = 0.0

        top_k = min(self.top_k, amplitude.size(0) - 1)
        if top_k <= 0:
            return x

        top_vals, top_idxs = torch.topk(amplitude, top_k)
        periods = (self.input_len / (top_idxs + 1).float()).long().clamp(min=2)
        weights = F.softmax(top_vals, dim=0)

        aggregated = torch.zeros_like(x)

        for p_idx, p in enumerate(periods):
            p = p.item()
            pad_len = (p - (T % p)) % p
            if pad_len > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len))
            else:
                x_pad = x

            padded_len = x_pad.size(1)
            n_periods = padded_len // p

            x_2d = x_pad.permute(0, 2, 1)
            x_2d = x_2d.reshape(B, D, n_periods, p)

            branch_outs = []
            for conv in self.conv_blocks:
                branch_outs.append(conv(x_2d))

            x_2d_out = torch.mean(torch.stack(branch_outs, dim=0), dim=0)

            x_back = x_2d_out.reshape(B, D, padded_len).permute(0, 2, 1)
            if pad_len > 0:
                x_back = x_back[:, :T, :]

            aggregated = aggregated + weights[p_idx] * x_back

        return aggregated + x


class TimesNetForecaster(nn.Module):
    def __init__(
        self,
        total_channels: int = 6,
        input_len: int = 60,
        pred_horizon: int = 5,
        top_k_periods: int = 3,
        d_model: int = 32,
        d_ff: int = 64,
        num_kernels: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_channels = total_channels
        self.input_len = input_len
        self.pred_horizon = pred_horizon
        self.d_model = d_model
        self.num_blocks = num_blocks

        self.embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            TimesBlock(input_len, top_k_periods, d_model, d_ff, num_kernels)
            for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(input_len * d_model, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        x_flat = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.embed(x_flat) + self.pos_embed

        for block, norm in zip(self.blocks, self.norms):
            h = self.dropout(norm(block(h)))

        h = h.reshape(B * C, T * self.d_model)
        per_ch_pred = self.head(h)
        per_ch_pred = per_ch_pred.reshape(B, C, self.pred_horizon)
        output = per_ch_pred.sum(dim=1)
        return output


if __name__ == "__main__":
    from experiments.sdtnet.config import CFG

    model = TimesNetForecaster(
        total_channels=CFG.TOTAL_CHANNELS,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        top_k_periods=CFG.TIMESNET_TOP_K_PERIODS,
        d_model=CFG.TIMESNET_D_MODEL,
        d_ff=CFG.TIMESNET_D_FF,
        num_kernels=CFG.TIMESNET_NUM_KERNELS,
        num_blocks=CFG.TIMESNET_NUM_BLOCKS,
        dropout=CFG.TIMESNET_DROPOUT,
    )
    x = torch.randn(4, CFG.INPUT_LEN, CFG.TOTAL_CHANNELS)
    out = model(x)
    assert out.shape == (4, CFG.PRED_HORIZON), f"Expected (4, {CFG.PRED_HORIZON}), got {out.shape}"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Model parameters: {total_params}")
    print("TimesNetForecaster smoke test passed")

import os
import sys

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1].repeat(1, pad_left)
        end = x[:, -1:].repeat(1, pad_right)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.unsqueeze(1)).squeeze(1)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    def __init__(
        self,
        total_channels: int = 6,
        input_len: int = 60,
        pred_horizon: int = 5,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.total_channels = total_channels
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        self.decomp = MovingAvgDecomp(moving_avg_kernel)
        self.linear_seasonal = nn.Linear(input_len, pred_horizon)
        self.linear_trend = nn.Linear(input_len, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x_flat = x.permute(0, 2, 1).reshape(batch * self.total_channels, self.input_len)

        seasonal, trend = self.decomp(x_flat)
        out = self.linear_seasonal(seasonal) + self.linear_trend(trend)

        out = out.reshape(batch, self.total_channels, self.pred_horizon)
        return out.sum(dim=1)


if __name__ == "__main__":
    from experiments.sdtnet.config import CFG

    model = DLinearForecaster(
        total_channels=CFG.TOTAL_CHANNELS,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        moving_avg_kernel=CFG.DLINEAR_MOVING_AVG_KERNEL,
    )
    x = torch.randn(4, CFG.INPUT_LEN, CFG.TOTAL_CHANNELS)
    out = model(x)
    assert out.shape == (4, CFG.PRED_HORIZON), f"Expected (4, {CFG.PRED_HORIZON}), got {out.shape}"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Model parameters: {total_params}")
    print("DLinearForecaster smoke test passed")

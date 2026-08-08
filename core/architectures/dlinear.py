import torch
import torch.nn as nn


class moving_avg(nn.Module):

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front_pad = self.kernel_size // 2
        back_pad = self.kernel_size - 1 - front_pad
        front = x[:, :1, :].expand(-1, front_pad, -1)
        end = x[:, -1:, :].expand(-1, back_pad, -1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class series_decomp(nn.Module):

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


class DLinear(nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        input_len: int = 128,
        pred_horizon: int = 5,
        kernel_size: int = 25,
        individual: bool = False,
        num_targets: int = 1,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.individual = individual

        self.decomp = series_decomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(input_len, pred_horizon) for _ in range(in_channels)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(input_len, pred_horizon) for _ in range(in_channels)
            ])
        else:
            self.linear_seasonal = nn.Linear(input_len, pred_horizon)
            self.linear_trend = nn.Linear(input_len, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.stack([
                lin(seasonal[:, i, :]) for i, lin in enumerate(self.linear_seasonal)
            ], dim=1)
            trend_out = torch.stack([
                lin(trend[:, i, :]) for i, lin in enumerate(self.linear_trend)
            ], dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        out = (seasonal_out + trend_out).permute(0, 2, 1)
        out = out[:, :, :self.num_targets]
        if self.num_targets == 1:
            return out[..., 0]
        return out

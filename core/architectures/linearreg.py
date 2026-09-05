import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(
        self,
        input_size: int = 3,
        input_len: int = 60,
        pred_horizon: int = 1,
        num_targets: int = 1,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.fc = nn.Linear(input_size * input_len, pred_horizon * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        out = self.fc(flat)
        if self.num_targets > 1:
            return out.view(batch, self.pred_horizon, self.num_targets)
        return out

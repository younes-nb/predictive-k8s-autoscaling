import os
import sys

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class ChannelIndependentTCNForecaster(nn.Module):
    def __init__(
        self,
        total_channels: int = 6,
        input_len: int = 60,
        pred_horizon: int = 5,
        num_filters: int = 64,
        kernel_size: int = 3,
        dilations: tuple = (1, 2, 4, 8),
        dropout: float = 0.1,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.total_channels = total_channels
        self.input_len = input_len
        self.pred_horizon = pred_horizon
        self.residual_prediction = residual_prediction

        layers = []
        in_ch = 1
        for d in dilations:
            layers.extend([
                nn.Conv1d(in_ch, num_filters, kernel_size,
                          padding=d * (kernel_size - 1) // 2,
                          dilation=d),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = num_filters

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters, pred_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x_flat = x.permute(0, 2, 1).reshape(
            batch * self.total_channels, 1, self.input_len
        )
        features = self.backbone(x_flat)
        delta = self.head(features)
        delta = delta.reshape(batch, self.total_channels, self.pred_horizon)

        if self.residual_prediction:
            last_obs = x[:, -1:, :]
            last_obs = last_obs.permute(0, 2, 1)
            pred = last_obs + delta
        else:
            pred = delta

        return pred.sum(dim=1)


if __name__ == "__main__":
    from experiments.sdtnet.config import CFG

    model = ChannelIndependentTCNForecaster(
        total_channels=CFG.TOTAL_CHANNELS,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        num_filters=CFG.TCN_NUM_FILTERS,
        kernel_size=CFG.TCN_KERNEL_SIZE,
        dilations=CFG.TCN_DILATIONS,
        dropout=CFG.TCN_DROPOUT,
        residual_prediction=CFG.RESIDUAL_PREDICTION,
    )
    x = torch.randn(4, CFG.INPUT_LEN, CFG.TOTAL_CHANNELS)
    out = model(x)
    assert out.shape == (4, CFG.PRED_HORIZON), f"Expected (4, {CFG.PRED_HORIZON}), got {out.shape}"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Model parameters: {total_params}")
    print("ChannelIndependentTCNForecaster smoke test passed")

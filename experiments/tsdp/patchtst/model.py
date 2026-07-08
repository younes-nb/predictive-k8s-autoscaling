import math
import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.attention import SDPBackend, sdpa_kernel

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tsdp.config import CFG as GLOBAL_CFG
from experiments.tsdp.patchtst.config import CFG as ARCH_CFG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class PatchTST(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_horizon: int,
        n_channels: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.pred_horizon = pred_horizon
        n_patches = (input_len - patch_len) // stride + 1
        assert n_patches > 0, (
            f"input_len={input_len} too small for patch_len={patch_len}, stride={stride}"
        )

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_patches + 1, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)

        batch_size, n_channels, seq_len = x.shape
        x = x.reshape(batch_size * n_channels, 1, seq_len)

        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.squeeze(1)

        emb = self.patch_proj(patches)
        emb = self.pos_encoder(emb)

        with sdpa_kernel(SDPBackend.MATH):
            out = self.transformer_encoder(emb)

        pooled = out.mean(dim=1)
        pred = self.head(pooled)

        pred = pred.reshape(batch_size, n_channels, self.pred_horizon)
        pred = pred.mean(dim=1)

        return pred


if __name__ == "__main__":
    model = PatchTST(
        input_len=GLOBAL_CFG.INPUT_LEN,
        pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        n_channels=GLOBAL_CFG.TOTAL_CHANNELS,
        patch_len=ARCH_CFG.PATCH_LEN,
        stride=ARCH_CFG.PATCH_STRIDE,
        d_model=ARCH_CFG.D_MODEL,
        n_heads=ARCH_CFG.N_HEADS,
        n_layers=ARCH_CFG.N_LAYERS,
        d_ff=ARCH_CFG.D_FF,
        dropout=ARCH_CFG.DROPOUT,
    )
    x = torch.randn(4, GLOBAL_CFG.INPUT_LEN, GLOBAL_CFG.TOTAL_CHANNELS)
    out = model(x)
    assert out.shape == (4, GLOBAL_CFG.PRED_HORIZON), f"Expected (4, {GLOBAL_CFG.PRED_HORIZON}), got {out.shape}"
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("PatchTST smoke test passed")

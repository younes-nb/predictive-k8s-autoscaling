
import os
import sys
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", "Using padding='same' with even kernel lengths")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

class CvcbmModel(nn.Module):

    def __init__(
        self,
        input_len: int = 30,
        kernel_sizes: tuple = (2, 4, 8),
        conv1_out_ch: int = 32,
        conv2_out_ch: int = 64,
        bilstm_hidden: tuple = (32, 64, 128),
    ):
        super().__init__()
        del input_len
        K = len(kernel_sizes)

        self.conv_set1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, conv1_out_ch, ks, padding="same"),
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
        h = bilstm_hidden
        self.bilstm1 = nn.LSTM(lstm_in, h[0], batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(h[0] * 2, h[1], batch_first=True, bidirectional=True)
        self.bilstm3 = nn.LSTM(h[1] * 2, h[2], batch_first=True, bidirectional=True)

        self.fc = nn.Linear(h[2] * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        xc = x.permute(0, 2, 1)

        out1 = torch.cat([conv(xc) for conv in self.conv_set1], dim=1)
        out2 = torch.cat([conv(out1) for conv in self.conv_set2], dim=1)

        seq = out2.permute(0, 2, 1)

        o1, _ = self.bilstm1(seq)
        o2, _ = self.bilstm2(o1)
        o3, _ = self.bilstm3(o2)

        last = o3[:, -1, :]
        return self.fc(last)

if __name__ == "__main__":
    from experiments.cvcbm.config import CFG

    model = CvcbmModel(
        input_len=CFG.INPUT_LEN,
        kernel_sizes=CFG.KERNEL_SIZES,
        conv1_out_ch=CFG.CONV1_OUT_CH,
        conv2_out_ch=CFG.CONV2_OUT_CH,
        bilstm_hidden=CFG.BILSTM_HIDDEN,
    )
    x = torch.randn(4, CFG.INPUT_LEN, 1)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {tuple(out.shape)}"
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("CvcbmModel smoke test passed")

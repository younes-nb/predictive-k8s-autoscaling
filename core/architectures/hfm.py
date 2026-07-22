import torch
import torch.nn as nn


class DSConv1D(nn.Module):
    """Depthwise-separable Conv1D block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2,
            groups=in_ch,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return self.dropout(x)


class HierarchicalFrequencyMixer(nn.Module):
    """HFM: Hierarchical Frequency-Mixer.

    Processes VMD / SWT-decomposed channels through separate frequency
    branches, fuses them with gated attention, and models temporal
    dependencies with a single GRU layer.

    Channel layout (cpu_mem_both, default decomposition):
        [cpu_vmd(9) | cpu_D4 | cpu_D3 | cpu_D2 | cpu_A4
         mem_vmd(9) | mem_D2 | mem_A2]
    Total = 24 channels.

    CPU-only layout (feature_set=cpu):
        [cpu_vmd(9) | cpu_D4 | cpu_D3 | cpu_D2 | cpu_A4]
    Total = 13 channels.
    """

    def __init__(
        self,
        input_len=64,
        pred_horizon=5,
        cpu_vmd=9,
        cpu_swt=4,
        mem_vmd=0,
        mem_swt=0,
        num_targets=2,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets

        self.cpu_vmd = cpu_vmd
        self.cpu_swt = cpu_swt
        self.mem_vmd = mem_vmd
        self.mem_swt = mem_swt

        # --- CPU branches ------------------------------------------------
        self.cpu_vmd_branch = nn.Sequential(
            DSConv1D(cpu_vmd, 48, 3, dropout),
            DSConv1D(48, 48, 3, dropout),
        )
        self.cpu_detail_branch = nn.Sequential(
            DSConv1D(cpu_swt - 1, 24, 5, dropout),
            DSConv1D(24, 24, 3, dropout),
        )
        self.cpu_trend_branch = nn.Sequential(
            nn.Conv1d(1, 24, 1),
            nn.GELU(),
        )

        has_mem = mem_vmd > 0 and mem_swt > 0

        # --- Memory branches (conditional) -------------------------------
        if has_mem:
            self.mem_vmd_branch = nn.Sequential(
                DSConv1D(mem_vmd, 48, 3, dropout),
                DSConv1D(48, 48, 3, dropout),
            )
            self.mem_detail_branch = nn.Sequential(
                DSConv1D(mem_swt - 1, 24, 5, dropout),
                DSConv1D(24, 24, 3, dropout),
            )
            self.mem_trend_branch = nn.Sequential(
                nn.Conv1d(1, 24, 1),
                nn.GELU(),
            )

        # --- Cross-frequency fusion --------------------------------------
        # CPU-only: 48+24+24=96, multi-target: 96*2=192
        fusion_in = 192 if has_mem else 96
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in, 96, 1),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )

        # --- Frequency gated attention -----------------------------------
        self.freq_attention = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 96),
            nn.Sigmoid(),
        )

        # --- Temporal modeling --------------------------------------------
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
        )

        # --- Output head -------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, pred_horizon * num_targets),
        )

    def forward(self, x):
        # x: (B, input_len, total_channels)
        x = x.transpose(1, 2)  # (B, total_channels, input_len)

        # --- Split CPU channels ------------------------------------------
        cpu_n = self.cpu_vmd + self.cpu_swt
        cpu_vmd = x[:, :self.cpu_vmd, :]
        cpu_detail = x[:, self.cpu_vmd:cpu_n - 1, :]
        cpu_trend = x[:, cpu_n - 1:cpu_n, :]

        cpu_vmd_feat = self.cpu_vmd_branch(cpu_vmd)
        cpu_detail_feat = self.cpu_detail_branch(cpu_detail)
        cpu_trend_feat = self.cpu_trend_branch(cpu_trend)

        parts = [cpu_vmd_feat, cpu_detail_feat, cpu_trend_feat]

        # --- Memory branches (conditional) -------------------------------
        if self.mem_vmd > 0 and self.mem_swt > 0:
            mem_start = cpu_n
            mem_vmd = x[:, mem_start:mem_start + self.mem_vmd, :]
            mem_detail = x[:, mem_start + self.mem_vmd:mem_start + self.mem_vmd + self.mem_swt - 1, :]
            mem_trend = x[:, mem_start + self.mem_vmd + self.mem_swt - 1:mem_start + self.mem_vmd + self.mem_swt, :]

            mem_vmd_feat = self.mem_vmd_branch(mem_vmd)
            mem_detail_feat = self.mem_detail_branch(mem_detail)
            mem_trend_feat = self.mem_trend_branch(mem_trend)

            parts.extend([mem_vmd_feat, mem_detail_feat, mem_trend_feat])

        # --- Concatenate frequency features ------------------------------
        fused = torch.cat(parts, dim=1)

        # --- Fuse channels -----------------------------------------------
        fused = self.fusion(fused)  # (B, 64, input_len)

        # --- Gated frequency attention -----------------------------------
        fused_t = fused.transpose(1, 2)  # (B, input_len, 64)
        attn = self.freq_attention(fused_t)
        fused_t = fused_t * attn

        # --- GRU temporal modeling ---------------------------------------
        gru_out, _ = self.gru(fused_t)  # (B, input_len, 64)
        last = gru_out[:, -1, :]  # (B, 64)

        # --- Forecast ----------------------------------------------------
        out = self.head(last)  # (B, pred_horizon * num_targets)
        return out.view(-1, self.pred_horizon, self.num_targets)

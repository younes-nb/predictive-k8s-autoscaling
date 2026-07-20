import torch
import torch.nn as nn
from torch.nn.utils import parametrizations

from core.architectures.tcn_bigru import TemporalBlock


class DualPathwayTcn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_len: int,
        pred_horizon: int,
        cpu_n_vmd: int,
        cpu_n_swt: int,
        mem_n_vmd: int = 0,
        mem_n_swt: int = 0,
        dropout: float = 0.1,
        num_targets: int = 1,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.cpu_n_vmd = cpu_n_vmd
        self.cpu_n_swt = cpu_n_swt
        self.mem_n_vmd = mem_n_vmd
        self.mem_n_swt = mem_n_swt

        micro_filters = (64, 128, 128)
        micro_dilations = (1, 2, 4)
        micro_kernel = 3
        macro_filters = (64, 128, 128, 128)
        macro_dilations = (2, 4, 8, 16)
        macro_kernel = 5

        self.cpu_micro_tcn = self._build_tcn(
            cpu_n_vmd, micro_filters, micro_dilations, micro_kernel, dropout,
        )
        self.cpu_macro_tcn = self._build_tcn(
            cpu_n_swt, macro_filters, macro_dilations, macro_kernel, dropout,
        )
        self.cpu_fusion = parametrizations.weight_norm(nn.Conv1d(256, 128, 1))
        self.cpu_attn_mlp = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1),
        )

        if num_targets > 1:
            self.mem_micro_tcn = self._build_tcn(
                mem_n_vmd, micro_filters, micro_dilations, micro_kernel, dropout,
            )
            self.mem_macro_tcn = self._build_tcn(
                mem_n_swt, macro_filters, macro_dilations, macro_kernel, dropout,
            )
            self.mem_fusion = parametrizations.weight_norm(nn.Conv1d(256, 128, 1))
            self.mem_attn_mlp = nn.Sequential(
                nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1),
            )
            self.shared_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
            self.cpu_head = nn.Linear(128, pred_horizon)
            self.mem_head = nn.Linear(128, pred_horizon)
        else:
            self.output_head = nn.Linear(128, pred_horizon)

        self._init_weights()

    @staticmethod
    def _build_tcn(in_ch, filters, dilations, kernel_size, dropout):
        layers = []
        tcn_in = in_ch
        for i, out_ch in enumerate(filters):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            layers.append(TemporalBlock(tcn_in, out_ch, kernel_size, d, dropout))
            tcn_in = out_ch
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.uniform_(m.weight, a=0.0, b=0.0015)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def _forward_metric(self, x, n_vmd, n_swt, micro_tcn, macro_tcn, fusion, attn_mlp):
        x_micro = x[:, :, :n_vmd]
        x_macro = x[:, :, n_vmd:n_vmd + n_swt]

        micro_out = micro_tcn(x_micro.transpose(1, 2))
        macro_out = macro_tcn(x_macro.transpose(1, 2))

        fused = fusion(torch.cat([micro_out, macro_out], dim=1))
        x_t = fused.transpose(1, 2)
        scores = attn_mlp(x_t)
        weights = torch.softmax(scores, dim=1)
        return (weights * x_t).sum(dim=1)

    def forward(self, x):
        cpu_n = self.cpu_n_vmd + self.cpu_n_swt
        cpu_x = x[:, :, :cpu_n]

        if self.num_targets > 1:
            mem_x = x[:, :, cpu_n:]
            cpu_vec = self._forward_metric(
                cpu_x, self.cpu_n_vmd, self.cpu_n_swt,
                self.cpu_micro_tcn, self.cpu_macro_tcn,
                self.cpu_fusion, self.cpu_attn_mlp,
            )
            mem_vec = self._forward_metric(
                mem_x, self.mem_n_vmd, self.mem_n_swt,
                self.mem_micro_tcn, self.mem_macro_tcn,
                self.mem_fusion, self.mem_attn_mlp,
            )
            shared = self.shared_fc(torch.cat([cpu_vec, mem_vec], dim=1))
            return torch.stack(
                [self.cpu_head(shared), self.mem_head(shared)], dim=-1,
            )
        else:
            cpu_vec = self._forward_metric(
                cpu_x, self.cpu_n_vmd, self.cpu_n_swt,
                self.cpu_micro_tcn, self.cpu_macro_tcn,
                self.cpu_fusion, self.cpu_attn_mlp,
            )
            return self.output_head(cpu_vec)

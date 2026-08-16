import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        horizon: int = 5,
        num_targets: int = 2,
        ensemble_size: int = 5,
        model_types: list = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.ensemble_size = ensemble_size
        
        if model_types is None:
            model_types = ["lstm", "gru", "bilstm", "tcn", "ridge"]
        
        self.models = nn.ModuleList()
        for i, mtype in enumerate(model_types[:ensemble_size]):
            if mtype in ["lstm", "gru", "bilstm"]:
                bidirectional = mtype in ["bilstm"]
                rnn_type = "lstm" if "lstm" in mtype else "gru"
                self.models.append(nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=bidirectional,
                ))
            elif mtype == "tcn":
                self.models.append(self._make_tcn(input_size, hidden_size, horizon, num_targets, dropout))
            elif mtype == "ridge":
                self.models.append(self._make_ridge(input_size, horizon, num_targets))
            else:
                raise ValueError(f"Unknown model type: {mtype}")
        
        self.fc = nn.ModuleList([
            nn.Linear(hidden_size * (2 if "bilstm" in mtype else 1), horizon * num_targets)
            if mtype in ["lstm", "gru", "bilstm"] else None
            for mtype in model_types[:ensemble_size]
        ])
        
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
    
    def _make_tcn(self, input_size, hidden_size, horizon, num_targets, dropout):
        class TCNBlock(nn.Module):
            def __init__(self, in_ch, out_ch, dilation):
                super().__init__()
                padding = (3 - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
                self.chomp = nn.Sequential()  # manual chomp
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                out = self.conv(x)
                out = out[:, :, :-self.conv.padding[0]]  # chomp
                return self.dropout(self.relu(out))
        
        layers = []
        in_ch = input_size
        for i in range(4):
            dilation = 2 ** i
            out_ch = hidden_size
            layers.append(TCNBlock(in_ch, out_ch, dilation))
            in_ch = out_ch
        return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
    
    def _make_ridge(self, input_size, horizon, num_targets):
        # This will be a simple linear layer (trained separately with ridge penalty)
        return nn.Linear(input_size * 128, horizon * num_targets)
    
    def forward(self, x):
        # x: (B, T, C)
        outputs = []
        for i, model in enumerate(self.models):
            if isinstance(model, nn.LSTM):
                out, _ = model(x)
                out = out[:, -1, :]
                out = self.fc[i](out)
            elif hasattr(model, 'children') and not list(model.children()):
                # TCN
                x_tcn = x.permute(0, 2, 1)  # (B, C, T)
                out = model(x_tcn)
                out = out.squeeze(-1)
                out = out.view(out.size(0), -1)
            else:
                # Ridge - flatten
                out = x.view(x.size(0), -1)
                out = model(out)
            
            if self.num_targets > 1:
                out = out.view(out.size(0), self.horizon, self.num_targets)
            outputs.append(out)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            ensemble_out += weights[i] * out
        
        return ensemble_out


class QuantileEnsembleForecaster(nn.Module):
    """Ensemble that outputs ordered quantiles for conformal prediction.
    
    Uses softplus parameterization to enforce q_lower <= q_median <= q_upper:
    q_median = head_q50(h)
    q_lower  = q_median - softplus(head_delta_low(h))
    q_upper  = q_median + softplus(head_delta_high(h))
    """
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        horizon: int = 5,
        num_targets: int = 2,
        quantiles: list = None,
        ensemble_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        # Default to [0.10, 0.50, 0.95] as per spec
        self.quantiles = quantiles or [0.10, 0.50, 0.95]
        self.num_quantiles = len(self.quantiles)
        self.ensemble_size = ensemble_size
        
        # Validate quantiles are ordered
        assert all(self.quantiles[i] < self.quantiles[i+1] for i in range(len(self.quantiles)-1)), "Quantiles must be strictly increasing"
        assert abs(self.quantiles[1] - 0.5) < 1e-6, "Middle quantile must be 0.5 (median)"
        
        # Base forecasters (BiLSTM variants)
        self.base_models = nn.ModuleList()
        self.is_bidirectional = []
        for i in range(ensemble_size):
            bidir = (i % 2 == 0)
            self.is_bidirectional.append(bidir)
            self.base_models.append(nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidir,
            ))
        
        # Three heads per base model for ordered quantiles:
        # q50 (median), delta_low (softplus), delta_high (softplus)
        hidden_mult = [2 if bidir else 1 for bidir in self.is_bidirectional]
        self.head_q50 = nn.ModuleList([
            nn.Linear(hidden_size * hm, horizon * num_targets)
            for hm in hidden_mult
        ])
        self.head_delta_low = nn.ModuleList([
            nn.Linear(hidden_size * hm, horizon * num_targets)
            for hm in hidden_mult
        ])
        self.head_delta_high = nn.ModuleList([
            nn.Linear(hidden_size * hm, horizon * num_targets)
            for hm in hidden_mult
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
    
    def forward(self, x):
        # x: (B, T, C)
        quantile_outputs = []
        for i, model in enumerate(self.base_models):
            out, _ = model(x)
            last = out[:, -1, :]  # (B, hidden * (2 if bidir else 1))
            
            # Three heads -> ordered quantiles
            q50 = self.head_q50[i](last).view(-1, self.horizon, self.num_targets)
            delta_low = F.softplus(self.head_delta_low[i](last)).view(-1, self.horizon, self.num_targets)
            delta_high = F.softplus(self.head_delta_high[i](last)).view(-1, self.horizon, self.num_targets)
            
            q_lower = q50 - delta_low
            q_upper = q50 + delta_high
            
            # Stack: [q_lower, q50, q_upper] -> (B, H, T, 3)
            quantile_out = torch.stack([q_lower, q50, q_upper], dim=-1)
            quantile_outputs.append(quantile_out)
        
        # Weighted ensemble of quantiles
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_quantiles = torch.zeros_like(quantile_outputs[0])
        for i, q_out in enumerate(quantile_outputs):
            ensemble_quantiles += weights[i] * q_out
        
        return ensemble_quantiles  # (B, H, num_targets, num_quantiles)
    
    def predict_intervals(self, x, alpha=0.1):
        """Return lower and upper bounds for prediction intervals."""
        quantiles = self.forward(x)
        lower_idx = 0
        upper_idx = self.num_quantiles - 1
        lower = quantiles[:, :, :, lower_idx]
        upper = quantiles[:, :, :, upper_idx]
        median = quantiles[:, :, :, self.num_quantiles // 2]
        return lower, upper, median


def pinball_loss(y_pred, y_true, quantiles, weights=None):
    """Pinball loss for quantile regression with optional per-quantile weights.
    
    Args:
        y_pred: (B, H, num_targets, num_quantiles)
        y_true: (B, H, num_targets)
        quantiles: tensor of quantile levels (num_quantiles,)
        weights: optional tensor of weights per quantile (num_quantiles,)
    """
    errors = y_true.unsqueeze(-1) - y_pred  # (B, H, T, Q)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    if weights is not None:
        loss = loss * weights.view(1, 1, 1, -1)
    return loss.mean()
# Conformal Prediction for HPA: Implementation Plan

Based on the detailed technical specification, this plan addresses the gaps between the current codebase and the target architecture:

```
BiLSTM (existing) → Quantile Heads (q0.05, q0.5, q0.95) → Weighted Pinball Loss
    → CQR Calibration → Online Adaptive Upper-Tail Conformal → Spike-Aware Bounds
```

---

## Current State Assessment

| Component | Status | Gap |
|-----------|--------|-----|
| **QuantileEnsembleForecaster** | ✅ Exists | No quantile ordering enforcement (q10 ≤ q50 ≤ q95) |
| **Pinball Loss** | ✅ Implemented | Equal weights; needs w_0.05=1, w_0.5=1, w_0.95=2 |
| **EnbPI Calibration** | ⚠️ Stubbed | `_fit_enbpi_calibrator` has `pass` — never fits |
| **Replay Evaluation** | ⚠️ Point-only | No quantile extraction, CQR intervals, coverage metrics |
| **Adaptive Conformal** | ⚠️ Partial | `ConformalCalibrator` exists but not integrated |
| **Spike Metrics** | ❌ Missing | No spike coverage, spike miss rate, mean excess error |

---

## Phase 1: Model Architecture & Loss (Core)

### 1.1 Enforce Quantile Ordering in `QuantileEnsembleForecaster`
**File:** `core/architectures/ensemble.py`

Replace independent quantile heads with softplus parameterization:

```python
# Current: independent Linear heads → can cross
# Target: q50 head + delta_low (softplus) + delta_high (softplus)
# q10 = q50 - softplus(delta_low)
# q95 = q50 + softplus(delta_high)
```

**Changes:**
- Modify `__init__`: three heads per base model (q50, delta_low, delta_high)
- Modify `forward`: apply softplus, compute ordered quantiles
- Keep `predict_intervals()` API unchanged

### 1.2 Weighted Pinball Loss
**File:** `training/enbpi_train.py` → `_compute_quantile_loss()`

```python
# Current: loss.mean() across all quantiles equally
# Target: weighted loss
weights = torch.tensor([1.0, 1.0, 2.0], device=preds.device)  # q0.05, q0.5, q0.95
loss = (weights * torch.max(quantiles * errors, (quantiles - 1) * errors)).mean()
```

**Config:** Add `--quantile_weights` arg (default: `1,1,2`)

---

## Phase 2: Offline CQR Calibration (Training Pipeline)

### 2.1 Implement Proper CQR Fitting in `_fit_enbpi_calibrator`
**File:** `training/enbpi_train.py`

Replace the stub (lines 393-422) with:

```python
def _fit_cqr_calibrator(model, val_loader, device, alpha=0.1):
    """Fit CQR conformal calibrator on validation residuals."""
    model.eval()
    all_q10, all_q50, all_q95, all_y = [], [], [], []
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.float().to(device)
            preds = model(x)  # (B, H, T, Q)
            all_q10.append(preds[:, :, :, 0].cpu())   # q0.05
            all_q50.append(preds[:, :, :, 1].cpu())   # q0.50
            all_q95.append(preds[:, :, :, 2].cpu())   # q0.95
            all_y.append(y)
    
    q10 = torch.cat(all_q10).numpy()   # (N, H, T)
    q50 = torch.cat(all_q50).numpy()
    q95 = torch.cat(all_q95).numpy()
    y   = torch.cat(all_y).numpy()
    
    calibrators = {}
    for t_idx in range(y.shape[2]):  # per target (CPU, Mem)
        calibrators[t_idx] = {}
        for h in range(y.shape[1]):  # per horizon step
            # CQR conformity scores: max(0, q10 - y, y - q95)
            scores = np.maximum(0, np.maximum(q10[:, h, t_idx] - y[:, h, t_idx],
                                               y[:, h, t_idx] - q95[:, h, t_idx]))
            # Conformal quantile
            q_conf = np.quantile(scores, 1 - alpha, method='higher')
            calibrators[t_idx][h] = {
                'q_conf': float(q_conf),
                'q10': q10[:, h, t_idx],  # for adaptive updates
                'q95': q95[:, h, t_idx],
            }
    return calibrators
```

### 2.2 Save Calibrator with Checkpoint
**File:** `training/enbpi_train.py` (after line 318)

```python
# Save calibrators in checkpoint
best_model_state["cqr_calibrators"] = calibrators
best_model_state["cqr_alpha"] = 0.1
torch.save(best_model_state, args.checkpoint_path)
```

### 2.3 Support Calibrator Type Selection
**File:** `training/enbpi_train.py` → add arg `--calibrator_type` (choices: `cqr`, `enbpi`, `aci`)

---

## Phase 3: Online Adaptive Conformal & Evaluation (Replay)

### 3.1 Implement `AdaptiveUpperConformal` Class
**File:** `analytics/adaptive_conformal.py` (new)

```python
class AdaptiveUpperConformal:
    """Separate upper/lower adaptive conformal for asymmetric errors."""
    def __init__(self, window_size=500, alpha=0.05, eta=0.01,
                 alpha_min=0.01, alpha_max=0.20):
        self.scores_upper = deque(maxlen=window_size)
        self.scores_lower = deque(maxlen=window_size)
        self.alpha_u = alpha
        self.alpha_l = alpha
        self.target_alpha = alpha
        self.eta = eta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def get_correction_upper(self, q95):
        if not self.scores_upper: return 0.0
        q_conf = np.quantile(self.scores_upper, 1 - self.alpha_u, method='higher')
        return q95 + q_conf
    
    def get_correction_lower(self, q10):
        if not self.scores_lower: return 0.0
        q_conf = np.quantile(self.scores_lower, 1 - self.alpha_l, method='higher')
        return q10 - q_conf
    
    def update(self, y, q10, q95):
        # Compute scores BEFORE adding to buffer (no leakage)
        score_u = max(0.0, y - q95)
        score_l = max(0.0, q10 - y)
        self.scores_upper.append(score_u)
        self.scores_lower.append(score_l)
        
        # Update alphas based on misses
        upper_miss = float(y > (q95 + self.get_correction_upper(q95)))
        lower_miss = float(y < (q10 - self.get_correction_lower(q10)))
        
        self.alpha_u = np.clip(self.alpha_u + self.eta * (self.target_alpha - upper_miss),
                               self.alpha_min, self.alpha_max)
        self.alpha_l = np.clip(self.alpha_l + self.eta * (self.target_alpha - lower_miss),
                               self.alpha_min, self.alpha_max)
```

### 3.2 Spike Metrics Computation
**File:** `analytics/spike_metrics.py` (new)

```python
def compute_spike_metrics(y_true, y_pred, q10, q95, upper_bound, spike_threshold=None):
    """Compute spike-aware metrics."""
    if spike_threshold is None:
        spike_threshold = np.quantile(y_true, 0.95)
    
    is_spike = y_true > spike_threshold
    n_spikes = is_spike.sum()
    
    if n_spikes == 0:
        return {}
    
    spike_coverage = ((y_true[is_spike] <= upper_bound[is_spike])).mean()
    spike_miss_rate = 1 - spike_coverage
    mean_excess = np.maximum(0, y_true[is_spike] - upper_bound[is_spike]).mean()
    avg_upper_during_spike = upper_bound[is_spike].mean()
    
    return {
        'spike_threshold': spike_threshold,
        'n_spikes': int(n_spikes),
        'spike_coverage': float(spike_coverage),
        'spike_miss_rate': float(spike_miss_rate),
        'mean_excess_error': float(mean_excess),
        'avg_upper_during_spike': float(avg_upper_during_spike),
    }
```

### 3.3 Update `replay_trace_inference.py`

**Key changes:**
1. **Extract full quantiles** from model output (not just median)
2. **Load CQR calibrators** from checkpoint
3. **Apply CQR correction** → get static intervals [q10 - q_conf, q95 + q_conf]
4. **Apply AdaptiveUpperConformal** online → dynamic upper/lower bounds
5. **Report metrics:**
   - Standard: MAE, RMSE (q50)
   - Interval: PICP, MPIW
   - Spike: spike_coverage, spike_miss_rate, mean_excess_error, avg_upper_during_spike
   - Adaptive: track alpha_u, alpha_l over time

**New CLI args:**
- `--calibrator_type` (cqr/enbpi/aci/none)
- `--adaptive_conformal` (bool)
- `--adaptive_window` (default 500)
- `--adaptive_eta` (default 0.01)
- `--spike_quantile` (default 0.95)

---

## Phase 4: Deployment Integration (CPA)

### 4.1 Update `deploy/cpa/model_builder.py`
- Load `cqr_calibrators` from checkpoint
- Initialize `AdaptiveUpperConformal` per target
- Expose `get_upper_bound_for_scaling(x)` method

### 4.2 Update CPA Autoscaler Logic
- Use `upper_bound` (not median) for replica calculation
- Log `cpa_upper_bound`, `cpa_spike_coverage` metrics

---

## File Change Summary

| Phase | File | Change Type |
|-------|------|-------------|
| 1.1 | `core/architectures/ensemble.py` | Modify `QuantileEnsembleForecaster` |
| 1.2 | `training/enbpi_train.py` | Modify `_compute_quantile_loss`, add arg |
| 2.1 | `training/enbpi_train.py` | Replace `_fit_enbpi_calibrator` with `_fit_cqr_calibrator` |
| 2.2 | `training/enbpi_train.py` | Save calibrators in checkpoint |
| 2.3 | `training/enbpi_train.py` | Add `--calibrator_type` arg |
| 3.1 | `analytics/adaptive_conformal.py` | **New file** |
| 3.2 | `analytics/spike_metrics.py` | **New file** |
| 3.3 | `analytics/replay_trace_inference.py` | Major update: quantile extraction, adaptive conformal, spike metrics |
| 4.1 | `deploy/cpa/model_builder.py` | Load calibrators, add upper-bound inference |
| 4.2 | `deploy/cpa/autoscaler.py` | Use upper bound for scaling decisions |

---

## Execution Order

```
1. Phase 1 (Model & Loss) → Retrain model
2. Phase 2 (Calibration)   → Run training with calibration (produces calibrated checkpoint)
3. Phase 3 (Evaluation)    → Run replay with new metrics
4. Phase 4 (Deployment)    → Build & deploy CPA image
```

---

## Validation Checklist

After Phase 3, the replay output should show:

```
REPLAY METRICS (pred[t] vs actual[t+5])
------------------------------------------------------------
CPU   Raw:  MSE 0.0089  MAE 0.067 (6.7%)  naive MAE 0.123 (12.3%)  delta -45.5%
CPU   Corr: MSE 0.0071  MAE 0.058 (5.8%)  delta -52.8%  (n=1200)
CPU   PICP: 94.2%  MPIW: 0.31  SpikeCov: 89.5%  SpikeMiss: 10.5%  MeanExcess: 0.042
CPU   Adaptive: alpha_u=0.032  alpha_l=0.067  (final)
Mem   ...
------------------------------------------------------------
```

**Success criteria:**
- Overall PICP ≈ 90-95% (target 1-α = 90%)
- **Spike Coverage ≥ 90%** (primary operational metric)
- Spike Miss Rate ≤ 10%
- Mean Excess Error small (upper bound close to actual spikes)
- Alpha adapts: increases during stable periods, decreases during spikes

---

## Questions for Clarification

1. **Quantile levels**: Spec uses q0.10/q0.50/q0.95; current code uses q0.05/q0.5/q0.95. Keep 0.05 or change to 0.10?
2. **Calibration horizon**: Calibrate only t+5 (as specified) or all 5 horizons?
3. **Spike threshold**: Use training 95th percentile, or fixed value (e.g., 0.8 CPU)?
4. **Adaptive window**: 500 default OK, or prefer 1000 for more stable estimates?
5. **EnbPI vs CQR**: The spec recommends CQR + adaptive upper-tail. Keep EnbPI as baseline comparison only?
6. **Deployment timeline**: Is Phase 4 needed now, or focus on Phases 1-3 for evaluation first?

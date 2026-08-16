#!/usr/bin/env python
"""Replay a deployment trace through a trained forecaster.

Loads a checkpoint (e.g. /proj/k8sautoscaledl-PG0/models/bilstm_nasa.pt),
walks an HPA-logs CSV (msname + feature columns) for one deployment in
1-minute steps, feeds each sliding window of `input_len` rows to the model,
and records the predictions (the horizon-th step of the model output, i.e.
the forecast for input_end + horizon minutes).

Input channels are derived from the checkpoint's feature_set; non-resource
features (e.g. http_mcr/providerrpc_mcr) are min-max scaled to [0,1] exactly
like preprocessing/build_windows.py so replays match training.

Predictions and per-window inference times are written to a CSV and plotted
against the actual CPU/memory load, mirroring analytics/cpa_experiment_report.py
(actual vs pred shifted by the horizon).

Usage examples:
  python analytics/replay_trace_inference.py --checkpoint /proj/k8sautoscaledl-PG0/models/bilstm_nasa.pt \
      --deployment frontend --start_hour 24 --hours 24
  python analytics/replay_trace_inference.py --checkpoint ... --deployment frontend \
      --start_hour 12 --hours 6 --simulate_live
"""

import argparse
import os
import sys
import time
from datetime import timedelta
from collections import deque

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class OnlineResidualCorrector:
    """Online residual correction using periodic batch AR re-fitting.
    
    Matches the offline batch LS performance by re-fitting the AR model
    on all available residuals at each step (or periodically).
    """
    
    def __init__(self, ar_order=2, residual_window=256, horizon=5, refit_every=1):
        self.ar_order = ar_order
        self.residual_window = residual_window
        self.horizon = horizon
        self.refit_every = refit_every
        
        self.pending = []
        self.res_cpu = []
        self.res_mem = []
        self.step_count = 0
        self.ar_weights_cpu = None
        self.ar_weights_mem = None
        
    def _refit_ar(self, residuals):
        """Fit AR model on residuals using batch least squares."""
        p = self.ar_order
        if len(residuals) < 2 * p:
            return None
        # Build design matrix: each row is [r_{t-1}, r_{t-2}, ..., r_{t-p}]
        X = np.zeros((len(residuals) - p, p))
        for i in range(p, len(residuals)):
            X[i-p, :] = residuals[i-p:i][::-1]
        y = residuals[p:]
        try:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
            return w
        except:
            return None
    
    def record(self, now, base_cpu, base_mem):
        self.pending.append({
            "time": float(now), 
            "cpu": float(base_cpu), 
            "mem": float(base_mem)
        })
        max_pending = max(2, self.horizon + 1)
        if len(self.pending) > max_pending:
            del self.pending[:-max_pending]
    
    def finalize(self, now, cpu_actual, mem_actual):
        matured = [p for p in self.pending if p["time"] + self.horizon * 60 <= now]
        if not matured:
            return
        for p in matured:
            if len(self.res_cpu) < self.residual_window:
                # First few residuals - just accumulate
                self.res_cpu.append(cpu_actual - p["cpu"])
                self.res_mem.append(mem_actual - p["mem"])
            else:
                self.res_cpu = self.res_cpu[-(self.residual_window-1):] + [cpu_actual - p["cpu"]]
                self.res_mem = self.res_mem[-(self.residual_window-1):] + [mem_actual - p["mem"]]
        self.pending = [p for p in self.pending if p["time"] + self.horizon * 60 > now]
        
        # Refit AR models
        if len(self.res_cpu) >= 2 * self.ar_order:
            self.ar_weights_cpu = self._refit_ar(self.res_cpu)
            self.ar_weights_mem = self._refit_ar(self.res_mem)
    
    def get_delta(self, target):
        p = self.ar_order
        if target == "cpu":
            w = self.ar_weights_cpu
            res_buf = self.res_cpu
        else:
            w = self.ar_weights_mem
            res_buf = self.res_mem
        
        ar = 0.0
        if w is not None and len(res_buf) >= p:
            x = np.asarray(res_buf[-p:][::-1], dtype=np.float64)
            ar = float(w @ x)
        return ar
    
    def step(self, now, pred_cpu, pred_mem, cpu_actual, mem_actual):
        self.step_count += 1
        self.finalize(now, cpu_actual, mem_actual)
        delta_cpu = self.get_delta("cpu")
        delta_mem = self.get_delta("mem")
        corrected_cpu = pred_cpu + delta_cpu
        corrected_mem = pred_mem + delta_mem
        self.record(now, pred_cpu, pred_mem)
        return corrected_cpu, corrected_mem

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from analytics.adaptive_conformal import AdaptiveUpperConformalPerTarget
    from analytics.spike_metrics import compute_spike_metrics, print_spike_metrics
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    print("Warning: conformal modules not available")

from core.models import RNNForecaster
from shared.features import (
    feature_names_for_feature_set,
    target_features_for_feature_set,
    get_feature_set,
)
from preprocessing.build_windows import _CSV_COLUMN_MAP, _CSV_COLUMN_MINMAX
from preprocessing.swt.decomposition import decompose_window
from preprocessing.swt.config import CFG as SWT_CFG

RNN_TYPES = ("lstm", "gru", "bilstm", "bigrue")
BUILDER_TYPES = ("cnn_bilstm", "dpam", "tcn", "tcn_dual", "quantile_ensemble")
DEFAULT_CSV = "/proj/k8sautoscaledl-PG0/hpa_historical_logs.csv"
DEFAULT_PLOTS_DIR = "/proj/k8sautoscaledl-PG0/analytics_out"
TARGET_COLS = ("cpu_utilization", "memory_utilization")


def parse_args():
    ap = argparse.ArgumentParser(description="Replay an HPA trace through a trained forecaster")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the model checkpoint .pt")
    ap.add_argument("--csv_path", default=DEFAULT_CSV,
                    help="HPA-logs CSV (Timestamp,Deployment,CPU,Memory; default: %(default)s)")
    ap.add_argument("--deployment", default="frontend",
                    help="Which deployment to replay (default: %(default)s)")
    ap.add_argument("--start_hour", type=float, default=0.0,
                    help="Hour of the deployment's trace to start replaying (0-based; "
                         "0 = first row of that deployment; default: %(default)s)")
    ap.add_argument("--hours", type=float, default=6.0,
                    help="How many hours of the trace to replay (default: %(default)s)")
    ap.add_argument("--input_len", type=int, default=None,
                    help="Override the window input length (default: taken from checkpoint)")
    ap.add_argument("--simulate_live", action="store_true",
                    help="Wait 1 minute between windows to mimic real-time CPA evaluation")
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--plots_dir", default=DEFAULT_PLOTS_DIR,
                    help="Directory for the predictions CSV and plots (default: %(default)s)")
    ap.add_argument("--adaptive_conformal", action="store_true",
                    help="Enable online adaptive conformal correction for upper/lower bounds")
    ap.add_argument("--adaptive_window", type=int, default=500,
                    help="Rolling window size for adaptive conformal (default: %(default)s)")
    ap.add_argument("--adaptive_eta", type=float, default=0.01,
                    help="Learning rate for adaptive alpha updates (default: %(default)s)")
    ap.add_argument("--spike_quantile", type=float, default=0.95,
                    help="Quantile threshold for spike definition (default: %(default)s)")
    ap.add_argument("--no_correction", action="store_true",
                    help="Disable residual AR correction (keep only conformal)")
    return ap.parse_args()


def _derive_rnn_from_state_dict(sd):
    input_size = sd["rnn.weight_ih_l0"].shape[1]
    hidden = sd["rnn.weight_hh_l0"].shape[1]
    rows = sd["rnn.weight_hh_l0"].shape[0]
    rnn_type = "lstm" if rows == 4 * hidden else "gru"
    num_layers = max(
        int(k[len("rnn.weight_ih_l"):])
        for k in sd
        if k.startswith("rnn.weight_ih_l") and k[len("rnn.weight_ih_l"):].isdigit()
    ) + 1
    bidirectional = any("_reverse" in k for k in sd)
    return input_size, hidden, num_layers, bidirectional, rnn_type


def _config_shim(feature_set, input_len, num_targets, pred_horizon):
    import types
    cfg = types.ModuleType("config")
    cfg.FEATURE_SET = feature_set
    cfg.INPUT_SIZE = len(feature_names_for_feature_set(feature_set))
    cfg.WINDOW_SIZE = input_len
    cfg.NUM_TARGETS = num_targets
    cfg.HIDDEN_SIZE = 128
    cfg.NUM_LAYERS = 3
    cfg.DROPOUT = 0.3
    cfg.HORIZON = pred_horizon
    sys.modules["config"] = cfg


def _build_from_builder(checkpoint, model_type, feature_set, input_len, num_targets, pred_horizon):
    ckpt_args = checkpoint.get("args", {}) or {}
    _config_shim(feature_set, input_len, num_targets, pred_horizon)
    deploy_dir = os.path.join(REPO_ROOT, "deploy", "cpa")
    if deploy_dir not in sys.path:
        sys.path.insert(0, deploy_dir)
    from model_builder import build_model
    return build_model(checkpoint, model_type)


def _resolve_feature_set_from_input_size(input_size, feature_set):
    """If checkpoint's input_size doesn't match feature_set, find matching one."""
    if input_size == len(feature_names_for_feature_set(feature_set)):
        return feature_set
    # Try to find a feature set with matching input size
    for fs in ["cpu_mem_http_rpc", "cpu_mem_both", "cpu_mem_http_rpc_replicas", "cpu"]:
        if len(feature_names_for_feature_set(fs)) == input_size:
            print(f"[INFO] Checkpoint input_size={input_size} doesn't match feature_set='{feature_set}' ({len(feature_names_for_feature_set(feature_set))} features). Using '{fs}' instead.")
            return fs
    print(f"[WARN] No feature set matches input_size={input_size}. Using '{feature_set}'.")
    return feature_set


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    model_type = checkpoint.get("model_type") or "bilstm"
    feature_set = ckpt_args.get("feature_set", "cpu_mem_both")
    input_len = int(ckpt_args.get("input_len", 128))
    pred_horizon = int(ckpt_args.get("pred_horizon", 5))
    
    # Resolve input_size from checkpoint (saved by training scripts)
    input_size = checkpoint.get("input_size")
    if input_size is None:
        input_size = len(feature_names_for_feature_set(feature_set))
    
    # Fix feature_set if input_size doesn't match
    feature_set = _resolve_feature_set_from_input_size(input_size, feature_set)
    
    num_targets = len(target_features_for_feature_set(feature_set))
    is_change_head = bool(ckpt_args.get("change_head", False) or ckpt_args.get("change_head_mem", False))
    sd = checkpoint["model_state_dict"]

    if model_type in BUILDER_TYPES:
        model = _build_from_builder(
            checkpoint, model_type, feature_set, input_len, num_targets, pred_horizon
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if is_change_head:
            sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}

        input_size, hidden, num_layers, bidirectional, rnn_type = _derive_rnn_from_state_dict(sd)
        dropout = float(hyperparams.get("dropout", ckpt_args.get("dropout", 0.1)))
        model = RNNForecaster(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            horizon=pred_horizon,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_targets=num_targets,
        )
        model.load_state_dict(sd)
        if is_change_head:
            from core.models import ChangeHeadForecaster
            inject_mask = None
            if ckpt_args.get("change_head_mem", False):
                inject_mask = [False] * num_targets
                if num_targets > 1:
                    inject_mask[-1] = True
                else:
                    inject_mask[0] = True
            model = ChangeHeadForecaster(model, inject_mask)
    model.to(device).eval()
    meta = {
        "model_type": model_type,
        "feature_set": feature_set,
        "input_len": input_len,
        "pred_horizon": pred_horizon,
        "num_targets": num_targets,
        "input_size": input_size,
    }
    return model, meta


def _feature_matrix(df_all, df_sub, feature_set):
    """Per-window input channels for a checkpoint's feature_set.

    Mirrors preprocessing/build_windows._load_csv_service_arrays: feature names
    resolve to CSV columns via _CSV_COLUMN_MAP, and columns in
    _CSV_COLUMN_MINMAX are min-max scaled to [0,1] using GLOBAL min/max over
    the full CSV (all services), so replays match training exactly.
    """
    feature_names = feature_names_for_feature_set(feature_set)
    cols = []
    for f in feature_names:
        if f not in _CSV_COLUMN_MAP:
            raise SystemExit(
                f"Feature '{f}' (feature_set={feature_set}) has no CSV column "
                f"mapping in _CSV_COLUMN_MAP"
            )
        c = _CSV_COLUMN_MAP[f]
        if c not in df_sub.columns:
            raise SystemExit(
                f"CSV is missing column '{c}' needed for feature '{f}' "
                f"(feature_set={feature_set})"
            )
        cols.append(c)

    mat = df_sub[cols].to_numpy(dtype=np.float32)
    for i, c in enumerate(cols):
        if c in _CSV_COLUMN_MINMAX:
            lo = float(df_all[c].min())
            hi = float(df_all[c].max())
            if hi - lo > 1e-12:
                mat[:, i] = (mat[:, i] - lo) / (hi - lo)
            else:
                mat[:, i] = 0.0
    return mat, cols


def _apply_swt_per_window(raw_feat: np.ndarray, feature_set: str, input_len: int) -> np.ndarray:
    """Apply SWT decomposition to each window individually.

    For each window of length input_len, decompose CPU and Memory channels
    into SWT coefficients. Returns array of shape (n_windows, input_len, n_channels).
    """
    spec = get_feature_set(feature_set)
    target_features = spec.get("targets", [spec.get("target")])
    has_mem = "memory_utilization" in target_features

    cpu_idx = 0
    mem_idx = 1 if has_mem else -1

    n_cpu_channels = SWT_CFG.SWT_LEVEL + 1
    n_mem_channels = (SWT_CFG.MEM_SWT_LEVEL + 1) if has_mem else 0
    total_channels = n_cpu_channels + n_mem_channels

    n_samples = raw_feat.shape[0]
    n_windows = n_samples - input_len + 1
    swt_windows = np.zeros((n_windows, input_len, total_channels), dtype=np.float32)

    for i in range(n_windows):
        window = raw_feat[i:i + input_len]
        cpu_ch = decompose_window(window[:, cpu_idx].astype(np.float64), SWT_CFG)
        if cpu_ch is None:
            cpu_ch = np.zeros((n_cpu_channels, input_len), dtype=np.float32)
            cpu_ch[0] = window[:, cpu_idx]
        swt_windows[i, :, :n_cpu_channels] = cpu_ch.T

        if has_mem:
            mem_ch = decompose_window(window[:, mem_idx].astype(np.float64), SWT_CFG)
            if mem_ch is None:
                mem_ch = np.zeros((n_mem_channels, input_len), dtype=np.float32)
                mem_ch[0] = window[:, mem_idx]
            swt_windows[i, :, n_cpu_channels:] = mem_ch.T

    return swt_windows


def replay(df, model, meta, raw_feat, model_feat, device, start_ts=None, end_ts=None, simulate_live=False, use_correction=True, use_adaptive_conformal=False, adaptive_window=500, adaptive_eta=0.01, spike_quantile=0.95, no_ar_correction=False, checkpoint_path=None):
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]
    num_targets = meta["num_targets"]

    if len(df) < input_len + pred_horizon:
        raise SystemExit(
            f"Deployment trace only has {len(df)} rows; need >= {input_len + pred_horizon}"
        )

    ts = df["timestamp"].to_numpy()
    n = len(df)

    # Create windows from model_feat (which is either raw_feat or SWT windows)
    if model_feat.ndim == 2:
        # Raw features: create sliding windows
        n_samples = model_feat.shape[0]
        n_windows = n_samples - input_len + 1
        model_feat_windows = np.zeros((n_windows, input_len, model_feat.shape[1]), dtype=np.float32)
        for i in range(n_windows):
            model_feat_windows[i] = model_feat[i:i + input_len]
    else:
        # Already windowed (SWT)
        model_feat_windows = model_feat

    # Warmup with first window
    warmup = torch.tensor(model_feat_windows[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        model(warmup)

    # Correction state
    ar_corrector = OnlineResidualCorrector(
        ar_order=2,
        residual_window=256,
        horizon=pred_horizon,
        refit_every=1,
    ) if use_correction and not no_ar_correction else None

    # Load CQR calibrators from checkpoint
    cqr_calibrators = None
    if use_adaptive_conformal and CONFORMAL_AVAILABLE:
        ckpt_full = torch.load(args.checkpoint if 'args' in locals() else '', map_location="cpu") if False else None
        # We'll load it properly below

    # Adaptive conformal for upper/lower bounds (CPU-focused, optional Memory)
    adaptive_cal = AdaptiveUpperConformalPerTarget(
        num_targets=num_targets,
        window_size=adaptive_window,
        alpha=0.05,
        eta=adaptive_eta,
        alpha_min=0.01,
        alpha_max=0.20,
    ) if use_adaptive_conformal and CONFORMAL_AVAILABLE else None

    # CQR static corrections (from checkpoint)
    cqr_q_conf = None
    if use_adaptive_conformal and CONFORMAL_AVAILABLE and checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if 'cqr_calibrators' in ckpt:
                cqr_calibrators = ckpt['cqr_calibrators']
                cqr_q_conf = {}
                for t_idx, cal in cqr_calibrators.items():
                    cqr_q_conf[t_idx] = cal.get('q_conf', 0.0)
                pass
            else:
                pass
        except Exception as e:
            print(f"Warning: Could not load CQR calibrators: {e}")

    rows = []
    t_total0 = time.perf_counter()
    for idx in range(input_len - 1, n):
        if start_ts is not None and ts[idx] < start_ts:
            continue
        if end_ts is not None and ts[idx] >= end_ts:
            break
        window_idx = idx - input_len + 1
        window = torch.tensor(model_feat_windows[window_idx],
                              dtype=torch.float32, device=device).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(window)
        preds = out[0] if isinstance(out, tuple) else out
        dt = time.perf_counter() - t0
        
        # Handle quantile ensemble output (B, H, num_targets, num_quantiles)
        if preds.dim() == 4:
            q10 = preds[0, -1, :, 0].cpu().numpy()  # (num_targets,)
            q50 = preds[0, -1, :, 1].cpu().numpy()
            q95 = preds[0, -1, :, 2].cpu().numpy()
        else:
            # Fallback for non-quantile models: use deterministic preds
            p = torch.round(preds[0, -1] * 100) / 100
            q50 = p.cpu().numpy()
            q10 = q50.copy()
            q95 = q50.copy()

        pred_cpu = float(np.round(q50[0] * 100) / 100)
        pred_mem = float(np.round(q50[1] * 100) / 100) if num_targets > 1 else float("nan")

        # Get adaptive conformal bounds (CPU upper = scaling-safe)
        upper_cpu = pred_cpu
        lower_cpu = pred_cpu
        upper_mem = pred_mem
        lower_mem = pred_mem
        if adaptive_cal is not None:
            # Apply CQR static correction first
            if cqr_q_conf is not None:
                q_conf_cpu = cqr_q_conf.get(0, 0.0)
                q_conf_mem = cqr_q_conf.get(1, 0.0)
                q10_corrected = q10.copy()
                q95_corrected = q95.copy()
                q10_corrected[0] = q10[0] - q_conf_cpu
                q95_corrected[0] = q95[0] + q_conf_cpu
                if num_targets > 1:
                    q10_corrected[1] = q10[1] - q_conf_mem
                    q95_corrected[1] = q95[1] + q_conf_mem
            else:
                q10_corrected = q10
                q95_corrected = q95
            
            lower_cpu, upper_cpu = adaptive_cal.get_interval(q10_corrected, q95_corrected)
            if num_targets > 1:
                # For memory we can use same calibrators but they'll adapt independently
                _, upper_mem = adaptive_cal.get_interval(q10_corrected, q95_corrected)

        now_ts = pd.Timestamp(ts[idx]).timestamp()
        cpu_actual = float(raw_feat[idx, 0])
        mem_actual = float(raw_feat[idx, 1]) if num_targets > 1 else 0.0

        # Apply AR correction
        if ar_corrector is not None:
            ar_corr_cpu, ar_corr_mem = ar_corrector.step(now_ts, pred_cpu, pred_mem, cpu_actual, mem_actual)
        else:
            ar_corr_cpu, ar_corr_mem = pred_cpu, pred_mem

        # Update adaptive conformal with observed values
        if adaptive_cal is not None:
            adaptive_cal.update(
                y=np.array([cpu_actual, mem_actual]) if num_targets > 1 else np.array([cpu_actual, 0.0]),
                q10=q10,
                q95=q95,
            )

        # Extract scalars for CPU (target 0)
        def to_scalar(x):
            if isinstance(x, (np.ndarray, list)):
                arr = np.asarray(x)
                return float(arr.flat[0]) if arr.size > 0 else float("nan")
            return float(x)
        
        lower_cpu_scalar = to_scalar(lower_cpu)
        upper_cpu_scalar = to_scalar(upper_cpu)
        lower_mem_scalar = to_scalar(lower_mem) if num_targets > 1 else float("nan")
        upper_mem_scalar = to_scalar(upper_mem) if num_targets > 1 else float("nan")

        rows.append(
            (ts[idx], cpu_actual, mem_actual, 
             pred_cpu, pred_mem,
             ar_corr_cpu, ar_corr_mem,
             lower_cpu_scalar, upper_cpu_scalar, lower_mem_scalar, upper_mem_scalar,
             dt)
        )
        if simulate_live:
            time.sleep(max(0.0, 60.0 - dt))
    t_total = time.perf_counter() - t_total0

    cols = ["timestamp", "cpu", "memory", "pred_cpu", "pred_mem", 
            "corr_cpu", "corr_mem",
            "lower_cpu", "upper_cpu", "lower_mem", "upper_mem",
            "inference_time_s"]
    res = pd.DataFrame(rows, columns=cols)
    return res, t_total


def print_metrics(res, pred_horizon, spike_quantile=0.95):
    print("\n" + "=" * 60)
    print("REPLAY METRICS (pred[t] vs actual[t+%d])" % pred_horizon)
    print("-" * 60)
    
    # Determine which rows have quantile predictions
    has_quantiles = "lower_cpu" in res.columns and res["upper_cpu"].notna().any()
    
    for i, (label, acol, pcol, ccol) in enumerate([
        (("CPU", "cpu", "pred_cpu", "corr_cpu")),
        ("Mem", "memory", "pred_mem", "corr_mem") if "pred_mem" in res.columns else None
    ]):
        if label is None:
            continue
        if res[pcol].isna().all():
            print(f"{label:5s}  no predictions for this target")
            continue
        frame = pd.DataFrame(
            {"y": res[acol].shift(-pred_horizon), "a": res[acol], "pred": res[pcol], "corr": res[ccol]}
        ).dropna()
        if len(frame) < pred_horizon:
            print(f"{label:5s}  too few aligned rows ({len(frame)})")
            continue
        mse = float(((frame["y"] - frame["pred"]) ** 2).mean())
        mae = float((frame["y"] - frame["pred"]).abs().mean())
        corr_mse = float(((frame["y"] - frame["corr"]) ** 2).mean())
        corr_mae = float((frame["y"] - frame["corr"]).abs().mean())
        naive_mae = float((frame["y"] - frame["a"]).abs().mean())
        d = (mae - naive_mae) / naive_mae * 100 if naive_mae > 0 else float("nan")
        d_corr = (corr_mae - naive_mae) / naive_mae * 100 if naive_mae > 0 else float("nan")
        print(f"{label:5s}  Raw:  MSE {mse:.5f}  MAE {mae:.5f} ({mae*100:.2f}%)  naive MAE {naive_mae:.5f} ({naive_mae*100:.2f}%)  delta {d:+.1f}%")
        print(f"{label:5s}  Corr: MSE {corr_mse:.5f}  MAE {corr_mae:.5f} ({corr_mae*100:.2f}%)  delta {d_corr:+.1f}%  (n={len(frame)})")
        
        # Spike metrics if quantile predictions available
        if has_quantiles and i == 0:
            tgt_idx = 0
            y_arr = res[acol].iloc[pred_horizon:].values
            lower = res[f"lower_{acol}"].iloc[pred_horizon:].values
            upper = res[f"upper_{acol}"].iloc[pred_horizon:].values
            if len(y_arr) > 0 and (upper > lower).any():
                if CONFORMAL_AVAILABLE:
                    try:
                        metrics_cpu = compute_spike_metrics(y_arr, lower, upper, spike_quantile=spike_quantile)
                        print_spike_metrics(metrics_cpu, target_name="CPU(Conformal)")
                    except Exception as e:
                        print(f"  Spike metrics error: {e}")
    inf = res["inference_time_s"]
    print("-" * 60)
    print(f"Windows: {len(res)}  |  avg inference {inf.mean()*1e3:.2f} ms  "
          f"|  p95 inference {inf.quantile(0.95)*1e3:.2f} ms")
    if has_quantiles and "upper_cpu" in res.columns:
        final_alphas = "N/A"
        print("=" * 60)
    print("=" * 60)


def _style_time_axis(ax, span_hours):
    if span_hours > 18:
        loc = mdates.HourLocator(interval=2)
    else:
        loc = mdates.MinuteLocator(interval=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_predictions(res, deployment, pred_horizon, plots_dir, num_targets):
    span_hours = (res["timestamp"].iloc[-1] - res["timestamp"].iloc[0]).total_seconds() / 3600.0

    panels = []
    if num_targets > 1:
        panels = [
            ("CPU", "cpu", "pred_cpu", "corr_cpu", "CPU Utilization (fraction of core)"),
            ("Memory", "memory", "pred_mem", "corr_mem", "Memory Utilization (fraction of request)"),
        ]
    else:
        panels = [("CPU", "cpu", "pred_cpu", "corr_cpu", "CPU Utilization (fraction of core)")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (title, acol, pcol, ccol, ylabel) in zip(axes, panels):
        actual = np.array(res[acol], dtype=float)
        pred = np.array(res[pcol], dtype=float)
        corr = np.array(res[ccol], dtype=float)
        pred[~np.isfinite(pred)] = np.nan
        corr[~np.isfinite(corr)] = np.nan
        ax.plot(res["timestamp"], actual, label="Actual", color="blue", alpha=0.6)
        ax.plot(res["timestamp"], pd.Series(pred).shift(pred_horizon).to_numpy(),
                label="Predicted (t+%d)" % pred_horizon, color="orange", linestyle="--", alpha=0.9)
        ax.plot(res["timestamp"], pd.Series(corr).shift(pred_horizon).to_numpy(),
                label="Corrected (t+%d)" % pred_horizon, color="green", linestyle=":", alpha=0.9)

        vmax = max(np.nanmax(actual), np.nanmax(pred), np.nanmax(corr)) if np.any(np.isfinite(corr)) else max(np.nanmax(actual), np.nanmax(pred))
        ax.set_ylim(0, max(1.0, vmax * 1.1))
        ax.set_title(f"Deployment: {deployment} — {title}", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        _style_time_axis(ax, span_hours)

    fig.suptitle(f"Replay inference — {deployment} ({res['timestamp'].iloc[0]} to {res['timestamp'].iloc[-1]})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(plots_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(plots_dir, f"replay_{deployment}_{stamp}.png")
    csv_path = os.path.join(plots_dir, f"replay_predictions_{deployment}_{stamp}.csv")
    fig.savefig(png_path, dpi=300)
    res.to_csv(csv_path, index=False)
    print(f"Plot saved to {png_path}")
    print(f"Predictions saved to {csv_path}")


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise SystemExit(f"CSV {args.csv_path} has no 'timestamp'/'Timestamp' column")

    id_col = "msname" if "msname" in df.columns else "Deployment"
    sub = df[df[id_col] == args.deployment].sort_values("timestamp").reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"Deployment {args.deployment!r} not found in {args.csv_path}")

    t_start = sub["timestamp"].iloc[0] + timedelta(hours=args.start_hour)
    t_end = t_start + timedelta(hours=args.hours)
    sel = sub[(sub["timestamp"] >= t_start) & (sub["timestamp"] < t_end)].reset_index(drop=True)
    if sel.empty:
        raise SystemExit(
            f"No rows for {args.deployment} in [{t_start}, {t_end}) — start_hour {args.start_hour} "
            f"exceeds the trace length ({sub['timestamp'].iloc[0]} .. {sub['timestamp'].iloc[-1]})"
        )

    model, meta = load_model(args.checkpoint, device)
    if args.input_len is not None:
        meta["input_len"] = args.input_len
    feat_raw, feat_cols = _feature_matrix(df, sub, meta["feature_set"])
    
    # Check if model was trained with SWT preprocessing
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}
    preprocess_approach = ckpt_args.get("preprocess_approach", "none")
    
    if preprocess_approach == "swt":
        feat_swt_windows = _apply_swt_per_window(feat_raw, meta["feature_set"], meta["input_len"])
        print(f"Raw feature shape: {feat_raw.shape}, SWT windows shape: {feat_swt_windows.shape}")
        model_feat = feat_swt_windows
    else:
        print(f"Raw feature shape: {feat_raw.shape} (no SWT preprocessing)")
        model_feat = feat_raw
    
    res, t_total = replay(
        sub, model, meta, feat_raw, model_feat, device,
        start_ts=t_start, end_ts=t_end,
        simulate_live=args.simulate_live,
        use_correction=not args.no_correction,
        use_adaptive_conformal=args.adaptive_conformal,
        adaptive_window=args.adaptive_window,
        adaptive_eta=args.adaptive_eta,
        spike_quantile=args.spike_quantile,
        checkpoint_path=args.checkpoint,
    )
    if res.empty:
        raise SystemExit(
            f"No window can end within [{t_start}, {t_end}): the first "
            f"{meta['input_len']} minutes of the trace are needed as context, so "
            f"predictions start at {sub['timestamp'].iloc[meta['input_len'] - 1]}.\n"
            f"Use a later --start_hour (>= {meta['input_len'] / 60:.2f}) or more --hours."
        )

    print_metrics(res, meta["pred_horizon"], spike_quantile=args.spike_quantile)
    print(f"\nReplay wall time: {t_total:.2f}s "
          f"({'real-time' if args.simulate_live else 'fast-forward (add --simulate_live for 1-min pacing)'})")

    plot_predictions(res, args.deployment, meta["pred_horizon"], args.plots_dir, meta["num_targets"])


if __name__ == "__main__":
    main()

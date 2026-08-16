#!/usr/bin/env python
"""Replay a deployment trace through a trained forecaster.

Loads a checkpoint, walks an HPA-logs CSV for one deployment in 1-minute steps,
feeds each sliding window to the model, and records predictions.

Online conformal prediction: frozen quantile model + adaptive conformal state.
Horizon-offset feedback: actual y(t+5) updates conformal state 5 steps later.
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
SPIKE_THRESHOLD_TRAIN = 0.6099  # Q0.95 of CPU t+5 targets from training


def parse_args():
    ap = argparse.ArgumentParser(description="Replay an HPA trace through a trained forecaster")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the model checkpoint .pt")
    ap.add_argument("--csv_path", default=DEFAULT_CSV,
                    help="HPA-logs CSV (default: %(default)s)")
    ap.add_argument("--deployment", default="frontend",
                    help="Which deployment to replay (default: %(default)s)")
    ap.add_argument("--start_hour", type=float, default=0.0,
                    help="Hour to start replaying (0-based; default: %(default)s)")
    ap.add_argument("--hours", type=float, default=6.0,
                    help="How many hours to replay (default: %(default)s)")
    ap.add_argument("--input_len", type=int, default=None,
                    help="Override window input length (default: from checkpoint)")
    ap.add_argument("--simulate_live", action="store_true",
                    help="Wait 1 minute between windows to mimic real-time CPA evaluation")
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--plots_dir", default=DEFAULT_PLOTS_DIR,
                    help="Directory for plots (default: %(default)s)")
    ap.add_argument("--adaptive_conformal", action="store_true",
                    help="Enable online adaptive conformal (upper/lower bounds)")
    ap.add_argument("--warmup_windows", type=int, default=500,
                    help="Number of warmup windows to fill conformal buffers (default: %(default)s)")
    ap.add_argument("--adaptive_window", type=int, default=500,
                    help="Rolling window size for adaptive conformal (default: %(default)s)")
    ap.add_argument("--adaptive_eta", type=float, default=0.01,
                    help="Learning rate for adaptive alpha (default: %(default)s)")
    ap.add_argument("--spike_threshold", type=float, default=SPIKE_THRESHOLD_TRAIN,
                    help=f"Spike threshold (default: {SPIKE_THRESHOLD_TRAIN} = train Q0.95)")
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
    if input_size == len(feature_names_for_feature_set(feature_set)):
        return feature_set
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
    input_size = checkpoint.get("input_size")
    if input_size is None:
        input_size = len(feature_names_for_feature_set(feature_set))
    feature_set = _resolve_feature_set_from_input_size(input_size, feature_set)
    num_targets = len(target_features_for_feature_set(feature_set))
    is_change_head = bool(ckpt_args.get("change_head", False) or ckpt_args.get("change_head_mem", False))
    sd = checkpoint["model_state_dict"]
    if model_type in BUILDER_TYPES:
        model = _build_from_builder(checkpoint, model_type, feature_set, input_len, num_targets, pred_horizon)
        model.load_state_dict(sd)
    else:
        if is_change_head:
            sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        input_size, hidden, num_layers, bidirectional, rnn_type = _derive_rnn_from_state_dict(sd)
        dropout = float(hyperparams.get("dropout", ckpt_args.get("dropout", 0.1)))
        model = RNNForecaster(
            input_size=input_size, hidden_size=hidden, num_layers=num_layers,
            dropout=dropout, horizon=pred_horizon, rnn_type=rnn_type,
            bidirectional=bidirectional, num_targets=num_targets,
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
    meta = {"model_type": model_type, "feature_set": feature_set, "input_len": input_len,
            "pred_horizon": pred_horizon, "num_targets": num_targets, "input_size": input_size}
    return model, meta


def _feature_matrix(df_all, df_sub, feature_set):
    feature_names = feature_names_for_feature_set(feature_set)
    cols = []
    for f in feature_names:
        if f not in _CSV_COLUMN_MAP:
            raise SystemExit(f"Feature '{f}' has no CSV column mapping")
        c = _CSV_COLUMN_MAP[f]
        if c not in df_sub.columns:
            raise SystemExit(f"CSV missing column '{c}' needed for feature '{f}'")
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


def _apply_swt_per_window(raw_feat, feature_set, input_len):
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


def replay(df, model, meta, raw_feat, model_feat, device,
           start_ts=None, end_ts=None, simulate_live=False,
           use_conformal=False, warmup_windows=500,
           adaptive_window=500, adaptive_eta=0.01,
           spike_threshold=0.6099, checkpoint_path=None):
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]
    num_targets = meta["num_targets"]

    if len(df) < input_len + pred_horizon:
        raise SystemExit(f"Deployment trace only has {len(df)} rows; need >= {input_len + pred_horizon}")

    ts = df["timestamp"].to_numpy()
    n = len(df)

    if model_feat.ndim == 2:
        n_samples = model_feat.shape[0]
        n_windows = n_samples - input_len + 1
        model_feat_windows = np.zeros((n_windows, input_len, model_feat.shape[1]), dtype=np.float32)
        for i in range(n_windows):
            model_feat_windows[i] = model_feat[i:i + input_len]
    else:
        model_feat_windows = model_feat

    # Warmup
    warmup = torch.tensor(model_feat_windows[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        model(warmup)

    # Initialize conformal state (CPU-only for scaling)
    adaptive_cal = AdaptiveUpperConformalPerTarget(
        num_targets=num_targets,
        window_size=adaptive_window,
        alpha=0.05,
        eta=adaptive_eta,
        alpha_min=0.01,
        alpha_max=0.20,
    ) if use_conformal and CONFORMAL_AVAILABLE else None

    # Pending queue for horizon-offset feedback
    pending = []

    rows = []
    t_total0 = time.perf_counter()
    warmup_count = 0
    in_warmup = use_conformal and (warmup_windows > 0)

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

        if preds.dim() == 4:
            q10 = preds[0, -1, :, 0].cpu().numpy()
            q50 = preds[0, -1, :, 1].cpu().numpy()
            q95 = preds[0, -1, :, 2].cpu().numpy()
        else:
            p = torch.round(preds[0, -1] * 100) / 100
            q50 = p.cpu().numpy()
            q10 = q50.copy()
            q95 = q50.copy()

        pred_cpu = float(np.round(q50[0] * 100) / 100)
        pred_mem = float(np.round(q50[1] * 100) / 100) if num_targets > 1 else float("nan")

        lower_cpu, upper_cpu = pred_cpu, pred_cpu
        lower_mem, upper_mem = pred_mem, pred_mem
        if adaptive_cal is not None:
            lower_cpu, upper_cpu = adaptive_cal.get_interval(q10, q95)
            if num_targets > 1:
                lower_mem, upper_mem = adaptive_cal.get_interval(q10, q95)

        now_ts = pd.Timestamp(ts[idx]).timestamp()
        cpu_actual = float(raw_feat[idx, 0])
        mem_actual = float(raw_feat[idx, 1]) if num_targets > 1 else 0.0

        # Warmup phase: fill conformal buffers (no metrics, just update state)
        if in_warmup and warmup_count < warmup_windows:
            if adaptive_cal is not None:
                adaptive_cal.states["cpu"].update(float(cpu_actual), float(q10[0]), float(q95[0]))
                if num_targets > 1:
                    adaptive_cal.states["memory"].update(float(mem_actual), float(q10[1]), float(q95[1]))
            warmup_count += 1
            # Still record for completeness but mark as warmup
            def to_scalar(x, target_idx=0):
                if isinstance(x, (np.ndarray, list)):
                    arr = np.asarray(x)
                    return float(arr.flat[target_idx]) if arr.size > target_idx else float("nan")
                return float(x)
            lc_scalar = to_scalar(lower_cpu, 0)
            uc_scalar = to_scalar(upper_cpu, 0)
            lm_scalar = to_scalar(lower_mem, 1) if num_targets > 1 else float("nan")
            um_scalar = to_scalar(upper_mem, 1) if num_targets > 1 else float("nan")
            rows.append((ts[idx], cpu_actual, mem_actual, pred_cpu, pred_mem,
                        lc_scalar, uc_scalar, lm_scalar, um_scalar, dt, True))
            if simulate_live:
                time.sleep(max(0.0, 60.0 - dt))
            t_total = time.perf_counter() - t_total0
            cols = ["timestamp", "cpu", "memory", "pred_cpu", "pred_mem",
                    "lower_cpu", "upper_cpu", "lower_mem", "upper_mem",
                    "inference_time_s", "warmup"]
            res = pd.DataFrame(rows, columns=cols)
            continue

        # After warmup: horizon-offset feedback
        if in_warmup and warmup_count >= warmup_windows:
            in_warmup = False
            print(f"[INFO] Warmup complete ({warmup_windows} windows). Starting online test.")

        # Horizon-offset: actual y(t+5) updates conformal state
        # At step idx (time t), we predict for t+5.
        # The actual for t+5 arrives at idx+5, so we store current pred
        # and update state when we reach idx+5.
        # For online test: store pending prediction
        if not in_warmup:
            pending.append({
                "idx": idx,
                "ts": now_ts,
                "q10": q10.copy(),
                "q50": q50.copy(),
                "q95": q95.copy(),
            })

        # Update conformal state from matured pending predictions
        # (actuals that are now 5 steps old)
        matured = [p for p in pending if p["idx"] <= idx - pred_horizon]
        if matured and adaptive_cal is not None:
            for p in matured:
                actual_idx = int(p["idx"] + pred_horizon)
                if actual_idx < n:
                    act_cpu = float(raw_feat[actual_idx, 0])
                    act_mem = float(raw_feat[actual_idx, 1]) if num_targets > 1 else 0.0
                    adaptive_cal.states["cpu"].update(act_cpu, p["q10"][0], p["q95"][0])
                    if num_targets > 1:
                        adaptive_cal.states["memory"].update(act_mem, p["q10"][1], p["q95"][1])
            pending = [p for p in pending if p["idx"] > idx - pred_horizon]

        # Extract scalars for storage (arrays have shape (num_targets,))
        def to_scalar(x, target_idx=0):
            if isinstance(x, (np.ndarray, list)):
                arr = np.asarray(x)
                return float(arr.flat[target_idx]) if arr.size > target_idx else float("nan")
            return float(x)
        
        lower_cpu_scalar = to_scalar(lower_cpu, 0)
        upper_cpu_scalar = to_scalar(upper_cpu, 0)
        lower_mem_scalar = to_scalar(lower_mem, 1) if num_targets > 1 else float("nan")
        upper_mem_scalar = to_scalar(upper_mem, 1) if num_targets > 1 else float("nan")

        rows.append((ts[idx], cpu_actual, mem_actual, pred_cpu, pred_mem,
                     lower_cpu_scalar, upper_cpu_scalar, lower_mem_scalar, upper_mem_scalar, dt, False))
        if simulate_live:
            time.sleep(max(0.0, 60.0 - dt))
    t_total = time.perf_counter() - t_total0

    cols = ["timestamp", "cpu", "memory", "pred_cpu", "pred_mem",
            "lower_cpu", "upper_cpu", "lower_mem", "upper_mem",
            "inference_time_s", "warmup"]
    res = pd.DataFrame(rows, columns=cols)
    return res, t_total, adaptive_cal


def print_metrics(res, pred_horizon, spike_threshold=0.6099):
    print("\n" + "=" * 60)
    print("REPLAY METRICS (pred[t] vs actual[t+%d])" % pred_horizon)
    print("-" * 60)

    # Skip warmup rows
    if "warmup" in res.columns:
        test_res = res[res["warmup"] == False].copy()
        n_warmup = int((res["warmup"] == True).sum())
        print(f"[INFO] Skipped {n_warmup} warmup windows")
    else:
        test_res = res.copy()

    has_conformal = ("lower_cpu" in test_res.columns and 
                     "upper_cpu" in test_res.columns and 
                     float((test_res["upper_cpu"] - test_res["lower_cpu"]).abs().sum()) > 0)

    for i, (label, acol, pcol, lcol, ucol) in enumerate([
        ("CPU", "cpu", "pred_cpu", "lower_cpu", "upper_cpu"),
        ("Mem", "memory", "pred_mem", "lower_mem", "upper_mem"),
    ]):
        if "pred_mem" not in test_res.columns and i > 0:
            continue
        if test_res[pcol].isna().all():
            print(f"{label:5s}  no predictions")
            continue

        y_arr = test_res[acol].iloc[pred_horizon:].values
        pred_arr = test_res[pcol].iloc[pred_horizon:].values
        lower = test_res[lcol].iloc[pred_horizon:].values
        upper = test_res[ucol].iloc[pred_horizon:].values

        if len(y_arr) == 0:
            continue

        mse = float(((y_arr - pred_arr) ** 2).mean())
        mae = float(np.abs(y_arr - pred_arr).mean())
        naive_mae = float(np.abs(y_arr - test_res[acol].iloc[pred_horizon - 1:-1].values).mean())
        d = (mae - naive_mae) / naive_mae * 100 if naive_mae > 0 else float("nan")
        print(f"{label:5s}  MSE {mse:.5f}  MAE {mae:.5f} ({mae*100:.2f}%)  naive MAE {naive_mae:.4f}  delta {d:+.1f}%")

        if has_conformal and i == 0:
            in_interval = (y_arr >= lower) & (y_arr <= upper)
            picp = float(in_interval.mean())
            mpiw = float((upper - lower).mean())
            print(f"{label:5s}  PICP: {picp:.2%}  MPIW: {mpiw:.4f}")

            if CONFORMAL_AVAILABLE:
                try:
                    metrics_cpu = compute_spike_metrics(y_arr, lower, upper,
                                                        spike_threshold=spike_threshold)
                    print_spike_metrics(metrics_cpu, target_name="CPU(Conformal)")
                except Exception as e:
                    print(f"  Spike metrics error: {e}")

    inf = test_res["inference_time_s"]
    print("-" * 60)
    print(f"Test windows: {len(test_res)} (warmup: {n_warmup if 'warmup' in res.columns else 0})")
    print(f"Windows: {len(res)}  |  avg inference {inf.mean()*1e3:.2f} ms  "
          f"|  p95 inference {inf.quantile(0.95)*1e3:.2f} ms")
    print("=" * 60)


def _style_time_axis(ax, span_hours):
    if span_hours > 18:
        loc = mdates.HourLocator(interval=2)
    else:
        loc = mdates.MinuteLocator(interval=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_predictions(res, deployment, pred_horizon, plots_dir, num_targets, use_conformal=False):
    span_hours = (res["timestamp"].iloc[-1] - res["timestamp"].iloc[0]).total_seconds() / 3600.0

    panels = []
    if num_targets > 1:
        panels = [
            ("CPU", "cpu", "pred_cpu", "lower_cpu", "upper_cpu", "CPU Utilization (fraction of core)"),
            ("Memory", "memory", "pred_mem", "lower_mem", "upper_mem", "Memory Utilization (fraction of request)"),
        ]
    else:
        panels = [("CPU", "cpu", "pred_cpu", "lower_cpu", "upper_cpu", "CPU Utilization (fraction of core)")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (title, acol, pcol, lcol, ucol, ylabel) in zip(axes, panels):
        actual = np.array(res[acol], dtype=float)
        pred = np.array(res[pcol], dtype=float)
        lower = np.array(res[lcol], dtype=float)
        upper = np.array(res[ucol], dtype=float)
        pred[~np.isfinite(pred)] = np.nan

        ax.plot(res["timestamp"], actual, label="Actual", color="blue", alpha=0.6)
        ax.plot(res["timestamp"], pd.Series(pred).shift(pred_horizon).to_numpy(),
                label="Predicted (t+%d)" % pred_horizon, color="orange", linestyle="-", alpha=0.9)
        ax.fill_between(res["timestamp"],
                        pd.Series(lower).shift(pred_horizon).to_numpy(),
                        pd.Series(upper).shift(pred_horizon).to_numpy(),
                        color="gray", alpha=0.2, label="Conformal Interval")

        if "warmup" in res.columns:
            warmup_mask = res["warmup"].values
            if warmup_mask.any():
                ax.axvspan(res["timestamp"].iloc[0], res["timestamp"][warmup_mask].iloc[-1],
                           alpha=0.1, color="yellow", label="Warmup")

        vmax = max(np.nanmax(actual), np.nanmax(pred), np.nanmax(upper))
        ax.set_ylim(0, max(1.0, vmax * 1.1))
        ax.set_title(f"Deployment: {deployment} — {title}", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        _style_time_axis(ax, span_hours)

    fig.suptitle(f"Replay inference — {deployment} ({res['timestamp'].iloc[0]} to {res['timestamp'].iloc[-1]}) | {'Adaptive Conformal' if use_conformal else 'Raw'}",
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
            f"exceeds trace length ({sub['timestamp'].iloc[0]} .. {sub['timestamp'].iloc[-1]})"
        )

    model, meta = load_model(args.checkpoint, device)
    if args.input_len is not None:
        meta["input_len"] = args.input_len
    feat_raw, feat_cols = _feature_matrix(df, sub, meta["feature_set"])

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

    res, t_total, conformal_state = replay(
        sub, model, meta, feat_raw, model_feat, device,
        start_ts=t_start, end_ts=t_end,
        simulate_live=args.simulate_live,
        use_conformal=args.adaptive_conformal,
        warmup_windows=args.warmup_windows if args.adaptive_conformal else 0,
        adaptive_window=args.adaptive_window,
        adaptive_eta=args.adaptive_eta,
        spike_threshold=args.spike_threshold,
        checkpoint_path=args.checkpoint,
    )
    if res.empty:
        raise SystemExit(
            f"No window can end within [{t_start}, {t_end}): first "
            f"{meta['input_len']} minutes needed as context, so "
            f"predictions start at {sub['timestamp'].iloc[meta['input_len'] - 1]}.\n"
            f"Use later --start_hour or more --hours."
        )

    # Filter test rows (after warmup)
    if "warmup" in res.columns:
        test_res = res[res["warmup"] == False]
    else:
        test_res = res

    if not test_res.empty:
        print_metrics(res, meta["pred_horizon"], spike_threshold=args.spike_threshold)
    print(f"\nReplay wall time: {t_total:.2f}s "
          f"({'real-time' if args.simulate_live else 'fast-forward (add --simulate_live for 1-min pacing)'})")

    plot_predictions(res, args.deployment, meta["pred_horizon"], args.plots_dir, meta["num_targets"], use_conformal=args.adaptive_conformal)


if __name__ == "__main__":
    main()
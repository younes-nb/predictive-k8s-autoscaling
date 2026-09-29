#!/usr/bin/env python

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
BUILDER_TYPES = ("cnn_bilstm", "dpam", "quantile_ensemble")
DEFAULT_CSV = "/proj/k8sautoscaledl-PG0/hpa_historical_logs.csv"
DEFAULT_PLOTS_DIR = "/proj/k8sautoscaledl-PG0/analytics_out"
TARGET_COLS = ("cpu_utilization", "memory_utilization")


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


def _apply_swt_per_window(raw_feat, feature_set, input_len, swt_level=None, mem_swt_level=None):
    from dataclasses import replace
    spec = get_feature_set(feature_set)
    feature_names = spec["features"]

    feature_cfgs = []
    for f in feature_names:
        lvl = swt_level if swt_level is not None else SWT_CFG.SWT_LEVEL
        if f == "memory_utilization" and mem_swt_level is not None:
            lvl = mem_swt_level
        feature_cfgs.append(replace(SWT_CFG, SWT_LEVEL=lvl))

    total_channels = sum(cfg.SWT_LEVEL + 1 for cfg in feature_cfgs)
    n_samples = raw_feat.shape[0]
    n_windows = n_samples - input_len + 1
    swt_windows = np.zeros((n_windows, input_len, total_channels), dtype=np.float32)

    for i in range(n_windows):
        window = raw_feat[i:i + input_len]
        ch_offset = 0
        for feat_idx, cfg in enumerate(feature_cfgs):
            n_ch = cfg.SWT_LEVEL + 1
            ch = decompose_window(window[:, feat_idx].astype(np.float64), cfg)
            if ch is None:
                ch = np.zeros((n_ch, input_len), dtype=np.float32)
                ch[0] = window[:, feat_idx]
            swt_windows[i, :, ch_offset:ch_offset + n_ch] = ch.T
            ch_offset += n_ch
    return swt_windows


def replay(df, model, meta, raw_feat, model_feat, device,
           start_ts=None, end_ts=None, simulate_live=False,
           checkpoint_path=None):
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

    warmup = torch.tensor(model_feat_windows[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        model(warmup)

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

        if preds.dim() == 4:
            q50 = preds[0, -1, :, 1].cpu().numpy()
        else:
            if preds.dim() == 3:
                p = torch.round(preds[0, -1] * 100) / 100
            else:
                p = torch.round(preds[0] * 100) / 100
            q50 = p.cpu().numpy()

        pred_cpu = float(np.round(q50[0] * 100) / 100)
        pred_mem = float(np.round(q50[1] * 100) / 100) if num_targets > 1 else float("nan")

        lower_cpu, upper_cpu = pred_cpu, pred_cpu
        lower_mem, upper_mem = pred_mem, pred_mem

        cpu_actual = float(raw_feat[idx, 0])
        mem_actual = float(raw_feat[idx, 1]) if num_targets > 1 else 0.0

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
                     lower_cpu_scalar, upper_cpu_scalar, lower_mem_scalar, upper_mem_scalar, dt))
        if simulate_live:
            time.sleep(max(0.0, 60.0 - dt))
    t_total = time.perf_counter() - t_total0

    cols = ["timestamp", "cpu", "memory", "pred_cpu", "pred_mem",
            "lower_cpu", "upper_cpu", "lower_mem", "upper_mem",
            "inference_time_s"]
    res = pd.DataFrame(rows, columns=cols)
    return res, t_total


def print_metrics(res, pred_horizon):
    print("\n" + "=" * 60)
    print("REPLAY METRICS (pred[t] vs actual[t+%d])" % pred_horizon)
    print("-" * 60)

    test_res = res.copy()

    for i, (label, acol, pcol) in enumerate([
        ("CPU", "cpu", "pred_cpu"),
        ("Mem", "memory", "pred_mem"),
    ]):
        if "pred_mem" not in test_res.columns and i > 0:
            continue
        if test_res[pcol].isna().all():
            print(f"{label:5s}  no predictions")
            continue

        y_arr = test_res[acol].iloc[pred_horizon:].values
        pred_arr = test_res[pcol].iloc[:-pred_horizon].values

        if len(y_arr) == 0:
            continue

        mse = float(((y_arr - pred_arr) ** 2).mean())
        mae = float(np.abs(y_arr - pred_arr).mean())
        naive_mae = float(np.abs(y_arr - test_res[acol].iloc[:-pred_horizon].values).mean())
        d = (mae - naive_mae) / naive_mae * 100 if naive_mae > 0 else float("nan")
        print(f"{label:5s}  MSE {mse:.5f}  MAE {mae:.5f} ({mae*100:.2f}%)  naive MAE {naive_mae:.4f}  delta {d:+.1f}%")

    inf = test_res["inference_time_s"]
    print("-" * 60)
    print(f"Test windows: {len(test_res)}")
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


def plot_predictions(res, deployment, pred_horizon, plots_dir, num_targets):
    span_hours = (res["timestamp"].iloc[-1] - res["timestamp"].iloc[0]).total_seconds() / 3600.0

    panels = []
    if num_targets > 1:
        panels = [
            ("CPU", "cpu", "pred_cpu", "CPU Utilization (fraction of core)"),
            ("Memory", "memory", "pred_mem", "Memory Utilization (fraction of request)"),
        ]
    else:
        panels = [("CPU", "cpu", "pred_cpu", "CPU Utilization (fraction of core)")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(18, 6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (title, acol, pcol, ylabel) in zip(axes, panels):
        actual = np.array(res[acol], dtype=float)
        pred = np.array(res[pcol], dtype=float)
        pred[~np.isfinite(pred)] = np.nan

        ax.plot(res["timestamp"], actual, label="Actual", color="blue", alpha=0.6)
        ax.plot(res["timestamp"], pd.Series(pred).shift(pred_horizon).to_numpy(),
                label="Predicted (t+%d)" % pred_horizon, color="orange", linestyle="-", alpha=0.9)

        vmax = max(np.nanmax(actual), np.nanmax(pred))
        ax.set_ylim(0, max(1.0, vmax * 1.1))
        ax.set_title(f"Deployment: {deployment} — {title}", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        _style_time_axis(ax, span_hours)

    fig.suptitle(f"Replay inference — {deployment} ({res['timestamp'].iloc[0]} to {res['timestamp'].iloc[-1]}) | Raw",
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
    preprocess_approach = ckpt.get("preprocess_approach", ckpt_args.get("preprocess_approach", "none"))

    if preprocess_approach == "swt":
        swt_level = ckpt_args.get("swt_level")
        mem_swt_level = ckpt_args.get("mem_swt_level")
        feat_swt_windows = _apply_swt_per_window(
            feat_raw, meta["feature_set"], meta["input_len"],
            swt_level=swt_level, mem_swt_level=mem_swt_level
        )
        print(f"Raw feature shape: {feat_raw.shape}, SWT windows shape: {feat_swt_windows.shape}")
        model_feat = feat_swt_windows
    else:
        print(f"Raw feature shape: {feat_raw.shape} (no SWT preprocessing)")
        model_feat = feat_raw

    res, t_total = replay(
        sub, model, meta, feat_raw, model_feat, device,
        start_ts=t_start, end_ts=t_end,
        simulate_live=args.simulate_live,
        checkpoint_path=args.checkpoint,
    )
    if res.empty:
        raise SystemExit(
            f"No window can end within [{t_start}, {t_end}): first "
            f"{meta['input_len']} minutes needed as context, so "
            f"predictions start at {sub['timestamp'].iloc[meta['input_len'] - 1]}.\n"
            f"Use later --start_hour or more --hours."
        )

    test_res = res

    if not test_res.empty:
        print_metrics(res, meta["pred_horizon"])
    print(f"\nReplay wall time: {t_total:.2f}s "
          f"({'real-time' if args.simulate_live else 'fast-forward (add --simulate_live for 1-min pacing)'})")

    plot_predictions(res, args.deployment, meta["pred_horizon"], args.plots_dir, meta["num_targets"])


if __name__ == "__main__":
    main()

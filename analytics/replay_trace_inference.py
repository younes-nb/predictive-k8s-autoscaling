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
)
from preprocessing.build_windows import _CSV_COLUMN_MAP, _CSV_COLUMN_MINMAX

RNN_TYPES = ("lstm", "gru", "bilstm", "bigrue")
BUILDER_TYPES = ("cnn_bilstm", "dpam")
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


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {}) or {}
    hyperparams = checkpoint.get("hyperparams", {}) or {}
    model_type = checkpoint.get("model_type") or "bilstm"
    feature_set = ckpt_args.get("feature_set", "cpu_mem_both")
    input_len = int(ckpt_args.get("input_len", 128))
    pred_horizon = int(ckpt_args.get("pred_horizon", 5))
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


def replay(df, model, meta, feat, device, start_ts=None, end_ts=None, simulate_live=False):
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]
    num_targets = meta["num_targets"]

    if len(df) < input_len + pred_horizon:
        raise SystemExit(
            f"Deployment trace only has {len(df)} rows; need >= {input_len + pred_horizon}"
        )

    ts = df["timestamp"].to_numpy()
    n = len(df)

    warmup = torch.tensor(feat[:input_len], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        model(warmup)

    rows = []
    t_total0 = time.perf_counter()
    for idx in range(input_len - 1, n):
        if start_ts is not None and ts[idx] < start_ts:
            continue
        if end_ts is not None and ts[idx] >= end_ts:
            break
        window = torch.tensor(feat[idx - input_len + 1:idx + 1],
                              dtype=torch.float32, device=device).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(window)
        preds = out[0] if isinstance(out, tuple) else out
        dt = time.perf_counter() - t0
        p = torch.round(preds[0, -1] * 100) / 100
        pred_cpu = p[0].item()
        pred_mem = p[1].item() if num_targets > 1 else float("nan")
        rows.append(
            (ts[idx], float(feat[idx, 0]), float(feat[idx, 1]) if num_targets > 1 else float("nan"),
             pred_cpu, pred_mem, dt)
        )
        if simulate_live:
            time.sleep(max(0.0, 60.0 - dt))
    t_total = time.perf_counter() - t_total0

    res = pd.DataFrame(
        rows, columns=["timestamp", "cpu", "memory", "pred_cpu", "pred_mem", "inference_time_s"]
    )
    return res, t_total


def print_metrics(res, pred_horizon):
    print("\n" + "=" * 60)
    print("REPLAY METRICS (pred[t] vs actual[t+%d])" % pred_horizon)
    print("-" * 60)
    for label, acol, pcol in (("CPU", "cpu", "pred_cpu"), ("Mem", "memory", "pred_mem")):
        if res[pcol].isna().all():
            print(f"{label:5s}  no predictions for this target")
            continue
        frame = pd.DataFrame(
            {"y": res[acol].shift(-pred_horizon), "a": res[acol], "pred": res[pcol]}
        ).dropna()
        if len(frame) < pred_horizon:
            print(f"{label:5s}  too few aligned rows ({len(frame)})")
            continue
        mse = float(((frame["y"] - frame["pred"]) ** 2).mean())
        mae = float((frame["y"] - frame["pred"]).abs().mean())
        naive_mae = float((frame["y"] - frame["a"]).abs().mean())
        d = (mae - naive_mae) / naive_mae * 100 if naive_mae > 0 else float("nan")
        print(f"{label:5s}  MSE {mse:.5f}  MAE {mae:.5f} ({mae*100:.2f}%)  "
              f"naive MAE {naive_mae:.5f} ({naive_mae*100:.2f}%)  delta {d:+.1f}%  (n={len(frame)})")
    inf = res["inference_time_s"]
    print("-" * 60)
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
                label="Predicted (t+%d)" % pred_horizon, color="orange", linestyle="--", alpha=0.9)

        vmax = max(np.nanmax(actual), np.nanmax(pred)) if np.any(np.isfinite(pred)) else np.nanmax(actual)
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
    feat, feat_cols = _feature_matrix(df, sub, meta["feature_set"])
    print(f"Checkpoint: {args.checkpoint}  model_type={meta['model_type']}  "
          f"feature_set={meta['feature_set']}  input_len={meta['input_len']}  "
          f"pred_horizon={meta['pred_horizon']}  num_targets={meta['num_targets']}")
    print(f"Input channels: {feat_cols}  (features={meta['feature_set']})")
    print(f"Replaying {args.deployment}: {t_start} -> {t_end} "
          f"({len(sel)} rows, {args.hours} hours)")

    res, t_total = replay(sub, model, meta, feat, device, start_ts=t_start, end_ts=t_end,
                          simulate_live=args.simulate_live)
    if res.empty:
        raise SystemExit(
            f"No window can end within [{t_start}, {t_end}): the first "
            f"{meta['input_len']} minutes of the trace are needed as context, so "
            f"predictions start at {sub['timestamp'].iloc[meta['input_len'] - 1]}.\n"
            f"Use a later --start_hour (>= {meta['input_len'] / 60:.2f}) or more --hours."
        )

    print_metrics(res, meta["pred_horizon"])
    print(f"\nReplay wall time: {t_total:.2f}s "
          f"({'real-time' if args.simulate_live else 'fast-forward (add --simulate_live for 1-min pacing)'})")

    plot_predictions(res, args.deployment, meta["pred_horizon"], args.plots_dir, meta["num_targets"])


if __name__ == "__main__":
    main()

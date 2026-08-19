#!/usr/bin/env python
"""Simulate traditional HPA vs predictive HPA on Alibaba trace data.

Replays CPU/memory traces from the Alibaba dataset through a trained forecaster,
simulates both traditional and predictive autoscaling controllers using
deploy-default HPA settings, and compares them on replica count, SLA violations,
and resource efficiency. Supports adaptive conformal prediction.
"""

import argparse
import os
import sys
import time
import glob
import json

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
    CONFORMAL_AVAILABLE = True
except ImportError:
    CONFORMAL_AVAILABLE = False
    print("Warning: conformal modules not available")

from core.models import RNNForecaster
from shared.features import (
    FEATURES,
    feature_names_for_feature_set,
    target_features_for_feature_set,
    get_feature_set,
)
from preprocessing.swt.decomposition import decompose_window
from preprocessing.swt.config import CFG as SWT_CFG

RNN_TYPES = ("lstm", "gru", "bilstm", "bigrue")
BUILDER_TYPES = ("cnn_bilstm", "dpam", "tcn", "tcn_dual", "quantile_ensemble")

DEFAULT_PLOTS_DIR = "/proj/k8sautoscaledl-PG0/analytics_out"
DEFAULT_PARQUET_ROOT = "/dataset/parquet"
DEFAULT_CHECKPOINT = "/proj/k8sautoscaledl-PG0/models/model.pt"

BASE_THRESHOLD = 0.80
TOLERANCE = 0.1
SCALE_UP_MAX_PERCENT = 100.0
SCALE_UP_MAX_PODS = 4
MIN_REPLICAS = 1
MAX_REPLICAS = 10
STABILIZATION_WINDOW_SECONDS = 300
EVAL_INTERVAL_SECONDS = 60
SCALE_UP_PERIOD_SECONDS = 15
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1


def parse_args():
    ap = argparse.ArgumentParser(
        description="Simulate traditional vs predictive HPA on Alibaba traces"
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--parquet_root", default=DEFAULT_PARQUET_ROOT)
    ap.add_argument("--msname", default=None,
                    help="Specific msname (default: auto-select best)")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--input_len", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plots_dir", default=DEFAULT_PLOTS_DIR)
    ap.add_argument("--adaptive_conformal", action="store_true")
    ap.add_argument("--calibration_minutes", type=int, default=200,
                    help="Minutes before evaluation used for online conformal calibration")
    ap.add_argument("--adaptive_window", type=int, default=500)
    ap.add_argument("--adaptive_eta", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=BASE_THRESHOLD)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--min_replicas", type=int, default=MIN_REPLICAS)
    ap.add_argument("--max_replicas", type=int, default=MAX_REPLICAS)
    ap.add_argument("--initial_replicas", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val_frac", type=float, default=VAL_FRAC)
    ap.add_argument("--max_services", type=int, default=0,
                    help="Limit auto-select to first N services (0=all)")
    return ap.parse_args()


# ================================================================
# Parquet Loading
# ================================================================

DEFAULT_SERVICE_ARRAYS = "/dataset/windows/_service_arrays.npy"
DEFAULT_SERVICE_INDEX = "/dataset/windows/_service_index.json"
DEFAULT_REPLICA_COUNTS = "/dataset/windows/_service_replica_counts.npy"
CACHE_FEATURES = ["cpu_utilization", "memory_utilization"]


def load_service_arrays_cache(arrays_path, index_path, feature_set,
                              replica_counts_path=None):
    """Load pre-aggregated service arrays from the build_windows cache.

    Returns dict: msname -> numpy array (N, num_features) and
    dict: msname -> float baseline replica count.
    """
    spec = get_feature_set(feature_set)
    feature_names = spec["features"]
    missing = [f for f in feature_names if f not in CACHE_FEATURES]
    if missing:
        raise FileNotFoundError(
            f"Cache missing features {missing}; cache only has {CACHE_FEATURES}"
        )

    feat_indices = [CACHE_FEATURES.index(f) for f in feature_names]

    with open(index_path) as f:
        data = json.load(f)
    index = data["index"]

    big = np.load(arrays_path, mmap_mode="r")
    print(f"Service arrays cache: {big.shape[0]} rows x {big.shape[1]} ch, "
          f"{len(index)} services")

    service_data = {}
    for svc_name, pos in index.items():
        arr = big[pos[0]:pos[0] + pos[1]][:, feat_indices].copy()
        service_data[svc_name] = arr

    baseline_replicas = {}
    if replica_counts_path and os.path.exists(replica_counts_path):
        rep_arr = np.load(replica_counts_path, mmap_mode="r")
        for svc_name, pos in index.items():
            baseline_replicas[svc_name] = max(1.0, float(rep_arr[pos[0]]))
        print(f"Loaded baseline replica counts for {len(baseline_replicas)} services")
    else:
        print("[WARN] No replica counts cache; using baseline_replicas=1 for all services")

    return service_data, CACHE_FEATURES[:len(feat_indices)], baseline_replicas


def load_alibaba_parquet(parquet_root, feature_set, service_arrays_path=None,
                         service_index_path=None, replica_counts_path=None):
    """Load Alibaba data per msname per minute.

    Tries the pre-aggregated service_arrays cache first (fast, handles cpu_mem_both).
    Falls back to parquet aggregation for other feature sets.
    """
    if service_arrays_path is None:
        service_arrays_path = DEFAULT_SERVICE_ARRAYS
    if service_index_path is None:
        service_index_path = DEFAULT_SERVICE_INDEX
    if replica_counts_path is None:
        replica_counts_path = DEFAULT_REPLICA_COUNTS

    if os.path.exists(service_arrays_path) and os.path.exists(service_index_path):
        try:
            raw_dict, cache_feats, baseline_replicas = load_service_arrays_cache(
                service_arrays_path, service_index_path, feature_set,
                replica_counts_path=replica_counts_path,
            )
            print(f"Loaded {len(raw_dict)} services from cache (features: {cache_feats})")
            return raw_dict, cache_feats, baseline_replicas
        except (FileNotFoundError, KeyError) as e:
            print(f"[WARN] Cache unavailable: {e}. Falling back to parquet.")

    raise SystemExit(
        "Parquet loading for non-cached feature sets is not yet supported "
        "due to dataset size (112GB msresource). Use cpu_mem_both feature set "
        "to use the pre-aggregated cache."
    )


# ================================================================
# Service Selection
# ================================================================

def select_best_msname(service_data, hours, input_len, pred_horizon,
                       train_frac, val_frac, max_services=0):
    """Select msname with highest avg+std CPU and memory in test split.

    service_data: dict of msname -> numpy array (N, num_features) or DataFrame.
    max_services: if >0, only evaluate this many services (sorted by name).
    """
    n_minutes = int(hours * 60)
    candidates = []

    svc_names = sorted(service_data.keys())
    if max_services > 0:
        svc_names = svc_names[:max_services]
        print(f"Limiting service selection to first {max_services} of "
              f"{len(service_data)} services")

    for svc_name in svc_names:
        arr = service_data[svc_name]
        if isinstance(arr, pd.DataFrame):
            n = len(arr)
        else:
            n = arr.shape[0]
        test_start = int(n * (train_frac + val_frac))
        if test_start + n_minutes > n:
            continue
        if test_start < input_len:
            continue

        segment = arr[test_start:test_start + n_minutes]
        if isinstance(segment, pd.DataFrame):
            cpu = segment["cpu_utilization"].values.astype(float)
            mem = segment["memory_utilization"].values.astype(float) if "memory_utilization" in segment.columns else np.zeros(1)
        else:
            cpu = segment[:, 0].astype(float)
            mem = segment[:, 1].astype(float) if segment.shape[1] > 1 else np.zeros(1)
        score = float(np.std(cpu) + np.std(mem))
        candidates.append((svc_name, score, test_start))

    if not candidates:
        raise SystemExit(
            f"No eligible msname found (need >= {input_len + n_minutes} rows "
            f"with {n_minutes} in test split)"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    print(f"Selected msname: {best[0]} (score={best[1]:.4f}, test_start={best[2]})")
    if len(candidates) > 1:
        print(f"  Top 5: {[(c[0], f'{c[1]:.4f}') for c in candidates[:5]]}")
    return best[0], best[2]


# ================================================================
# Model Loading (reused from replay_trace_inference.py)
# ================================================================

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


def _build_from_builder(checkpoint, model_type, feature_set, input_len,
                        num_targets, pred_horizon):
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
            print(f"[INFO] input_size={input_size} doesn't match '{feature_set}', using '{fs}'")
            return fs
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
    is_change_head = bool(
        ckpt_args.get("change_head", False) or ckpt_args.get("change_head_mem", False)
    )
    sd = checkpoint["model_state_dict"]
    if model_type in BUILDER_TYPES:
        model = _build_from_builder(
            checkpoint, model_type, feature_set, input_len, num_targets, pred_horizon
        )
        model.load_state_dict(sd)
    else:
        if is_change_head:
            sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        input_size, hidden, num_layers, bidirectional, rnn_type = (
            _derive_rnn_from_state_dict(sd)
        )
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
                inject_mask[-1 if num_targets > 1 else 0] = True
            model = ChangeHeadForecaster(model, inject_mask)
    model.to(device).eval()
    meta = {
        "model_type": model_type, "feature_set": feature_set,
        "input_len": input_len, "pred_horizon": pred_horizon,
        "num_targets": num_targets, "input_size": input_size,
    }
    return model, meta


# ================================================================
# Feature Construction
# ================================================================

def apply_swt(raw_feat, feature_set, input_len):
    """Apply SWT decomposition per sliding window."""
    spec = get_feature_set(feature_set)
    target_features = spec.get("targets", [spec.get("target")])
    has_mem = "memory_utilization" in target_features
    cpu_idx, mem_idx = 0, 1 if has_mem else -1
    n_cpu_ch = SWT_CFG.SWT_LEVEL + 1
    n_mem_ch = (SWT_CFG.MEM_SWT_LEVEL + 1) if has_mem else 0
    total_ch = n_cpu_ch + n_mem_ch
    n_samples = raw_feat.shape[0]
    n_windows = n_samples - input_len + 1
    out = np.zeros((n_windows, input_len, total_ch), dtype=np.float32)
    for i in range(n_windows):
        w = raw_feat[i:i + input_len]
        cpu_ch = decompose_window(w[:, cpu_idx].astype(np.float64), SWT_CFG)
        if cpu_ch is None:
            cpu_ch = np.zeros((n_cpu_ch, input_len), dtype=np.float32)
            cpu_ch[0] = w[:, cpu_idx]
        out[i, :, :n_cpu_ch] = cpu_ch.T
        if has_mem:
            mem_ch = decompose_window(w[:, mem_idx].astype(np.float64), SWT_CFG)
            if mem_ch is None:
                mem_ch = np.zeros((n_mem_ch, input_len), dtype=np.float32)
                mem_ch[0] = w[:, mem_idx]
            out[i, :, n_cpu_ch:] = mem_ch.T
    return out


# ================================================================
# HPA Simulation
# ================================================================

def _scale_up_ceiling(current_replicas):
    rep = int(current_replicas)
    periods = max(1, int(EVAL_INTERVAL_SECONDS // SCALE_UP_PERIOD_SECONDS))
    for _ in range(periods):
        rep = min(
            MAX_REPLICAS,
            rep + max(int(rep * SCALE_UP_MAX_PERCENT / 100), SCALE_UP_MAX_PODS),
        )
    return rep


def _desired_replicas(cpu_demand, mem_demand, current_replicas,
                      threshold, tolerance, num_targets):
    cpu_ratio = cpu_demand / threshold
    if abs(cpu_ratio - 1.0) <= tolerance:
        cpu_ratio = 1.0
    raw = int(np.ceil(current_replicas * cpu_ratio))

    if num_targets > 1:
        mem_ratio = mem_demand / threshold
        if abs(mem_ratio - 1.0) <= tolerance:
            mem_ratio = 1.0
        raw = max(raw, int(np.ceil(current_replicas * mem_ratio)))

    raw = max(MIN_REPLICAS, min(MAX_REPLICAS, raw))
    if raw > current_replicas:
        raw = min(raw, _scale_up_ceiling(current_replicas))
    return raw


def _run_calibration(raw_feat, model_feat, model, meta, device,
                     calibration_start, calibration_minutes,
                     num_targets, adaptive_window, adaptive_eta):
    """Run online conformal calibration before the evaluation window.

    Iterates over the calibration segment, runs inference, and feeds the
    q10/q95 predictions and actual values into an AdaptiveUpperConformalPerTarget
    calibrator.  Returns the fully calibrated calibrator for use in simulate_trace.
    """
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]

    if model_feat.ndim == 2:
        n_samp = model_feat.shape[0]
        n_win = n_samp - input_len + 1
        mw = np.zeros((n_win, input_len, model_feat.shape[1]), dtype=np.float32)
        for i in range(n_win):
            mw[i] = model_feat[i:i + input_len]
    else:
        mw = model_feat

    cal = AdaptiveUpperConformalPerTarget(
        num_targets=num_targets, window_size=adaptive_window,
        alpha=0.05, eta=adaptive_eta, alpha_min=0.01, alpha_max=0.20,
    )

    pending = []
    n = raw_feat.shape[0]
    end_idx = min(calibration_start + calibration_minutes, n)

    for idx in range(calibration_start, end_idx):
        widx = idx - input_len + 1
        if widx < 0 or widx >= len(mw):
            continue

        window = torch.tensor(mw[widx], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(window)
        preds = out[0] if isinstance(out, tuple) else out

        if preds.dim() == 4:
            q10 = preds[0, -1, :, 0].cpu().numpy()
            q95 = preds[0, -1, :, 2].cpu().numpy()
        else:
            p = torch.round(preds[0, -1] * 100) / 100
            q10 = p.cpu().numpy()
            q95 = q10.copy()

        cpu_actual = float(raw_feat[idx, 0])
        mem_actual = float(raw_feat[idx, 1]) if num_targets > 1 else 0.0

        # Direct online calibration using current observation
        cal.states["cpu"].update(cpu_actual, float(q10[0]), float(q95[0]))
        if num_targets > 1:
            cal.states["memory"].update(mem_actual, float(q10[1]), float(q95[1]))

        # Delayed calibration using matured predictions
        pending.append({"idx": idx, "q10": q10.copy(), "q95": q95.copy()})
        matured = [p for p in pending if p["idx"] <= idx - pred_horizon]
        for p in matured:
            aidx = int(p["idx"] + pred_horizon)
            if aidx < n:
                ac = float(raw_feat[aidx, 0])
                am = float(raw_feat[aidx, 1]) if num_targets > 1 else 0.0
                cal.states["cpu"].update(ac, p["q10"][0], p["q95"][0])
                if num_targets > 1:
                    cal.states["memory"].update(am, p["q10"][1], p["q95"][1])
        pending = [p for p in pending if p["idx"] > idx - pred_horizon]

    print(f"[INFO] Conformal calibration complete ({end_idx - calibration_start} min)")
    return cal


def simulate_trace(raw_feat, model_feat, model, meta, device,
                   start_idx, n_minutes,
                   threshold=BASE_THRESHOLD, tolerance=TOLERANCE,
                   num_targets=2, initial_replicas=1,
                   adaptive_cal=None,
                   adaptive_window=500, adaptive_eta=0.01,
                   timestamps=None):
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]

    if model_feat.ndim == 2:
        n_samp = model_feat.shape[0]
        n_win = n_samp - input_len + 1
        mw = np.zeros((n_win, input_len, model_feat.shape[1]), dtype=np.float32)
        for i in range(n_win):
            mw[i] = model_feat[i:i + input_len]
    else:
        mw = model_feat

    wt = torch.tensor(mw[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        model(wt)

    trad_rep = initial_replicas
    pred_rep = initial_replicas
    trad_hist = []
    pred_hist = []
    pending = []
    n = raw_feat.shape[0]
    end_idx = min(start_idx + n_minutes, n)
    results = []

    for idx in range(start_idx, end_idx):
        widx = idx - input_len + 1
        if widx < 0 or widx >= len(mw):
            continue

        window = torch.tensor(mw[widx], dtype=torch.float32, device=device).unsqueeze(0)
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
            q10, q95 = q50.copy(), q50.copy()

        pred_cpu = float(np.round(q50[0] * 100) / 100)
        pred_mem = float(np.round(q50[1] * 100) / 100) if num_targets > 1 else 0.0

        lower_cpu, upper_cpu = pred_cpu, pred_cpu
        lower_mem, upper_mem = pred_mem, pred_mem
        if adaptive_cal is not None:
            la, ua = adaptive_cal.get_interval(q10, q95)
            lower_cpu, upper_cpu = float(la[0]), float(ua[0])
            if num_targets > 1:
                lower_mem, upper_mem = float(la[1]), float(ua[1])

        cpu_actual = float(raw_feat[idx, 0])
        mem_actual = float(raw_feat[idx, 1]) if num_targets > 1 else 0.0

        # Online conformal feedback with delayed horizon alignment
        if adaptive_cal is not None:
            pending.append({"idx": idx, "q10": q10.copy(), "q95": q95.copy()})
            matured = [p for p in pending if p["idx"] <= idx - pred_horizon]
            for p in matured:
                aidx = int(p["idx"] + pred_horizon)
                if aidx < n:
                    ac = float(raw_feat[aidx, 0])
                    am = float(raw_feat[aidx, 1]) if num_targets > 1 else 0.0
                    adaptive_cal.states["cpu"].update(ac, p["q10"][0], p["q95"][0])
                    if num_targets > 1:
                        adaptive_cal.states["memory"].update(am, p["q10"][1], p["q95"][1])
            pending = [p for p in pending if p["idx"] > idx - pred_horizon]

        # Traditional HPA: scale on actual values
        trad_raw = _desired_replicas(cpu_actual, mem_actual, trad_rep,
                                     threshold, tolerance, num_targets)
        trad_hist.append({"idx": idx, "r": trad_raw})
        stab = STABILIZATION_WINDOW_SECONDS // 60
        tw = [h["r"] for h in trad_hist if h["idx"] > idx - stab]
        trad_final = trad_raw if trad_raw > trad_rep else (max(tw) if tw else trad_raw)
        trad_rep = int(max(MIN_REPLICAS, min(MAX_REPLICAS, trad_final)))

        # Predictive HPA: scale on predicted (or conformal upper) values
        if adaptive_cal is not None:
            pcpu, pmem = upper_cpu, upper_mem
        else:
            pcpu, pmem = pred_cpu, pred_mem

        pred_raw = _desired_replicas(pcpu, pmem, pred_rep,
                                     threshold, tolerance, num_targets)
        pred_hist.append({"idx": idx, "r": pred_raw})
        pw = [h["r"] for h in pred_hist if h["idx"] > idx - stab]
        pred_final = pred_raw if pred_raw > pred_rep else (max(pw) if pw else pred_raw)
        pred_rep = int(max(MIN_REPLICAS, min(MAX_REPLICAS, pred_final)))

        ts_val = pd.Timestamp(timestamps[idx]) if timestamps is not None else None
        results.append({
            "timestamp": ts_val,
            "cpu": cpu_actual,
            "memory": mem_actual,
            "pred_cpu": pred_cpu,
            "pred_mem": pred_mem,
            "lower_cpu": lower_cpu,
            "upper_cpu": upper_cpu,
            "lower_mem": lower_mem,
            "upper_mem": upper_mem,
            "traditional_replicas": trad_rep,
            "predictive_replicas": pred_rep,
            "inference_time_s": dt,
        })

    return results


# ================================================================
# Metrics
# ================================================================

def compute_metrics(results, pred_horizon, threshold, use_conformal=False):
    """Compute paper-aligned metrics: replica stats, prediction accuracy
    (MSE/MAE/RMSE/R2/MAPE/MDA), persistence comparison, and
    conformal interval quality (PICP/MPIW) when applicable."""
    df = pd.DataFrame(results)

    if df.empty:
        print("No evaluation data")
        return {}

    m = {}

    # --- Replica metrics (per controller) ---
    for p, c in [("traditional", "traditional_replicas"), ("predictive", "predictive_replicas")]:
        r = df[c].values
        m[f"{p}_avg_replicas"] = float(np.mean(r))
        m[f"{p}_max_replicas"] = int(np.max(r))
        m[f"{p}_min_replicas"] = int(np.min(r))
        m[f"{p}_scaling_actions"] = int(np.sum(np.abs(np.diff(r)) > 0))

    # --- Replica reduction ---
    ta = m.get("traditional_avg_replicas", 0)
    pa = m.get("predictive_avg_replicas", 0)
    m["replica_reduction_pct"] = ((ta - pa) / ta * 100) if ta > 0 else 0.0

    # --- Prediction accuracy (horizon-aligned) ---
    ac = df["cpu"].values.astype(float)
    pc = np.roll(df["pred_cpu"].values.astype(float), pred_horizon)
    pc[:pred_horizon] = np.nan
    am = df["memory"].values.astype(float)
    pm = np.roll(df["pred_mem"].values.astype(float), pred_horizon)
    pm[:pred_horizon] = np.nan

    for target_name, actual, pred in [("cpu", ac, pc), ("memory", am, pm)]:
        valid = ~np.isnan(pred)
        actual = actual[valid]
        pred = pred[valid]
        err = pred - actual
        abs_err = np.abs(err)
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(mse))

        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        nonzero = np.abs(actual) > 1e-12
        mape = float(np.mean(np.abs(err[nonzero]) / np.abs(actual[nonzero]))) * 100.0 if np.sum(nonzero) > 0 else 0.0

        if len(actual) > 1:
            actual_dir = np.sign(actual[1:] - actual[:-1])
            pred_dir = np.sign(pred[1:] - actual[:-1])
            mda = float(np.mean(actual_dir == pred_dir)) * 100.0
        else:
            mda = 0.0

        m[f"pred_{target_name}_mse"] = mse
        m[f"pred_{target_name}_mae"] = mae
        m[f"pred_{target_name}_rmse"] = rmse
        m[f"pred_{target_name}_r2"] = r2
        m[f"pred_{target_name}_mape"] = mape
        m[f"pred_{target_name}_mda"] = mda

        # --- Persistence comparison (current load as naive baseline) ---
        y_last = np.roll(actual, 1)
        y_last[0] = actual[0]
        mse_naive = float(np.mean((y_last - actual) ** 2))
        mae_naive = float(np.mean(np.abs(y_last - actual)))
        r2_vs_persistence = 1.0 - mse / mse_naive if mse_naive > 1e-12 else float("nan")
        mae_vs_persistence = mae / mae_naive if mae_naive > 1e-12 else float("nan")
        beat_persistence = float(np.mean(np.abs(err) < np.abs(y_last - actual)) * 100.0)

        m[f"{target_name}_r2_vs_persistence"] = r2_vs_persistence
        m[f"{target_name}_beat_persistence"] = beat_persistence
        m[f"{target_name}_mae_vs_persistence"] = mae_vs_persistence

    # --- Conformal interval quality (PICP, MPIW) when upper/lower bounds differ ---
    if use_conformal and "upper_cpu" in df.columns:
        for target_name, col_act, col_lo, col_hi in [
            ("cpu", "cpu", "lower_cpu", "upper_cpu"),
            ("memory", "memory", "lower_mem", "upper_mem"),
        ]:
            actual = df[col_act].values.astype(float)
            lo = df[col_lo].values.astype(float)
            hi = df[col_hi].values.astype(float)
            in_interval = (actual >= lo) & (actual <= hi)
            picp = float(np.mean(in_interval))
            mpiw = float(np.mean(hi - lo))
            m[f"{target_name}_picp"] = picp
            m[f"{target_name}_mpiw"] = mpiw

    m["n_evaluation_minutes"] = len(df)
    return m


# ================================================================
# Plotting
# ================================================================

def plot_results(results, msname, plots_dir, pred_horizon, use_conformal, threshold,
                  num_targets=2):
    df = pd.DataFrame(results)
    has_ts = df["timestamp"].notna().all()
    x = df["timestamp"] if has_ts else df.index

    os.makedirs(plots_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    targets = [
        ("cpu", "blue", "orange", "orange", "Actual CPU", "Predicted CPU",
         "cpu", "lower_cpu", "upper_cpu", "pred_cpu"),
    ]
    if num_targets > 1:
        targets.append(
            ("memory", "blue", "orange", "orange", "Actual Memory", "Predicted Memory",
             "memory", "lower_mem", "upper_mem", "pred_mem"),
        )

    for (target, color_actual, color_pred, color_pred_fill,
         label_actual, label_pred,
         actual_col, lo_col, hi_col, pred_col) in targets:

        fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True)

        # Shift predictions forward by pred_horizon so each predicted point
        # aligns with the actual value it is forecasting
        pred_shifted = df[pred_col].shift(pred_horizon)
        lo_shifted = df[lo_col].shift(pred_horizon) if lo_col in df.columns else None
        hi_shifted = df[hi_col].shift(pred_horizon) if hi_col in df.columns else None

        # --- Panel 1: Raw + predicted utilization ---
        ax = axes[0]
        ax.plot(x, df[actual_col], label=label_actual, color=color_actual, alpha=0.7)
        ax.plot(x, pred_shifted, label=f"{label_pred} (+{pred_horizon}m ahead)",
                color=color_pred, alpha=0.7)
        if use_conformal and lo_shifted is not None:
            ax.fill_between(x, lo_shifted, hi_shifted,
                            color=color_pred_fill, alpha=0.15,
                            label=f"{target.upper()} Conformal (+{pred_horizon}m)")
        ax.axhline(y=threshold, color="black", linestyle="--", alpha=0.5,
                   label=f"Threshold={threshold}")
        ax.set_ylabel("Utilization")
        ax.set_title(f"{msname} -- {target.upper()} Utilization", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True)
        ax.grid(True, alpha=0.3)

        # --- Panel 2: Replica counts ---
        ax = axes[1]
        ax.plot(x, df["traditional_replicas"], label="Traditional HPA",
                color="blue", linewidth=1.5)
        pred_label = "Predictive HPA" + (" + Conformal" if use_conformal else "")
        ax.plot(x, df["predictive_replicas"], label=pred_label, color="red", linewidth=1.5)
        ax.set_ylabel("Replicas")
        ax.set_title("Replica Count Comparison", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # --- Panel 3: Prediction error (horizon-aligned) ---
        ax = axes[2]
        err = pred_shifted.values.astype(float) - df[actual_col].values.astype(float)
        ax.plot(x, err, label=f"{target.upper()} Pred Error (h={pred_horizon})",
                color=color_pred, alpha=0.7, linewidth=1.2)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.3)
        ax.axhline(y=-threshold, color="red", linestyle=":", alpha=0.3)
        ax.set_ylabel("Error (Predicted - Actual)")
        ax.set_title(f"{target.upper()} Prediction Error (h={pred_horizon}m)",
                     fontweight="bold")
        ax.set_xlabel("Time")
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True)
        ax.grid(True, alpha=0.3)

        if has_ts:
            span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
            loc = mdates.HourLocator(interval=2) if span > 18 else mdates.MinuteLocator(interval=5)
            for a in axes:
                a.xaxis.set_major_locator(loc)
                a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")

        fig.suptitle(f"HPA Simulation -- {msname} ({target.upper()})", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])

        png = os.path.join(plots_dir, f"hpa_sim_{msname}_{target}_{stamp}.png")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {png}")


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, meta = load_model(args.checkpoint, device)
    if args.input_len is not None:
        meta["input_len"] = args.input_len

    print("Loading Alibaba parquet...")
    service_data, cache_feats, _ = load_alibaba_parquet(
        args.parquet_root, meta["feature_set"]
    )

    if args.msname:
        msname = args.msname
        if msname not in service_data:
            raise SystemExit(f"msname '{msname}' not found in data")
        arr = service_data[msname]
        n = arr.shape[0]
        test_start = int(n * (args.train_frac + args.val_frac))
        print(f"Using msname: {msname} (N={n}, test_start={test_start})")
    else:
        msname, test_start = select_best_msname(
            service_data, args.hours, meta["input_len"], meta["pred_horizon"],
            args.train_frac, args.val_frac, max_services=args.max_services,
        )

    arr = service_data[msname]
    n_minutes = int(args.hours * 60)
    raw_feat = arr.astype(np.float32)
    print(f"Raw feature array: {raw_feat.shape}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}
    preprocess = ckpt_args.get("preprocess_approach", "none")

    if preprocess == "swt":
        model_feat = apply_swt(raw_feat, meta["feature_set"], meta["input_len"])
        print(f"SWT windows: {model_feat.shape}")
    else:
        model_feat = raw_feat

    timestamps = None
    eval_minutes = n_minutes
    cal_minutes = args.calibration_minutes if args.adaptive_conformal else 0

    if args.adaptive_conformal:
        cal_start = test_start - cal_minutes
        if cal_start < meta["input_len"]:
            raise SystemExit(
                f"Not enough data for calibration: test_start={test_start}, "
                f"calibration_minutes={cal_minutes}, input_len={meta['input_len']}"
            )
        print(f"\nCalibration phase: {cal_minutes} min "
              f"[{cal_start}:{test_start}]")
        adaptive_cal = _run_calibration(
            raw_feat, model_feat, model, meta, device,
            calibration_start=cal_start, calibration_minutes=cal_minutes,
            num_targets=meta["num_targets"],
            adaptive_window=args.adaptive_window,
            adaptive_eta=args.adaptive_eta,
        )
    else:
        adaptive_cal = None

    print(f"Evaluation phase: {args.hours}h ({eval_minutes} min) "
          f"[{test_start}:{test_start + eval_minutes}]")
    results = simulate_trace(
        raw_feat, model_feat, model, meta, device,
        start_idx=test_start, n_minutes=eval_minutes,
        threshold=args.threshold, tolerance=args.tolerance,
        num_targets=meta["num_targets"],
        initial_replicas=args.initial_replicas,
        adaptive_cal=adaptive_cal,
        adaptive_window=args.adaptive_window,
        adaptive_eta=args.adaptive_eta,
        timestamps=timestamps,
    )

    if not results:
        raise SystemExit("No simulation results. Check --hours and test split size.")

    metrics = compute_metrics(results, meta["pred_horizon"], args.threshold,
                              use_conformal=args.adaptive_conformal)

    print("\n" + "=" * 72)
    print("HPA SIMULATION RESULTS")
    print("=" * 72)
    print(f"Service: {msname}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.adaptive_conformal:
        print(f"Conformal: Yes (calibration={cal_minutes} min, "
              f"eval={eval_minutes} min)")
    else:
        print(f"Conformal: No")
    print(f"Evaluation: {args.hours}h ({metrics.get('n_evaluation_minutes', 0)} min)")
    print("-" * 72)
    print(f"{'Metric':<35} {'Traditional':>12} {'Predictive':>12}")
    print("-" * 72)
    print(f"{'Avg Replicas':<35} {metrics.get('traditional_avg_replicas', 0):>12.2f} "
          f"{metrics.get('predictive_avg_replicas', 0):>12.2f}")
    print(f"{'Max Replicas':<35} {metrics.get('traditional_max_replicas', 0):>12d} "
          f"{metrics.get('predictive_max_replicas', 0):>12d}")
    print(f"{'Min Replicas':<35} {metrics.get('traditional_min_replicas', 0):>12d} "
          f"{metrics.get('predictive_min_replicas', 0):>12d}")
    print(f"{'Scaling Actions':<35} {metrics.get('traditional_scaling_actions', 0):>12d} "
          f"{metrics.get('predictive_scaling_actions', 0):>12d}")
    print(f"{'Replica Reduction':<35} {metrics.get('replica_reduction_pct', 0):>+11.1f}%")
    print("-" * 72)
    print(f"{'Prediction Accuracy (CPU)':<35} {'Value':>12}")
    print("-" * 72)
    for k, label in [
        ("pred_cpu_mse", "MSE"), ("pred_cpu_mae", "MAE"), ("pred_cpu_rmse", "RMSE"),
        ("pred_cpu_r2", "R²"), ("pred_cpu_mape", "MAPE (%)"), ("pred_cpu_mda", "MDA (%)"),
    ]:
        print(f"  {label:<33} {metrics.get(k, 0):>12.6f}")
    print("-" * 72)
    print(f"{'Prediction Accuracy (Memory)':<35} {'Value':>12}")
    print("-" * 72)
    for k, label in [
        ("pred_memory_mse", "MSE"), ("pred_memory_mae", "MAE"), ("pred_memory_rmse", "RMSE"),
        ("pred_memory_r2", "R²"), ("pred_memory_mape", "MAPE (%)"), ("pred_memory_mda", "MDA (%)"),
    ]:
        print(f"  {label:<33} {metrics.get(k, 0):>12.6f}")
    print("-" * 72)
    print(f"{'Persistence Comparison':<35} {'Value':>12}")
    print("-" * 72)
    for tgt in ["cpu", "memory"]:
        for k, label in [
            (f"{tgt}_r2_vs_persistence", f"{tgt.upper()} R² vs persistence"),
            (f"{tgt}_beat_persistence", f"{tgt.upper()} beat-persistence (%)"),
            (f"{tgt}_mae_vs_persistence", f"{tgt.upper()} MAE vs persistence"),
        ]:
            v = metrics.get(k, float("nan"))
            print(f"  {label:<33} {v:>12.4f}")
    if args.adaptive_conformal and metrics.get("cpu_picp") is not None:
        print("-" * 72)
        print(f"{'Conformal Interval Quality':<35} {'Value':>12}")
        print("-" * 72)
        for tgt in ["cpu", "memory"]:
            picp = metrics.get(f"{tgt}_picp", 0)
            mpiw = metrics.get(f"{tgt}_mpiw", 0)
            print(f"  {tgt.upper()+' PICP':<33} {picp:>11.1%}")
            print(f"  {tgt.upper()+' MPIW':<33} {mpiw:>12.6f}")
    print("=" * 72)

    os.makedirs(args.plots_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.plots_dir, f"hpa_sim_{msname}_{stamp}.csv")
    json_path = os.path.join(args.plots_dir, f"hpa_sim_{msname}_{stamp}.json")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    metrics.update({
        "msname": msname, "checkpoint": args.checkpoint, "hours": args.hours,
        "threshold": args.threshold, "use_conformal": args.adaptive_conformal,
    })
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")

    plot_results(results, msname, args.plots_dir, meta["pred_horizon"],
                 args.adaptive_conformal, args.threshold,
                 num_targets=meta["num_targets"])


if __name__ == "__main__":
    main()

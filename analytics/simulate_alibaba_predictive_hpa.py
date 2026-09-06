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
    mcr_column_indices,
    normalize_mcr_array,
    is_derived_feature,
    derived_time_value,
)
from preprocessing.swt.decomposition import decompose_window
from preprocessing.swt.config import CFG as SWT_CFG

RNN_TYPES = ("lstm", "gru", "bilstm", "bigrue")
BUILDER_TYPES = ("cnn_bilstm", "dpam", "tcn", "tcn_dual", "quantile_ensemble", "linearreg", "dlinear")

DEFAULT_PLOTS_DIR = "/proj/k8sautoscaledl-PG0/analytics_out"
DEFAULT_PARQUET_ROOT = "/dataset/parquet"
DEFAULT_CHECKPOINT = "/proj/k8sautoscaledl-PG0/models/model.pt"

BASE_THRESHOLD = 0.80
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1


def parse_args():
    ap = argparse.ArgumentParser(
        description="Simulate traditional vs predictive HPA on Alibaba traces"
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--parquet_root", default=DEFAULT_PARQUET_ROOT)
    ap.add_argument("--windows_dir", default=None,
                    help="Path to windows directory (contains _service_arrays.npy)")
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
# Legacy fallback for caches written before build_windows stored channel names
# in _service_index.json (those caches always held cpu/mem/rpc in this order).
CACHE_FEATURES = ["cpu_utilization", "memory_utilization", "providerrpc_mcr", "http_mcr"]


def load_service_arrays_cache(arrays_path, index_path, feature_set,
                              replica_counts_path=None):
    """Load pre-aggregated service arrays from the build_windows cache.

    Returns (service_data, target_data, cache_feats, baseline_replicas):
    service_data maps msname -> numpy array (N, num_features); target_data
    maps msname -> numpy array (N, num_extra_targets) for targets outside the
    input features (None when every target is an input channel).
    baseline maps msname -> float baseline replica count.
    """
    spec = get_feature_set(feature_set)
    feature_names = spec["features"]
    target_names = spec["targets"]
    target_only = [t for t in target_names if t not in feature_names]

    with open(index_path) as f:
        data = json.load(f)
    index = data["index"]
    # Channel order == the feature list stored at build time; fall back to the
    # legacy fixed order for old caches.
    cached_features = data.get("features") or CACHE_FEATURES
    missing = [f for f in feature_names + target_only if f not in cached_features]
    if missing:
        raise FileNotFoundError(
            f"Cache missing features {missing}; cache only has {cached_features}"
        )

    feat_indices = [cached_features.index(f) for f in feature_names]
    extra_indices = [cached_features.index(t) for t in target_only]

    big = np.load(arrays_path, mmap_mode="r")
    if big.shape[1] < max(feat_indices) + 1:
        raise FileNotFoundError(
            f"Cache width {big.shape[1]} incompatible with channels "
            f"{cached_features}; rebuild windows with --recompute_windows"
        )
    print(f"Service arrays cache: {big.shape[0]} rows x {big.shape[1]} ch, "
          f"{len(index)} services")

    service_data = {}
    target_data = {}
    for svc_name, pos in index.items():
        arr = big[pos[0]:pos[0] + pos[1]][:, feat_indices].copy()
        service_data[svc_name] = arr
        if extra_indices:
            target_data[svc_name] = big[pos[0]:pos[0] + pos[1]][:, extra_indices].copy()
    if not extra_indices:
        target_data = None

    # Mirror build_windows MCR scaling: the cache file always holds raw values
    # while train windows were built from [0,1]-normalized MCR channels. The
    # recorded scope selects the identical transform so the model sees the
    # scale it was trained on: per-service min/max ("per_service",
    # --normalize_mcr) or dataset-wide min/max ("global", the default).
    # Legacy caches without a scope key fall back to the old boolean
    # (true -> per_service, absent -> none/raw).
    scope = data.get("mcr_norm_scope")
    if scope is None:
        scope = "per_service" if data.get("normalize_mcr") else "none"
    mcr_cols = mcr_column_indices(feature_names)
    if mcr_cols and scope == "per_service":
        print(f"[INFO] Applying recorded per-service MCR normalization "
              f"to {[feature_names[j] for j in mcr_cols]}")
        for arr in service_data.values():
            normalize_mcr_array(arr, mcr_cols)
    elif mcr_cols and scope == "global":
        print(f"[INFO] Applying recorded dataset-wide MCR normalization "
              f"to {[feature_names[j] for j in mcr_cols]}")
        glo = {}
        for j in mcr_cols:
            los = [float(arr[:, j].min()) for arr in service_data.values()]
            his = [float(arr[:, j].max()) for arr in service_data.values()]
            glo[j] = (min(los), max(his))
        for arr in service_data.values():
            normalize_mcr_array(arr, mcr_cols, lo_hi=glo)

    baseline_replicas = {}
    if replica_counts_path and os.path.exists(replica_counts_path):
        rep_arr = np.load(replica_counts_path, mmap_mode="r")
        for svc_name, pos in index.items():
            baseline_replicas[svc_name] = max(1.0, float(rep_arr[pos[0]]))
        print(f"Loaded baseline replica counts for {len(baseline_replicas)} services")
    else:
        print("[WARN] No replica counts cache; using baseline_replicas=1 for all services")

    return service_data, target_data, [cached_features[i] for i in feat_indices], baseline_replicas


def resolve_msname(requested, available):
    """Resolve a user-passed --msname against cached service ids.

    Alibaba ids are stored with an 'MS_' prefix (e.g. 'MS_15819'), but users
    often pass the bare number ('15819'). Accept both directions.
    Returns the matched id, or None if no match.
    """
    if requested in available:
        return requested
    # Bare number -> prefixed
    prefixed = f"MS_{requested}"
    if prefixed in available:
        print(f"[INFO] Resolved msname '{requested}' -> '{prefixed}'")
        return prefixed
    # Case-insensitive prefixed match (e.g. 'ms_15819' -> 'MS_15819')
    upper_prefixed = prefixed.upper()
    for svc in available:
        if svc.upper() == upper_prefixed:
            print(f"[INFO] Resolved msname '{requested}' -> '{svc}'")
            return svc
    # Prefixed -> bare (cache built without prefix)
    if requested.upper().startswith("MS_"):
        bare = requested[len("MS_"):]
        if bare in available:
            print(f"[INFO] Resolved msname '{requested}' -> '{bare}'")
            return bare
        for svc in available:
            if svc.upper() == bare.upper():
                print(f"[INFO] Resolved msname '{requested}' -> '{svc}'")
                return svc
    return None


def load_alibaba_parquet(parquet_root, feature_set, service_arrays_path=None,
                         service_index_path=None, replica_counts_path=None,
                         windows_dir=None):
    """Load Alibaba data per msname per minute.

    Tries the pre-aggregated service_arrays cache first (fast, handles cpu_mem_both).
    Falls back to reading parquet directly for other feature sets.
    """
    if windows_dir:
        if service_arrays_path is None:
            service_arrays_path = os.path.join(windows_dir, "_service_arrays.npy")
        if service_index_path is None:
            service_index_path = os.path.join(windows_dir, "_service_index.json")
        if replica_counts_path is None:
            replica_counts_path = os.path.join(windows_dir, "_service_replica_counts.npy")
    if service_arrays_path is None:
        service_arrays_path = DEFAULT_SERVICE_ARRAYS
    if service_index_path is None:
        service_index_path = DEFAULT_SERVICE_INDEX
    if replica_counts_path is None:
        replica_counts_path = DEFAULT_REPLICA_COUNTS

    if os.path.exists(service_arrays_path) and os.path.exists(service_index_path):
        try:
            raw_dict, target_dict, cache_feats, baseline_replicas = load_service_arrays_cache(
                service_arrays_path, service_index_path, feature_set,
                replica_counts_path=replica_counts_path,
            )
            print(f"Loaded {len(raw_dict)} services from cache (features: {cache_feats})")
            return raw_dict, target_dict, cache_feats, baseline_replicas
        except (FileNotFoundError, KeyError) as e:
            print(f"[WARN] Cache unavailable: {e}. Falling back to parquet.")

    return _load_from_parquet(parquet_root, feature_set)


def _load_from_parquet(parquet_root, feature_set):
    """Load per-service feature arrays by reading parquet files directly."""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    spec = get_feature_set(feature_set)
    feature_names = spec["features"]
    target_only = [t for t in spec["targets"] if t not in feature_names]
    service_col = "msname"

    tables_needed = {}
    for feat_name in feature_names + target_only:
        if is_derived_feature(feat_name):
            continue  # synthesized from the minute index below, no table read
        meta = FEATURES[feat_name]
        t = meta["table"]
        c = meta["column"]
        tables_needed.setdefault(t, []).append((feat_name, c))

    service_data = {}

    for table_name, col_pairs in tables_needed.items():
        table_dir = os.path.join(parquet_root, table_name)
        if not os.path.isdir(table_dir):
            print(f"[WARN] Table dir not found: {table_dir}")
            continue

        # Only real parquet parts: the dirs also contain marker files
        # (e.g. msr_*.done) that are not parquet and break dataset discovery.
        part_files = sorted(glob.glob(os.path.join(table_dir, "part-*.parquet")))
        if not part_files:
            print(f"[WARN] No part-*.parquet files in {table_dir}")
            continue

        print(f"Reading table '{table_name}' from {table_dir} "
              f"({len(part_files)} parts)...")
        try:
            dataset = ds.dataset(part_files, format="parquet")
        except Exception as e:
            print(f"[WARN] Could not read {table_dir}: {e}")
            continue

        feat_names = [fp[0] for fp in col_pairs]
        raw_cols = [fp[1] for fp in col_pairs]
        columns_needed = list(set([service_col, "timestamp"] + raw_cols))

        try:
            table = dataset.to_table(columns=columns_needed).to_pandas()
        except Exception as e:
            print(f"[WARN] Could not read columns from {table_name}: {e}")
            continue

        print(f"  Rows: {len(table)}")
        table["ts_min"] = (table["timestamp"] / 60000).astype(int)

        for svc_name, group in table.groupby(service_col):
            if svc_name not in service_data:
                service_data[svc_name] = {}
            grouped = group.groupby("ts_min").agg({c: "mean" for _, c in col_pairs})
            for feat_name, raw_col in col_pairs:
                if feat_name not in service_data[svc_name]:
                    service_data[svc_name][feat_name] = {}
                for m, row in grouped.iterrows():
                    service_data[svc_name][feat_name][int(m)] = float(row[raw_col])

    N = max(max(d.values(), key=max).keys() if d.values() else [0]
            for s in service_data.values() for d in s.values()) + 1

    out_dict = {}
    target_dict = {}
    out_feats = feature_names
    all_names = feature_names + target_only
    for svc_name, feat_dict in service_data.items():
        arr = np.zeros((N, len(all_names)), dtype=np.float32)
        # Minutes actually observed for this service (union over real
        # features); derived calendar features are synthesized exactly for
        # these, mirroring build_windows' drop_nulls on real columns.
        real_minutes = set()
        for fname in all_names:
            if not is_derived_feature(fname) and fname in feat_dict:
                real_minutes.update(feat_dict[fname].keys())
        for fi, fname in enumerate(all_names):
            if is_derived_feature(fname):
                for m in real_minutes:
                    if m < N:
                        arr[m, fi] = derived_time_value(fname, int(m))
            elif fname in feat_dict:
                for m, v in feat_dict[fname].items():
                    if m < N:
                        arr[m, fi] = v
            else:
                arr[:, fi] = np.nan
        mask = ~np.isnan(arr).all(axis=1)
        if mask.sum() == 0:
            continue
        good = np.where(mask)[0]
        for fi in range(arr.shape[1]):
            if np.isnan(arr[:, fi]).any():
                arr[:, fi] = np.interp(
                    np.arange(N), good, arr[good, fi],
                    left=float(arr[good[0], fi]) if len(good) > 0 else 0.0,
                    right=float(arr[good[-1], fi]) if len(good) > 0 else 0.0,
                )
        out_dict[svc_name] = arr[:, :len(feature_names)]
        if target_only:
            target_dict[svc_name] = arr[:, len(feature_names):]
    if not target_only:
        target_dict = None

    baseline_replicas = {}
    print(f"Loaded {len(out_dict)} services from parquet (features: {out_feats})")
    return out_dict, target_dict, out_feats, baseline_replicas


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

def _target_actual_lookup(meta, tgt_actual):
    """Resolve (tgt_chan, tgt_vec, mem_chan) for ground-truth actuals.

    The primary target's actuals come from the explicit tgt_actual vector when
    the target is not a model input; otherwise from its input channel
    (channel 0 for all current sets). Memory actuals come from the memory
    input channel when the feature set has one, else 0.0 (N/A panel).
    """
    feat_names = feature_names_for_feature_set(meta["feature_set"])
    tgt_names = target_features_for_feature_set(meta["feature_set"])
    if tgt_actual is not None:
        tgt_chan, tgt_vec = 0, tgt_actual
    elif tgt_names and tgt_names[0] in feat_names:
        tgt_chan, tgt_vec = feat_names.index(tgt_names[0]), None
    else:
        tgt_chan, tgt_vec = 0, None
    mem_chan = (feat_names.index("memory_utilization")
                if "memory_utilization" in feat_names else None)
    return tgt_chan, tgt_vec, mem_chan


def _run_calibration(raw_feat, model_feat, model, meta, device,
                     calibration_start, calibration_minutes,
                     num_targets, adaptive_window, adaptive_eta,
                     tgt_actual=None):
    """Run online conformal calibration before the evaluation window.

    Iterates over the calibration segment, runs inference, and feeds the
    q10/q95 predictions and actual values into an AdaptiveUpperConformalPerTarget
    calibrator.  Returns the fully calibrated calibrator for use in simulate_trace.
    tgt_actual optionally carries the primary target's full-trace actuals when
    the target is not a model input channel.
    """
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]
    tgt_chan, tgt_vec, mem_chan = _target_actual_lookup(meta, tgt_actual)

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
        elif preds.dim() == 3:
            p = torch.round(preds[0, -1] * 100) / 100
            q10 = p.cpu().numpy()
            q95 = q10.copy()
        else:
            p = torch.round(preds[0] * 100) / 100
            q10 = p.cpu().numpy().ravel()
            q95 = q10.copy()

        cpu_actual = float(tgt_vec[idx]) if tgt_vec is not None else float(raw_feat[idx, tgt_chan])
        mem_actual = float(raw_feat[idx, mem_chan]) if mem_chan is not None else 0.0

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
                ac = float(tgt_vec[aidx]) if tgt_vec is not None else float(raw_feat[aidx, tgt_chan])
                am = float(raw_feat[aidx, mem_chan]) if mem_chan is not None else 0.0
                cal.states["cpu"].update(ac, p["q10"][0], p["q95"][0])
                if num_targets > 1:
                    cal.states["memory"].update(am, p["q10"][1], p["q95"][1])
        pending = [p for p in pending if p["idx"] > idx - pred_horizon]

    print(f"[INFO] Conformal calibration complete ({end_idx - calibration_start} min)")
    return cal


def simulate_trace(raw_feat, model_feat, model, meta, device,
                   start_idx, n_minutes,
                   threshold=BASE_THRESHOLD,
                   num_targets=2,
                   adaptive_cal=None,
                   adaptive_window=500, adaptive_eta=0.01,
                   timestamps=None, tgt_actual=None):
    """tgt_actual optionally carries the primary target's full-trace actuals
    when the target is not a model input channel (e.g. cpu for http_time)."""
    input_len = meta["input_len"]
    pred_horizon = meta["pred_horizon"]
    tgt_chan, tgt_vec, mem_chan = _target_actual_lookup(meta, tgt_actual)

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
        elif preds.dim() == 3:
            p = torch.round(preds[0, -1] * 100) / 100
            q50 = p.cpu().numpy()
            q10, q95 = q50.copy(), q50.copy()
        else:
            p = torch.round(preds[0] * 100) / 100
            q50 = p.cpu().numpy().ravel()
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

        # Primary-target actuals (explicit vector when the target is not a
        # model input; else its input channel). Memory only when the feature
        # set has a memory input; single-target models still record 0 pred.
        cpu_actual = float(tgt_vec[idx]) if tgt_vec is not None else float(raw_feat[idx, tgt_chan])
        mem_actual = float(raw_feat[idx, mem_chan]) if mem_chan is not None else 0.0

        # Online conformal feedback with delayed horizon alignment
        if adaptive_cal is not None:
            pending.append({"idx": idx, "q10": q10.copy(), "q95": q95.copy()})
            matured = [p for p in pending if p["idx"] <= idx - pred_horizon]
            for p in matured:
                aidx = int(p["idx"] + pred_horizon)
                if aidx < n:
                    ac = float(tgt_vec[aidx]) if tgt_vec is not None else float(raw_feat[aidx, tgt_chan])
                    am = float(raw_feat[aidx, mem_chan]) if mem_chan is not None else 0.0
                    adaptive_cal.states["cpu"].update(ac, p["q10"][0], p["q95"][0])
                    if num_targets > 1:
                        adaptive_cal.states["memory"].update(am, p["q10"][1], p["q95"][1])
            pending = [p for p in pending if p["idx"] > idx - pred_horizon]

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
            "inference_time_s": dt,
        })

    return results


# ================================================================
# Metrics
# ================================================================

def _pearson(a, b):
    """Pearson correlation coefficient."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else float("nan")


def _compute_one_step(y_pred, y_true, y_last, y_second_last=None):
    """Compute all per-step metrics matching training/metrics.py.

    MDA: model direction = sign(y_pred - y_last).
    When y_second_last is provided, naive direction = sign(y_last - y_second_last)
    (input trend), otherwise naive also uses sign(y_pred - y_last).
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    n = len(y_true)

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    nonzero = np.abs(y_true) > 1e-12
    if int(np.sum(nonzero)) > 0:
        mape = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero]))) * 100.0
    else:
        mape = 0.0

    actual_dir = np.sign(y_true - y_last)
    pred_dir = np.sign(y_pred - y_last)
    mda = float(np.mean(actual_dir == pred_dir)) * 100.0

    under_mask = y_pred < y_true
    over_mask = y_pred > y_true
    n_under = int(np.sum(under_mask))
    n_over = int(np.sum(over_mask))

    under_rate = (n_under / n * 100.0) if n > 0 else 0.0
    over_rate = (n_over / n * 100.0) if n > 0 else 0.0

    if n_under > 0:
        mean_under = float(np.mean(y_true[under_mask] - y_pred[under_mask]))
        max_under = float(np.max(y_true[under_mask] - y_pred[under_mask]))
    else:
        mean_under = 0.0
        max_under = 0.0

    if n_over > 0:
        mean_over = float(np.mean(y_pred[over_mask] - y_true[over_mask]))
        max_over = float(np.max(y_pred[over_mask] - y_true[over_mask]))
    else:
        mean_over = 0.0
        max_over = 0.0

    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse, "R²": r2,
        "MAPE (%)": mape, "MDA (%)": mda,
        "Under-Pred Rate (%)": under_rate, "Over-Pred Rate (%)": over_rate,
        "Mean Under Error": mean_under, "Mean Over Error": mean_over,
        "Max Under Error": max_under, "Max Over Error": max_over,
    }


METRIC_NAMES = [
    "MSE", "MAE", "RMSE", "R²", "MAPE (%)", "MDA (%)",
    "Under-Pred Rate (%)", "Over-Pred Rate (%)",
    "Mean Under Error", "Mean Over Error",
    "Max Under Error", "Max Over Error",
]
PCT_METRICS = {"MAPE (%)", "MDA (%)", "Under-Pred Rate (%)", "Over-Pred Rate (%)"}


def _delta_pct(model_val, naive_val, is_pct_metric=False):
    if is_pct_metric:
        diff = model_val - naive_val
        return f"{diff:+.1f}"
    denom = abs(naive_val)
    if denom < 1e-12:
        return "N/A"
    pct = (model_val - naive_val) / denom * 100.0
    return f"{pct:+.1f}"


def _report_entries(target_features=None):
    """Map model target features to (display, act, pred, lo, hi) result columns.

    Only actual forecast targets are reported (e.g. cpu_mem_rpc reports CPU
    only, not memory). Unknown/non-cpu-mem targets fall back to the slot the
    simulator stored them in (first target -> cpu columns, second -> memory).
    """
    cpu_entry = ("cpu", "cpu", "pred_cpu", "lower_cpu", "upper_cpu")
    mem_entry = ("memory", "memory", "pred_mem", "lower_mem", "upper_mem")
    if not target_features:
        return [cpu_entry, mem_entry]
    entries = []
    if "cpu_utilization" in target_features:
        entries.append(cpu_entry)
    if "memory_utilization" in target_features:
        entries.append(mem_entry)
    if entries:
        return entries
    # Generic fallback for feature sets with neither cpu nor memory targets
    # (e.g. mcr_http): label with the feature name, read from the used slot.
    for i, tf in enumerate(target_features):
        slot = cpu_entry if i == 0 else mem_entry
        entries.append((str(tf), slot[1], slot[2], slot[3], slot[4]))
    return entries


def compute_metrics(results, pred_horizon, threshold, use_conformal=False,
                    target_features=None):
    """Compute all metrics from training/metrics.py plus persistence diagnostics.

    Only the actual forecast targets are evaluated/reported (derived from the
    checkpoint's feature set); non-target inputs (e.g. memory when the target
    is CPU-only) are skipped.
      - Model metrics: last-step and naive forecaster side-by-side
      - Naive forecaster: persistence (current load = prediction for future)
      - Persistence diagnostics: Pearson correlations, R², beat-rate, MAE ratio
      - Conformal interval quality (PICP, MPIW) when applicable
    """
    df = pd.DataFrame(results)
    if df.empty:
        print("No evaluation data")
        return {}

    m = {}
    header_printed = False

    for target_name, col_act, col_pred, col_lo, col_hi in _report_entries(target_features):
        actual_all = df[col_act].values.astype(float)

        # Model prediction (horizon-aligned)
        if use_conformal and col_hi in df.columns:
            model_pred_all = np.roll(df[col_hi].values.astype(float), pred_horizon)
        else:
            model_pred_all = np.roll(df[col_pred].values.astype(float), pred_horizon)

        # Naive forecaster: persistence baseline
        # At minute t, predict actual[t] for minute t+pred_horizon
        naive_pred_all = actual_all.copy()

        # Build y_last for MDA: value at prediction time (pred_horizon steps ago)
        y_last_all = np.roll(actual_all, pred_horizon)
        y_second_last_all = np.roll(actual_all, pred_horizon + 1)

        # Drop first pred_horizon entries (no prediction available)
        model_pred_all[:pred_horizon] = np.nan
        naive_pred_all[pred_horizon:] = naive_pred_all[:len(naive_pred_all) - pred_horizon]
        naive_pred_all[:pred_horizon] = np.nan
        y_last_all[:pred_horizon] = np.nan
        y_second_last_all[:pred_horizon] = np.nan

        valid = ~np.isnan(model_pred_all)
        actual = actual_all[valid]
        model_pred = model_pred_all[valid]
        naive_pred = naive_pred_all[valid]
        y_last = y_last_all[valid]
        y_second_last = y_second_last_all[valid]

        n = len(actual)

        # --- Model metrics ---
        model_m = _compute_one_step(model_pred, actual, y_last, y_second_last)

        # --- Naive/persistence metrics ---
        naive_m = _compute_one_step(naive_pred, actual, y_last)
        # Override naive MDA: use input trend direction sign(y_last - y_second_last)
        # as the naive's "predicted direction" (trend-based forecast)
        actual_dir = np.sign(actual - y_last)
        naive_trend_dir = np.sign(y_last - y_second_last)
        naive_m["MDA (%)"] = float(np.mean(actual_dir == naive_trend_dir)) * 100.0

        # --- Persistence diagnostics ---
        corr_pred_true = _pearson(model_pred, actual)
        corr_pred_current = _pearson(model_pred, y_last)
        corr_current_true = _pearson(y_last, actual)
        mse_model = float(np.mean((model_pred - actual) ** 2))
        mse_naive = float(np.mean((y_last - actual) ** 2))
        r2_vs_persistence = 1.0 - mse_model / mse_naive if mse_naive > 1e-12 else float("nan")
        mae_model = float(np.mean(np.abs(model_pred - actual)))
        mae_naive_val = float(np.mean(np.abs(y_last - actual)))
        mae_vs_persistence = mae_model / mae_naive_val if mae_naive_val > 1e-12 else float("nan")
        beat_persistence = float(np.mean(np.abs(model_pred - actual) < np.abs(y_last - actual)) * 100.0)

        # --- Store all metrics ---
        for name in METRIC_NAMES:
            m[f"{target_name}_{name}_last_step"] = model_m[name]
            m[f"{target_name}_{name}_naive"] = naive_m[name]
            d = _delta_pct(model_m[name], naive_m[name],
                           is_pct_metric=(name in PCT_METRICS))
            m[f"{target_name}_{name}_delta"] = d

        m[f"{target_name}_corr_pred_true"] = corr_pred_true
        m[f"{target_name}_corr_pred_current"] = corr_pred_current
        m[f"{target_name}_corr_current_true"] = corr_current_true
        m[f"{target_name}_r2_vs_persistence"] = r2_vs_persistence
        m[f"{target_name}_beat_persistence"] = beat_persistence
        m[f"{target_name}_mae_vs_persistence"] = mae_vs_persistence

        # --- Print table ---
        if not header_printed:
            print(f"\n{'Metric':<28s} {'Model':>14s} {'Naive':>14s} {'Δ (%)':>10s}")
            print("-" * 70)
            header_printed = True

        print(f"\n--- {target_name.upper()} ---")
        for name in METRIC_NAMES:
            ls = model_m[name]
            nv = naive_m[name]
            d = _delta_pct(ls, nv, is_pct_metric=(name in PCT_METRICS))
            if name in PCT_METRICS:
                print(f"  {name:<26s} {ls:>13.4f}% {nv:>13.4f}% {d:>10s}")
            else:
                print(f"  {name:<26s} {ls:>14.8e} {nv:>14.8e} {d:>10s}")

        print(f"\n  Persistence Diagnostics ({target_name.upper()}):")
        print(f"    ρ(pred, truth)        {corr_pred_true:>10.4f}")
        print(f"    ρ(pred, current)      {corr_pred_current:>10.4f}")
        print(f"    ρ(current, truth)     {corr_current_true:>10.4f}")
        print(f"    R² vs persistence     {r2_vs_persistence:>10.4f}")
        print(f"    Beat-persistence (%)  {beat_persistence:>10.2f}")
        print(f"    MAE vs persistence    {mae_vs_persistence:>10.4f}")

    # --- Conformal interval quality (PICP, MPIW) ---
    if use_conformal and "upper_cpu" in df.columns:
        print(f"\n--- Conformal Interval Quality ---")
        for tname, col_act, col_lo, col_hi in [
            (e[0], e[1], e[3], e[4]) for e in _report_entries(target_features)
        ]:
            a = df[col_act].values.astype(float)
            lo = df[col_lo].values.astype(float)
            hi = df[col_hi].values.astype(float)
            in_interval = (a >= lo) & (a <= hi)
            picp = float(np.mean(in_interval))
            mpiw = float(np.mean(hi - lo))
            m[f"{tname}_picp"] = picp
            m[f"{tname}_mpiw"] = mpiw
            print(f"  {tname.upper()} PICP: {picp:.1%}   MPIW: {mpiw:.6f}")

    m["n_evaluation_minutes"] = len(df)
    return m


# ================================================================
# Plotting
# ================================================================

def _mcr_display_name(feature_name):
    """Human-readable panel label for an MCR feature (http_mcr -> HTTP MCR)."""
    low = feature_name.lower()
    if "http" in low:
        return "HTTP MCR"
    if "providerrpc" in low:
        return "Provider RPC MCR"
    if "consumerrpc" in low:
        return "Consumer RPC MCR"
    if "providermq" in low:
        return "Provider MQ MCR"
    if "consumermq" in low:
        return "Consumer MQ MCR"
    return feature_name.replace("_", " ").upper()


def _target_display_name(feature_name):
    """Human-readable label for the forecast target (cpu_utilization -> CPU)."""
    low = feature_name.lower()
    if low == "cpu_utilization":
        return "CPU"
    if low == "memory_utilization":
        return "Memory"
    if "mcr" in low or "rpc" in low or "http" in low:
        return _mcr_display_name(feature_name)
    return feature_name.replace("_", " ").upper()


def plot_results(results, msname, plots_dir, pred_horizon, use_conformal, threshold,
                  num_targets=2, raw_feat=None, feature_names=None, start_idx=0,
                  target_features=None):
    """Panels are dynamic: the primary-target pair (unshifted + shifted) is
    always drawn; the memory panel only when the set has a memory input; the
    MCR panel only for the first MCR input. Anything absent from the set is
    not plotted, and derived time features (minute/hour/day) never are.
    """
    df = pd.DataFrame(results)
    has_ts = df["timestamp"].notna().all()
    x = df["timestamp"] if has_ts else np.arange(len(df))

    os.makedirs(plots_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    feats = list(feature_names) if feature_names else []
    primary = (target_features[0] if target_features
               else "cpu_utilization")
    tlabel = _target_display_name(primary)
    has_mem = "memory_utilization" in feats
    mem_idx = feats.index("memory_utilization") if has_mem else None

    actual_cpu = df["cpu"].values.astype(float)
    pred_cpu = df["pred_cpu"].values.astype(float)
    actual_mem = df["memory"].values.astype(float) if "memory" in df.columns else None
    pred_mem = df["pred_mem"].values.astype(float) if "pred_mem" in df.columns else None

    # Results cover the eval window [start_idx, start_idx+len(df)) of the full
    # trace. Older runs stored zeros in df["memory"] for single-target models;
    # fall back to the raw memory channel so the panel is never flat-zero when
    # the data exists. Only when the set actually has a memory input (else no
    # memory panel is drawn at all).
    if (actual_mem is None or not np.any(actual_mem)) and raw_feat is not None \
            and has_mem and mem_idx is not None \
            and mem_idx < raw_feat.shape[1]:
        actual_mem = raw_feat[start_idx:start_idx + len(df), mem_idx].astype(float)
    if not has_mem:
        actual_mem, pred_mem = None, None

    # MCR context feature aligned to the eval window (not the trace head):
    # first MCR input in the set (derived time features never qualify).
    mcr_full, mcr_label = None, "MCR"
    if raw_feat is not None and feats:
        for fi, fn in enumerate(feats):
            if 'mcr' in fn.lower() and not is_derived_feature(fn):
                if fi < raw_feat.shape[1]:
                    mcr_full = raw_feat[start_idx:start_idx + len(df), fi].astype(float)
                    mcr_label = _mcr_display_name(fn)
                break

    zoom_window = 60

    def _find_zoom_center(actual):
        best_std = -1
        best_center = zoom_window
        for i in range(zoom_window, len(actual) - zoom_window):
            s = np.std(actual[i - zoom_window:i + zoom_window])
            if s > best_std:
                best_std = s
                best_center = i
        return best_center

    def _plot_panels(x_vals, actual_c, pred_c, actual_m, pred_m, title_suffix,
                     zoom_start=None, zoom_end=None, filename=None,
                     mcr_vals=None, mark_spikes=False,
                     spike_threshold=threshold):
        if zoom_start is not None:
            mask = np.arange(len(x_vals)) >= zoom_start
            if zoom_end is not None:
                mask &= np.arange(len(x_vals)) < zoom_end
            x_vals = x_vals[mask]
            actual_c = actual_c[mask]
            pred_c = pred_c[mask]
            if actual_m is not None:
                actual_m = actual_m[mask]
            if pred_m is not None:
                pred_m = pred_m[mask]
            if mcr_vals is not None:
                mcr_vals = mcr_vals[mask]

        # Minutes where the actual target exceeds the spike threshold: dashed
        # vertical guides drawn through every subplot (zoom view only).
        spike_idx = (np.where(np.asarray(actual_c, dtype=float) > spike_threshold)[0]
                     if mark_spikes else np.zeros(0, dtype=int))

        def _spike_lines(ax):
            if len(spike_idx) == 0:
                return
            xv = np.asarray(x_vals)
            for k, si in enumerate(spike_idx):
                ax.axvline(xv[si], color="red", linestyle="--",
                           linewidth=1.0, alpha=0.5,
                           label=f"{tlabel} > {spike_threshold:g}" if k == 0 else None)

        # Horizon alignment: a prediction made at minute t targets t+h, so the
        # prediction shifts RIGHT by h while actual load stays at its own
        # timestamps. Title stats use the same aligned pairing.
        h = max(int(pred_horizon), 0)
        a_al = np.asarray(actual_c, dtype=float)
        p_al = np.asarray(pred_c, dtype=float)
        if 0 < h < len(a_al):
            a_al, p_al = a_al[h:], p_al[:-h]

        # Panel set is data-driven: target pair always; memory/MCR only when
        # the feature set has them (time features never get a panel).
        show_mem = has_mem and actual_m is not None and np.any(actual_m)
        show_mcr = mcr_vals is not None and len(mcr_vals) == len(x_vals)
        n_panels = 2 + (1 if show_mem else 0) + (1 if show_mcr else 0)

        fig, axes = plt.subplots(n_panels, 1, figsize=(22, 4 * n_panels), sharex=True)
        fig.suptitle(f"{msname} — {title_suffix}\n"
                     f"MSE={np.mean((p_al - a_al)**2):.6f}, "
                     f"MAE={np.mean(np.abs(p_al - a_al)):.6f}",
                     fontsize=13, fontweight="bold", y=0.99)

        ai = 0
        ax = axes[ai]; ai += 1
        ax.plot(x_vals, actual_c, color='#1976D2', linewidth=1.5, label=f'Actual {tlabel}')
        ax.plot(x_vals, pred_c, color='#FF5722', linewidth=1.5, alpha=0.8, label=f'Predicted {tlabel}')
        _spike_lines(ax)
        ax.set_ylim(0, 1); ax.set_ylabel(tlabel)
        ax.set_title(f'Predicted vs Actual {tlabel} (unshifted)')
        ax.legend(); ax.grid(True, alpha=0.2)

        ax = axes[ai]; ai += 1
        ax.plot(x_vals, actual_c, color='#1976D2', linewidth=1.5, label=f'Actual {tlabel}')
        if 0 < h < len(x_vals):
            ax.plot(np.asarray(x_vals)[h:], np.asarray(pred_c, dtype=float)[:-h],
                    color='#FF5722', linewidth=1.5, alpha=0.8,
                    label=f'Predicted {tlabel} (shifted +{h})')
        else:
            ax.plot(x_vals, pred_c, color='#FF5722', linewidth=1.5, alpha=0.8,
                    label=f'Predicted {tlabel}')
        _spike_lines(ax)
        ax.set_ylim(0, 1); ax.set_ylabel(tlabel)
        ax.set_title(f'Predicted vs Actual {tlabel} (prediction shifted +{h}: '
                     f'pred[t] at actual[t+{h}])')
        ax.legend(); ax.grid(True, alpha=0.2)

        if show_mem:
            ax = axes[ai]; ai += 1
            ax.plot(x_vals, actual_m, color='#4CAF50', linewidth=1.5, label='Actual Memory')
            if pred_m is not None and np.any(pred_m):
                ax.plot(x_vals, pred_m, color='#FF9800', linewidth=1.5, alpha=0.8,
                        label='Predicted Memory')
            _spike_lines(ax)
            ax.set_ylim(0, 1); ax.set_ylabel('Memory')
            ax.set_title('Memory')
            ax.legend(); ax.grid(True, alpha=0.2)

        if show_mcr:
            ax = axes[ai]; ai += 1
            ax.plot(x_vals, mcr_vals, color='#9C27B0', linewidth=1.5, label=mcr_label)
            _spike_lines(ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel(mcr_label)
            ax.set_title(mcr_label)
            ax.set_xlabel('Minute'); ax.legend(); ax.grid(True, alpha=0.2)
        else:
            axes[ai - 1].set_xlabel('Minute')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filepath = os.path.join(plots_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {filepath}")
        return filepath

    x_range = np.arange(len(df))

    full_path = _plot_panels(
        x_range, actual_cpu, pred_cpu,
        actual_mem, pred_mem,
        "Full Test Set (unshifted + shifted, h={})".format(pred_horizon),
        filename=f"hpa_sim_{msname}_full_{stamp}.png",
        mcr_vals=mcr_full,
    )

    zoom_center = _find_zoom_center(actual_cpu)
    zoom_start = max(0, zoom_center - zoom_window)
    zoom_end = min(len(df), zoom_center + zoom_window)

    zoom_path = _plot_panels(
        x_range, actual_cpu, pred_cpu,
        actual_mem, pred_mem,
        f"Zoomed (min {zoom_start}-{zoom_end}, highest-std window)",
        zoom_start=zoom_start, zoom_end=zoom_end,
        filename=f"hpa_sim_{msname}_zoom_{stamp}.png",
        mcr_vals=mcr_full, mark_spikes=True,
    )

    return full_path, zoom_path


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
    service_data, target_data, cache_feats, _ = load_alibaba_parquet(
        args.parquet_root, meta["feature_set"],
        windows_dir=args.windows_dir,
    )

    if args.msname:
        msname = args.msname
        if msname not in service_data:
            resolved = resolve_msname(msname, service_data)
            if resolved is not None:
                msname = resolved
            else:
                available = sorted(service_data.keys())
                preview = ", ".join(available[:20])
                hint = ""
                if f"MS_{msname}" in service_data:
                    hint = f" Did you mean 'MS_{msname}'?"
                raise SystemExit(
                    f"msname '{msname}' not found in data.{hint} "
                    f"Available ({len(available)}): {preview}"
                    f"{'...' if len(available) > 20 else ''} "
                    f"(cache index: {args.windows_dir or DEFAULT_SERVICE_INDEX}; "
                    f"if you changed --msname with --skip_preprocessing, rebuild "
                    f"with --recompute_windows)"
                )
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

    # Primary-target actuals. When the target is not a model input channel
    # (e.g. cpu for http_time), they come from the separately loaded target
    # data; otherwise the trace functions read the target's input channel.
    spec = get_feature_set(meta["feature_set"])
    primary_target = spec["targets"][0]
    if primary_target in spec["features"]:
        tgt_actual = None
    else:
        extras = [t for t in spec["targets"] if t not in spec["features"]]
        if target_data is None or msname not in target_data:
            raise SystemExit(
                f"Target '{primary_target}' is not a model input and no target "
                f"data was loaded for '{msname}'. Rebuild windows for "
                f"feature_set='{meta['feature_set']}'."
            )
        tgt_actual = target_data[msname][:, extras.index(primary_target)].astype(np.float32)
        print(f"Target actuals '{primary_target}': {tgt_actual.shape}")

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
            tgt_actual=tgt_actual,
        )
    else:
        adaptive_cal = None

    print(f"Evaluation phase: {args.hours}h ({eval_minutes} min) "
          f"[{test_start}:{test_start + eval_minutes}]")
    results = simulate_trace(
        raw_feat, model_feat, model, meta, device,
        start_idx=test_start, n_minutes=eval_minutes,
        threshold=args.threshold,
        num_targets=meta["num_targets"],
        adaptive_cal=adaptive_cal,
        adaptive_window=args.adaptive_window,
        adaptive_eta=args.adaptive_eta,
        timestamps=timestamps,
        tgt_actual=tgt_actual,
    )

    if not results:
        raise SystemExit("No simulation results. Check --hours and test split size.")

    report_targets = target_features_for_feature_set(meta["feature_set"])
    metrics = compute_metrics(results, meta["pred_horizon"], args.threshold,
                              use_conformal=args.adaptive_conformal,
                              target_features=report_targets)

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
    print("=" * 72)

    # Metrics are printed by compute_metrics; now print summary JSON keys
    # (only actual forecast targets, not fixed CPU+Memory).
    for tgt, _, _, _, _ in _report_entries(report_targets):
        print(f"\n{tgt.upper()} Summary:")
        print(f"  R² (model):          {metrics.get(f'{tgt}_R²_last_step', 0):>10.4f}")
        print(f"  R² (naive):          {metrics.get(f'{tgt}_R²_naive', 0):>10.4f}")
        print(f"  Beat-persistence:    {metrics.get(f'{tgt}_beat_persistence', 0):>10.2f}%")
        print(f"  ρ(pred, truth):      {metrics.get(f'{tgt}_corr_pred_true', 0):>10.4f}")
    if args.adaptive_conformal:
        for tgt, _, _, _, _ in _report_entries(report_targets):
            if metrics.get(f"{tgt}_picp") is not None:
                print(f"  {tgt.upper()} PICP: {metrics.get(f'{tgt}_picp', 0):.1%}   "
                      f"MPIW: {metrics.get(f'{tgt}_mpiw', 0):.6f}")
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

    feature_names = feature_names_for_feature_set(meta["feature_set"])
    plot_results(results, msname, args.plots_dir, meta["pred_horizon"],
                 args.adaptive_conformal, args.threshold,
                 num_targets=meta["num_targets"],
                 raw_feat=raw_feat, feature_names=feature_names,
                 start_idx=test_start, target_features=report_targets)


if __name__ == "__main__":
    main()

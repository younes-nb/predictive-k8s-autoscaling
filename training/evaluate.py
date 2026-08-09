import os
import sys
import random
import argparse
import logging
import time
import math
import warnings
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.distributed\.all_gather_into_tensor` is deprecated.*",
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

from shared.config_paths import PATHS
from shared.config_training_defaults import TRAINING
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.logging_utils import setup_logging
from shared.features import target_features_for_feature_set, feature_names_for_feature_set
from core.dataset import ShardedWindowsDataset

from training.metrics import compute_metrics, find_max_inference_batch_size
from training.train_helpers import head_slice_dataset_by_pct
from types import SimpleNamespace
from training.sfoa_configs import get_config


MODEL_TYPES = ("lstm", "gru", "bilstm", "bigrue", "cnn_bilstm", "dlinear", "dpam")
PREPROCESS_APPROACHES = ("none", "smoothing", "swt", "cskv")


SWT_FIELD_MAP = {
    "swt_level": "SWT_LEVEL", "mem_swt_level": "MEM_SWT_LEVEL",
}


def _preprocess_raw_window(x_np, preprocess_approach, args=None):
    if preprocess_approach == "none":
        return x_np
    elif preprocess_approach == "smoothing":
        from preprocessing.smooth_windows import smooth_array
        window_size = getattr(args, "smoothing_window", 5) if args is not None else 5
        return smooth_array(x_np, window_size=window_size)
    elif preprocess_approach == "swt":
        from dataclasses import replace
        from preprocessing.swt.decomposition import decompose_window
        from preprocessing.swt.config import CFG as SWT_CFG
        base = SWT_CFG
        if args is not None:
            overrides = {}
            for cli_attr, cfg_attr in SWT_FIELD_MAP.items():
                val = getattr(args, cli_attr, None)
                if val is not None:
                    overrides[cfg_attr] = val
            if overrides:
                base = replace(SWT_CFG, **overrides)
        cpu_cfg = base
        mem_cfg = replace(
            base,
            SWT_LEVEL=base.MEM_SWT_LEVEL,
        )

        channels = []
        for f in range(x_np.shape[1]):
            cfg = mem_cfg if f == 1 else cpu_cfg
            ch = decompose_window(x_np[:, f].astype(np.float64), cfg)
            channels.append(ch)
        stacked = np.concatenate(channels, axis=0)
        return stacked.T
    elif preprocess_approach == "cskv":
        from preprocessing.cskv.decomposition import (
            ceemdan_decompose, cluster_imfs, vmd_decompose,
        )
        from preprocessing.cskv.config import CFG as CSKV_CFG
        sig = x_np[:, 0].astype(np.float64)
        imfs, residue = ceemdan_decompose(
            sig, CSKV_CFG.CEEMDAN_EPSILON, CSKV_CFG.CEEMDAN_TRIALS,
        )
        co_imfs = cluster_imfs(
            imfs, residue,
            m=CSKV_CFG.SE_M,
            r_frac=CSKV_CFG.SE_R_FRAC,
            max_se_samples=CSKV_CFG.SE_MAX_SAMPLES,
            n_clusters=CSKV_CFG.N_CLUSTERS,
        )
        vmd_modes = vmd_decompose(
            co_imfs[0],
            K=CSKV_CFG.VMD_K,
            alpha=CSKV_CFG.VMD_ALPHA,
            tau=CSKV_CFG.VMD_TAU,
            DC=CSKV_CFG.VMD_DC,
            init=CSKV_CFG.VMD_INIT,
            tol=CSKV_CFG.VMD_TOL,
        )
        channel_list = [vmd_modes[k].astype(np.float32) for k in range(vmd_modes.shape[0])]
        for k in range(1, CSKV_CFG.N_CLUSTERS):
            channel_list.append(np.asarray(co_imfs[k], dtype=np.float32))
        return np.stack(channel_list, axis=1)
    else:
        raise ValueError(f"Unknown preprocess_approach: {preprocess_approach}")


def _benchmark_worker(idx, raw_ds, model, preprocess_approach, device, args=None):
    x_raw, *_ = raw_ds[idx]
    x_np = x_raw.numpy()

    t0 = time.perf_counter()

    x_processed = _preprocess_raw_window(x_np, preprocess_approach, args)
    x_tensor = torch.from_numpy(x_processed).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _build_model_from_checkpoint(checkpoint, input_size, device):
    ckpt_args = checkpoint.get("args", {})
    model_type = checkpoint.get("model_type", "lstm")
    num_targets = len(target_features_for_feature_set(ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)))

    cfg = get_config(model_type)
    hyperparams = checkpoint.get("hyperparams", cfg.DEFAULTS)
    return cfg.build_model(hyperparams, input_size, SimpleNamespace(**ckpt_args), num_targets, device)


def _near_constant_valid_indices(ds, target_idxs):
    valid = []
    for i in range(len(ds)):
        x, *_ = ds[i]
        x_np = x.numpy()
        if any(np.std(x_np[:, f].astype(np.float64)) < 1e-12 for f in target_idxs):
            continue
        valid.append(i)
    return valid


def _filter_near_constant_windows(ds, target_idxs):
    valid = _near_constant_valid_indices(ds, target_idxs)
    if len(valid) == len(ds):
        return ds
    return Subset(ds, valid)


def _load_test_dataset(args, ckpt_args, device, log_info, feature_set_name="cpu"):
    split = getattr(args, "split", "test")
    pct = getattr(args, "val_pct", None) if split == "val" else getattr(args, "test_pct", 100.0)
    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    model_type = ckpt_args.get("model_type", "lstm")
    preprocess_approach = ckpt_args.get("preprocess_approach", "none")

    target_features = target_features_for_feature_set(feature_set_name)
    feature_names = feature_names_for_feature_set(feature_set_name)
    target_idxs_in_features = [feature_names.index(f) for f in target_features]

    if preprocess_approach == "none":
        test_ds = ShardedWindowsDataset(
            args.windows_dir, split, input_len, horizon
        )
        test_ds = _filter_near_constant_windows(test_ds, target_idxs_in_features)
        total_test_samples = len(test_ds)
        test_ds = head_slice_dataset_by_pct(test_ds, pct)
        log_info(f"{split.capitalize()} samples (Total): {total_test_samples}")
        log_info(
            f"{split.capitalize()} samples (Used):  {len(test_ds)}/{total_test_samples} "
            f"({float(pct):g}%)"
        )
        if len(test_ds) > 0:
            first_x, *_ = test_ds[0]
            input_size = first_x.shape[-1]
        else:
            input_size = 1
        return test_ds, input_size
    elif preprocess_approach == "smoothing":
        smooth_dir = getattr(args, "preprocess_dir", None) or args.windows_dir
        smooth_ds = ShardedWindowsDataset(
            smooth_dir, split, input_len, horizon
        )
        raw_ds = ShardedWindowsDataset(
            args.windows_dir, split, input_len, horizon
        )
        valid = _near_constant_valid_indices(raw_ds, target_idxs_in_features)
        test_ds = Subset(smooth_ds, valid) if len(valid) < len(smooth_ds) else smooth_ds
        total_test_samples = len(test_ds)
        test_ds = head_slice_dataset_by_pct(test_ds, pct)
        log_info(f"{split.capitalize()} samples (Total): {total_test_samples}")
        log_info(
            f"{split.capitalize()} samples (Used):  {len(test_ds)}/{total_test_samples} "
            f"({float(pct):g}%)"
        )
        if len(test_ds) > 0:
            first_x, *_ = test_ds[0]
            input_size = first_x.shape[-1]
        else:
            input_size = 1
        return test_ds, input_size
    elif preprocess_approach == "swt":
        preprocess_dir = getattr(args, "preprocess_dir", None)
        if not preprocess_dir:
            raise RuntimeError("--preprocess_dir required for swt evaluate")
        from preprocessing.swt.dataset import SwtDataset
        from preprocessing.swt.config import CFG as SWT_CFG
        swt_kw = dict(
            input_len=input_len, pred_horizon=horizon,
            feature_set=feature_set_name,
        )
        for attr, cli_arg in [
            ("swt_level", "swt_level"), ("mem_swt_level", "mem_swt_level"),
        ]:
            val = getattr(args, cli_arg, None)
            if val is not None:
                swt_kw[attr] = val
        test_ds_full = SwtDataset(preprocess_dir, split, **swt_kw)
        input_size = test_ds_full.n_channels
        test_ds = head_slice_dataset_by_pct(test_ds_full, pct)
        log_info(f"{split.capitalize()} samples (SWT): {len(test_ds)}/{len(test_ds_full)} ({float(pct):g}%)")
        return test_ds, input_size
    elif preprocess_approach == "cskv":
        preprocess_dir = getattr(args, "preprocess_dir", None)
        if not preprocess_dir:
            raise RuntimeError("--preprocess_dir required for cskv evaluate")
        from preprocessing.cskv.dataset import CskvDataset
        from preprocessing.cskv.config import CFG as CSKV_CFG
        test_ds = CskvDataset(
            preprocess_dir, split,
            input_len=input_len, pred_horizon=horizon,
        )
        input_size = test_ds.total_channels
        return test_ds, input_size
    else:
        raise ValueError(f"Unknown preprocess_approach: {preprocess_approach}")


def _prepare_benchmark_indices(args, ckpt_args, log_info):
    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    preprocess_approach = ckpt_args.get("preprocess_approach", "none")
    n_bench = getattr(args, "inference_bench_samples", 0)

    raw_ds = ShardedWindowsDataset(
        args.windows_dir, getattr(args, "split", "test"), input_len, horizon,
    )
    split = getattr(args, "split", "test")
    pct = getattr(args, "val_pct", None) if split == "val" else getattr(args, "test_pct", 100.0)
    raw_ds = head_slice_dataset_by_pct(raw_ds, pct)
    if len(raw_ds) == 0:
        log_info("No raw windows found for inference latency benchmark.")
        return None, None

    if n_bench <= 0:
        indices = list(range(len(raw_ds)))
    else:
        n_samples = min(n_bench, len(raw_ds))
        rng = random.Random(42)
        indices = rng.sample(range(len(raw_ds)), n_samples)

    if preprocess_approach == "swt":
        STDSTD = 1e-12
        _valid = []
        for idx in indices:
            x_raw, *_ = raw_ds[idx]
            x_np = x_raw.numpy()
            if any(np.std(x_np[:, f]) < STDSTD for f in range(x_np.shape[1])):
                continue
            _valid.append(idx)
        n_skipped = len(indices) - len(_valid)
        if n_skipped:
            log_info(f"Skipped {n_skipped} windows with near-zero std for inference benchmark")
        indices = _valid
        if not indices:
            log_info("No valid windows after filtering near-zero std; skipping benchmark.")
            return None, None

    return raw_ds, indices


def _run_single_sample_benchmark(raw_ds, indices, model, device, args, ckpt_args, log_info, label):
    preprocess_approach = ckpt_args.get("preprocess_approach", "none")

    model.eval()

    bench_workers = getattr(args, "bench_workers", 0)
    use_parallel = device.type == "cpu"
    if use_parallel:
        if bench_workers <= 0:
            bench_workers = max(1, int(0.9 * (os.cpu_count() or 1)))
        log_info(f"Parallel benchmark: {bench_workers} worker threads")

    latencies = []

    if use_parallel and bench_workers > 1:
        with ThreadPoolExecutor(max_workers=bench_workers) as executor:
            futures = {
                executor.submit(
                    _benchmark_worker, idx, raw_ds, model,
                    preprocess_approach, device, args,
                ): idx
                for idx in indices
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Benchmark", unit="sample",
            ):
                latencies.append(future.result())
    else:
        for idx in tqdm(indices, desc="Benchmark", unit="sample"):
            latencies.append(
                _benchmark_worker(idx, raw_ds, model, preprocess_approach, device, args)
            )

    latencies = np.array(latencies)

    log_info(f"\n=== Single-Sample Inference Latency Benchmark ({label}) ===")
    log_info(f"Preprocessing:       {preprocess_approach}")
    log_info(f"Samples Benchmarked: {len(latencies)}")
    log_info(f"Min:     {np.min(latencies):.3f} ms")
    log_info(f"P50:     {np.percentile(latencies, 50):.3f} ms")
    log_info(f"P95:     {np.percentile(latencies, 95):.3f} ms")
    log_info(f"Max:     {np.max(latencies):.3f} ms")
    log_info(f"Average: {np.mean(latencies):.3f} ms")


def evaluate(args):
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    seed = getattr(args, "seed", TRAINING.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_info = lambda msg: (
        logging.info(msg) if accelerator.is_local_main_process else None
    )

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging(args.split)

    from preprocessing.swt.config import CFG as SWT_CFG
    swt_defaults = {
        "swt_level": SWT_CFG.SWT_LEVEL,
        "mem_swt_level": SWT_CFG.MEM_SWT_LEVEL,
    }
    for attr, default in swt_defaults.items():
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, default)

    log_info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        log_info(f"{key:<20}: {value}")
    log_info("-" * 30)
    log_info(f"Device: {device} | Distributed Processes: {accelerator.num_processes}")

    if not os.path.exists(args.checkpoint_path):
        if accelerator.is_local_main_process:
            logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    log_info(f"Loading checkpoint: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model_type = checkpoint.get("model_type", "lstm")
    preprocess_approach = checkpoint.get("preprocess_approach", "none")
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    feature_set_name = ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)
    target_features = target_features_for_feature_set(feature_set_name)
    feature_names = feature_names_for_feature_set(feature_set_name)
    num_targets = len(target_features)
    target_idxs_in_features = [feature_names.index(f) for f in target_features]

    log_info(f"Model Type:         {model_type}")
    log_info(f"Preprocess Approach:{preprocess_approach}")
    log_info(f"Target Feature(s):  {target_features}")

    log_info("\n--- Loading Test Dataset ---")
    test_ds, input_size = _load_test_dataset(args, ckpt_args, device, log_info, feature_set_name)

    model = _build_model_from_checkpoint(checkpoint, input_size, device)
    sd = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)[:10]}")
    if unexpected:
        log_info(f"Note: dropping {len(unexpected)} stale checkpoint keys: {sorted(unexpected)[:5]}")

    if device.type != "cpu":
        log_info("Tuning inference batch size to hardware limits...")
        max_batch = find_max_inference_batch_size(model, input_size, args, device)
        safe_batch_size = int(max_batch * 0.9)
        safe_batch_size = 2 ** int(math.log2(max(1, safe_batch_size)))
        log_info(f"Auto-selected per-GPU Inference Batch Size: {safe_batch_size}")
        args.batch_size = safe_batch_size

    system_cores = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() or 1
    optimal_workers = min(system_cores, 4 * gpu_count, 12)
    log_info(f"Dynamically set num_workers to {optimal_workers}")

    def _worker_init_fn(worker_id):
        wseed = seed + worker_id
        random.seed(wseed)
        np.random.seed(wseed)
        torch.manual_seed(wseed)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=(device.type != "cpu"),
        worker_init_fn=_worker_init_fn,
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    if accelerator.is_local_main_process:
        raw_ds, bench_indices = _prepare_benchmark_indices(args, ckpt_args, log_info)
        if raw_ds is not None:
            raw_model = accelerator.unwrap_model(model)
            if device.type != "cpu":
                _run_single_sample_benchmark(
                    raw_ds, bench_indices, raw_model, device,
                    args, ckpt_args, log_info, label=str(device),
                )
                cpu_model = _build_model_from_checkpoint(
                    checkpoint, input_size, torch.device("cpu")
                )
                cpu_missing, _ = cpu_model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                if cpu_missing:
                    raise RuntimeError(f"Checkpoint missing keys (cpu): {sorted(cpu_missing)[:10]}")
                _run_single_sample_benchmark(
                    raw_ds, bench_indices, cpu_model, torch.device("cpu"),
                    args, ckpt_args, log_info, label="cpu",
                )
                del cpu_model
            else:
                _run_single_sample_benchmark(
                    raw_ds, bench_indices, raw_model, device,
                    args, ckpt_args, log_info, label="cpu",
                )
    accelerator.wait_for_everyone()

    all_preds = []
    all_trues = []
    all_lasts = []

    model.eval()
    start_time = time.time()

    for i, batch in enumerate(tqdm(test_loader, desc="Inference", unit="batch")):
        x, y = batch[0], batch[1]

        with torch.no_grad():
            mu = model(x)

        if preprocess_approach in ("swt", "cskv"):
            batch_last = batch[2]
            gathered_mu, gathered_y, gathered_last = accelerator.gather_for_metrics((mu, y, batch_last))
        else:
            gathered_mu, gathered_y, gathered_x = accelerator.gather_for_metrics((mu, y, x))

        if accelerator.is_local_main_process:
            if preprocess_approach in ("swt", "cskv"):
                y_last = gathered_last.cpu().numpy()
            else:
                y_last = gathered_x[:, -1, :].cpu().numpy()
            all_preds.append(gathered_mu.cpu().numpy())
            all_trues.append(gathered_y.cpu().numpy())
            all_lasts.append(y_last)

    if not accelerator.is_local_main_process:
        return

    inference_time = time.time() - start_time

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_last_all = np.concatenate(all_lasts, axis=0)
    if y_last_all.ndim == 1:
        y_last_all = y_last_all[:, np.newaxis]

    total_samples = y_pred.shape[0]

    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    split = args.split
    pct = args.val_pct if split == "val" else args.test_pct
    if preprocess_approach in ("none", "smoothing"):
        raw_ref = ShardedWindowsDataset(
            args.windows_dir, split, input_len, horizon
        )
        valid = _near_constant_valid_indices(raw_ref, target_idxs_in_features)
        raw_ref = Subset(raw_ref, valid) if len(valid) < len(raw_ref) else raw_ref
        raw_ref = head_slice_dataset_by_pct(raw_ref, pct)
        last_lasts = []
        second_lasts = []
        for idx in range(len(raw_ref)):
            x_np = raw_ref[idx][0].numpy()
            last_lasts.append(x_np[-1, :])
            second_lasts.append(x_np[-2, :])
        y_last_all = np.stack(last_lasts, axis=0)
        y_second_last_all = np.stack(second_lasts, axis=0)
    else:
        raw_test_ds = ShardedWindowsDataset(
            args.windows_dir, split, input_len, horizon
        )
        raw_test_ds = head_slice_dataset_by_pct(raw_test_ds, pct)
        raw_second_lasts = []
        for idx in range(len(raw_test_ds)):
            x_raw, *_ = raw_test_ds[idx]
            x_np = x_raw.numpy()
            if any(np.std(x_np[:, f].astype(np.float64)) < 1e-12 for f in target_idxs_in_features):
                continue
            raw_second_lasts.append(x_np[-2, :])
        y_second_last_all = np.stack(raw_second_lasts, axis=0)
    if y_second_last_all.ndim == 1:
        y_second_last_all = y_second_last_all[:, np.newaxis]
    if y_last_all.ndim == 1:
        y_last_all = y_last_all[:, np.newaxis]

    log_info("\n=== Inference Summary ===")
    log_info(f"Model: {model_type}")

    for t_idx, t_name in zip(target_idxs_in_features, target_features):
        y_last_t = y_last_all[:, t_idx]
        y_second_last_t = y_second_last_all[:, t_idx]
        if num_targets > 1:
            y_pred_t = y_pred[:, :, t_idx]
            y_true_t = y_true[:, :, t_idx]
        else:
            y_pred_t = y_pred
            y_true_t = y_true
        compute_metrics(
            y_pred_t, y_true_t, y_last_t, horizon, total_samples, log_info,
            target_name=t_name, y_second_last=y_second_last_t,
        )

    avg_inference_time_ms = (inference_time / max(1, total_samples)) * 1000.0
    log_info(f"Total Inference Time:  {inference_time:.2f}s")
    log_info(f"Avg Latency per Sample:{avg_inference_time_ms:.4f} ms")
    log_info("-" * 30)
    log_info(f"Log Saved to: {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument(
        "--test_pct",
        type=float,
        default=TRAINING.TEST_PCT,
        help="Percentage of test samples for evaluation; 25 means 25%%, not 0.25 (100 uses all; <=0 uses all).",
    )
    p.add_argument(
        "--val_pct",
        type=float,
        default=TRAINING.VAL_PCT,
        help="Percentage of val samples for evaluation when --split val; 25 means 25%%, not 0.25 (100 uses all; <=0 uses all).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Which split to evaluate on (default: %(default)s)",
    )
    p.add_argument("--preprocess_dir", default=None, help="Preprocessing output dir (for smoothing/swt/cskv)")
    p.add_argument("--smoothing_window", type=int, default=5, help="Moving average window size (for 'smoothing' approach)")
    p.add_argument("--seed", type=int, default=TRAINING.SEED, help="Random seed for reproducibility")
    p.add_argument(
        "--inference_bench_samples", type=int, default=1000,
        help="Number of raw windows for single-sample latency benchmark. "
             "<=0 means use all test windows.",
    )
    p.add_argument(
        "--bench_workers", type=int, default=0,
        help="Worker threads for inference benchmark (CPU only). "
             "0 = auto (90%% of CPU cores).",
    )
    p.add_argument("--swt_level", type=int, default=None, help="SWT level for CPU (swt only, default: config)")
    p.add_argument("--mem_swt_level", type=int, default=None, help="SWT level for memory (swt only, default: config)")

    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

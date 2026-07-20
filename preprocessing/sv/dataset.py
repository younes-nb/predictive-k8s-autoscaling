import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.sv.config import CFG as SV_CFG
from shared.config_preprocessing_defaults import PREPROCESSING

logger = logging.getLogger(__name__)

CPU_CHANNEL_DIRS = [f"vmd_mode_{k}" for k in range(10)] + ["D2", "A2"]
MEM_CHANNEL_DIRS = [f"mem_vmd_mode_{k}" for k in range(10)] + ["mem_D2", "mem_A2"]
CPU_N_CHANNELS = len(CPU_CHANNEL_DIRS)


def _load_service_windows(
    svc_name, svc_idx, preprocess_dir, split, input_len, pred_horizon, stride,
    train_frac, val_frac, channel_dirs, n_channels, has_mem,
):
    base = f"service_{svc_idx:05d}.npy"

    original_dir = os.path.join(preprocess_dir, "original")
    orig_path = os.path.join(original_dir, f"service_{svc_idx:05d}.npy")
    if not os.path.exists(orig_path):
        return None
    try:
        original = np.load(orig_path).astype(np.float64)
    except (EOFError, ValueError):
        return None

    mem_original = None
    if has_mem:
        mem_orig_path = os.path.join(preprocess_dir, "mem_original", base)
        if not os.path.exists(mem_orig_path):
            return None
        try:
            mem_original = np.load(mem_orig_path).astype(np.float64)
        except (EOFError, ValueError):
            return None

    channel_arrays = []
    for d in channel_dirs:
        ch_path = os.path.join(preprocess_dir, d, base)
        if not os.path.exists(ch_path):
            break
        try:
            ch_arr = np.load(ch_path).astype(np.float64)
            channel_arrays.append(ch_arr)
        except (EOFError, ValueError):
            break

    if len(channel_arrays) != n_channels:
        return None

    ref_shape = channel_arrays[0].shape
    num_windows = ref_shape[0]
    for arr in channel_arrays:
        if arr.shape[0] != ref_shape[0]:
            return None

    if has_mem and mem_original is not None and len(mem_original) != len(original):
        return None

    idx_tr = int(num_windows * train_frac)
    idx_val = int(num_windows * (train_frac + val_frac))

    if idx_tr == 0 or idx_tr >= idx_val or idx_val >= num_windows:
        return None

    if split == "train":
        w_start, w_end = 0, idx_tr
    elif split == "val":
        w_start, w_end = idx_tr, idx_val
    else:
        w_start, w_end = idx_val, num_windows

    if w_start >= w_end:
        return None

    local_X, local_y, local_last = [], [], []
    for j in range(w_start, w_end):
        pos = j * stride
        if pos + input_len + pred_horizon > len(original):
            continue

        X = np.stack(
            [channel_arrays[c][j] for c in range(n_channels)],
            axis=1,
        ).astype(np.float32)

        cpu_y = original[pos + input_len: pos + input_len + pred_horizon].astype(np.float32)
        cpu_last = original[pos + input_len - 1].astype(np.float32)

        if has_mem and mem_original is not None:
            mem_y = mem_original[pos + input_len: pos + input_len + pred_horizon].astype(np.float32)
            mem_last = mem_original[pos + input_len - 1].astype(np.float32)
            y = np.stack([cpu_y, mem_y], axis=-1)
            last_val = np.stack([cpu_last, mem_last])
        else:
            y = cpu_y
            last_val = cpu_last

        if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any():
            continue

        local_X.append(X)
        local_y.append(y)
        local_last.append(last_val)

    if not local_X:
        return None

    return (
        np.stack(local_X, axis=0),
        np.stack(local_y, axis=0),
        np.stack(local_last, axis=0),
    )


class SvDataset(Dataset):
    def __init__(
        self,
        preprocess_dir: str,
        split: str,
        input_len: int = 60,
        pred_horizon: int = 5,
        stride: int = 5,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        num_workers: int = 0,
        max_services: int = 0,
        feature_set: str = "cpu",
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        self.has_mem = feature_set == "cpu_mem_both"
        self.channel_dirs = CPU_CHANNEL_DIRS + (MEM_CHANNEL_DIRS if self.has_mem else [])
        self.n_channels = len(self.channel_dirs)

        svc_to_idx_path = os.path.join(preprocess_dir, "_svc_to_idx.json")
        if os.path.exists(svc_to_idx_path):
            with open(svc_to_idx_path) as f:
                svc_to_idx = json.load(f)
            logger.info(
                "Loaded service index mapping from %s (%d services)",
                svc_to_idx_path, len(svc_to_idx),
            )
        else:
            svc_to_idx = None

        if svc_to_idx:
            service_items = list(svc_to_idx.items())
            if max_services and len(service_items) > max_services:
                rng = random.Random(42)
                rng.shuffle(service_items)
                service_items = service_items[:max_services]
                service_items.sort(key=lambda x: x[1])
            n_services = len(service_items)
        else:
            import glob
            channel_base = os.path.join(preprocess_dir, self.channel_dirs[0])
            service_files = sorted(glob.glob(os.path.join(channel_base, "service_*.npy")))
            if not service_files:
                raise FileNotFoundError(
                    f"No channel files found in {channel_base}. "
                    "Run preprocess_services.py first."
                )
            service_items = []
            for sf in service_files:
                idx = int(os.path.basename(sf).replace("service_", "").replace(".npy", ""))
                service_items.append((f"svc_{idx}", idx))
            n_services = len(service_items)

        log_interval = max(1, n_services // 10) if n_services > 10 else 1
        t_start = time.time()

        all_parts = []

        if num_workers is not None and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                futures = {
                    exe.submit(
                        _load_service_windows, svc_name, svc_idx,
                        preprocess_dir, split, input_len, pred_horizon,
                        stride, train_frac, val_frac,
                        self.channel_dirs, self.n_channels, self.has_mem,
                    ): (svc_name, svc_idx)
                    for svc_name, svc_idx in service_items
                }
                completed = 0
                for f in as_completed(futures):
                    completed += 1
                    result = f.result()
                    if result is not None:
                        all_parts.append(result)
                    if completed % log_interval == 0:
                        elapsed = time.time() - t_start
                        logger.info(
                            "SvDataset[%s] %d/%d services (%.0f%%) in %.1fs",
                            split, completed, n_services,
                            100 * completed / n_services, elapsed,
                        )
        else:
            for pos, (svc_name, svc_idx) in enumerate(service_items):
                result = _load_service_windows(
                    svc_name, svc_idx,
                    preprocess_dir, split, input_len, pred_horizon,
                    stride, train_frac, val_frac,
                    self.channel_dirs, self.n_channels, self.has_mem,
                )
                if result is not None:
                    all_parts.append(result)
                if (pos + 1) % log_interval == 0:
                    elapsed = time.time() - t_start
                    logger.info(
                        "SvDataset[%s] %d/%d services (%.0f%%) in %.1fs",
                        split, pos + 1, n_services,
                        100 * (pos + 1) / n_services, elapsed,
                    )

        if not all_parts:
            logger.warning(
                "SvDataset[%s]: no valid windows found in %s",
                split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, self.n_channels), dtype=torch.float32)
            if self.has_mem:
                self.y = torch.empty((0, pred_horizon, 2), dtype=torch.float32)
                self.last = torch.empty((0, 2), dtype=torch.float32)
            else:
                self.y = torch.empty((0, pred_horizon), dtype=torch.float32)
                self.last = torch.empty((0,), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.concatenate([p[0] for p in all_parts], axis=0))
            self.y = torch.from_numpy(np.concatenate([p[1] for p in all_parts], axis=0))
            self.last = torch.from_numpy(np.concatenate([p[2] for p in all_parts], axis=0))

        logger.info(
            "SvDataset[%s]: %d windows, X=%s, y=%s from %d services in %.1fs",
            split, len(self.X), tuple(self.X.shape), tuple(self.y.shape),
            n_services, time.time() - t_start,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]


def _smoke_check(preprocess_dir: str, split: str, num_workers: int = 0,
                  feature_set: str = "cpu") -> None:
    from preprocessing.sv.config import CFG
    from shared.config_preprocessing_defaults import PREPROCESSING

    ds = SvDataset(
        preprocess_dir,
        split,
        input_len=PREPROCESSING.INPUT_LEN,
        pred_horizon=PREPROCESSING.PRED_HORIZON,
        stride=PREPROCESSING.STRIDE,
        train_frac=PREPROCESSING.TRAIN_FRAC,
        val_frac=PREPROCESSING.VAL_FRAC,
        num_workers=num_workers,
        feature_set=feature_set,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    expected_x_shape = (PREPROCESSING.INPUT_LEN, ds.n_channels)
    assert tuple(x.shape) == expected_x_shape, \
        f"Bad x shape: {tuple(x.shape)} expected {expected_x_shape}"
    if ds.has_mem:
        expected_y_shape = (PREPROCESSING.PRED_HORIZON, 2)
        expected_last_shape = (2,)
    else:
        expected_y_shape = (PREPROCESSING.PRED_HORIZON,)
        expected_last_shape = ()
    assert tuple(y.shape) == expected_y_shape, \
        f"Bad y shape: {tuple(y.shape)} expected {expected_y_shape}"
    assert tuple(last.shape) == expected_last_shape, \
        f"Bad last shape: {tuple(last.shape)} expected {expected_last_shape}"
    print(f"Dataset windows: {len(ds)}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last={tuple(last.shape)}")
    print("SvDataset smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check SvDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--feature_set", default="cpu")
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split, num_workers=args.num_workers,
                 feature_set=args.feature_set)

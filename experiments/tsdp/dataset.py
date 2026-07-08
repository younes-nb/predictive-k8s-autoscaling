import argparse
import glob
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.tsdp.config import CFG as TSDP_CFG

logger = logging.getLogger(__name__)

CHANNEL_DIRS = ["D1", "D2", "D3", "A3", "low_freq"]
N_CHANNELS = len(CHANNEL_DIRS)


def _load_service_windows(
    sf, preprocess_dir, split, input_len, pred_horizon, stride, train_frac, val_frac,
):
    base = os.path.basename(sf)
    svc_idx = int(base.replace("service_", "").replace(".npy", ""))

    original_dir = os.path.join(preprocess_dir, "original")
    orig_path = os.path.join(original_dir, f"service_{svc_idx:05d}.npy")
    if not os.path.exists(orig_path):
        return None
    try:
        original = np.load(orig_path).astype(np.float64)
    except (EOFError, ValueError):
        return None

    channel_arrays = []
    for d in CHANNEL_DIRS:
        ch_path = os.path.join(preprocess_dir, d, base)
        if not os.path.exists(ch_path):
            break
        try:
            ch_arr = np.load(ch_path).astype(np.float64)
            channel_arrays.append(ch_arr)
        except (EOFError, ValueError):
            break

    if len(channel_arrays) != N_CHANNELS:
        return None

    ref_shape = channel_arrays[0].shape
    num_windows = ref_shape[0]
    for arr in channel_arrays:
        if arr.shape[0] != ref_shape[0]:
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
            [channel_arrays[c][j] for c in range(N_CHANNELS)],
            axis=1,
        ).astype(np.float32)

        y = original[pos + input_len: pos + input_len + pred_horizon].astype(np.float32)
        last_val = original[pos + input_len - 1].astype(np.float32)

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


class TsdpDataset(Dataset):
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
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        channel_base_dirs = [
            os.path.join(preprocess_dir, d) for d in CHANNEL_DIRS
        ]

        service_files = sorted(glob.glob(os.path.join(channel_base_dirs[0], "service_*.npy")))
        if not service_files:
            raise FileNotFoundError(
                f"No channel files found in {channel_base_dirs[0]}. "
                "Run preprocess_services.py first."
            )

        n_services = len(service_files)
        log_interval = max(1, n_services // 10) if n_services > 10 else 1
        t_start = time.time()

        task_args = (
            preprocess_dir, split, input_len, pred_horizon,
            stride, train_frac, val_frac,
        )

        all_parts = []

        if num_workers is not None and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                futures = {
                    exe.submit(_load_service_windows, sf, *task_args): sf
                    for sf in service_files
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
                            "TsdpDataset[%s] %d/%d services (%.0f%%) in %.1fs",
                            split, completed, n_services,
                            100 * completed / n_services, elapsed,
                        )
        else:
            for idx, sf in enumerate(service_files):
                result = _load_service_windows(sf, *task_args)
                if result is not None:
                    all_parts.append(result)
                if (idx + 1) % log_interval == 0:
                    elapsed = time.time() - t_start
                    logger.info(
                        "TsdpDataset[%s] %d/%d services (%.0f%%) in %.1fs",
                        split, idx + 1, n_services,
                        100 * (idx + 1) / n_services, elapsed,
                    )

        if not all_parts:
            logger.warning(
                "TsdpDataset[%s]: no valid windows found in %s",
                split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, N_CHANNELS), dtype=torch.float32)
            self.y = torch.empty((0, pred_horizon), dtype=torch.float32)
            self.last = torch.empty((0,), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.concatenate([p[0] for p in all_parts], axis=0))
            self.y = torch.from_numpy(np.concatenate([p[1] for p in all_parts], axis=0))
            self.last = torch.from_numpy(np.concatenate([p[2] for p in all_parts], axis=0))

        logger.info(
            "TsdpDataset[%s]: %d windows from %d service files in %.1fs",
            split, len(self.X), n_services, time.time() - t_start,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]


def _smoke_check(preprocess_dir: str, split: str, num_workers: int = 0) -> None:
    from experiments.tsdp.config import CFG

    ds = TsdpDataset(
        preprocess_dir,
        split,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC,
        val_frac=CFG.VAL_FRAC,
        num_workers=num_workers,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    expected_x_shape = (CFG.INPUT_LEN, N_CHANNELS)
    assert tuple(x.shape) == expected_x_shape, \
        f"Bad x shape: {tuple(x.shape)} expected {expected_x_shape}"
    assert tuple(y.shape) == (CFG.PRED_HORIZON,), \
        f"Bad y shape: {tuple(y.shape)}"
    assert last.dim() == 0, f"Bad last shape: {tuple(last.shape)}"
    print(f"Dataset windows: {len(ds)}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last_dim={last.dim()}")
    print("TsdpDataset smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check TsdpDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split, num_workers=args.num_workers)

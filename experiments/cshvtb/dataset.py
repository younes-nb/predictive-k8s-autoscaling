import argparse
import glob
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CHANNEL_DIRS = ["vmd_0", "vmd_1", "vmd_2", "vmd_3", "vmd_4", "lowfreq_0"]
N_CHANNELS = len(CHANNEL_DIRS)


class CshvtbDataset(Dataset):
    def __init__(
        self,
        preprocess_dir: str,
        split: str,
        input_len: int = 60,
        pred_horizon: int = 5,
        stride: int = 5,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        channel_base_dirs = [
            os.path.join(preprocess_dir, d) for d in CHANNEL_DIRS
        ]
        original_dir = os.path.join(preprocess_dir, "original")

        service_files = sorted(glob.glob(os.path.join(channel_base_dirs[0], "service_*.npy")))
        if not service_files:
            raise FileNotFoundError(
                f"No channel files found in {channel_base_dirs[0]}. "
                "Run preprocess_services.py first."
            )

        all_X, all_y, all_last = [], [], []

        for sf in service_files:
            base = os.path.basename(sf)
            svc_idx = int(base.replace("service_", "").replace(".npy", ""))

            orig_path = os.path.join(original_dir, f"service_{svc_idx:05d}.npy")
            if not os.path.exists(orig_path):
                logger.warning("Original signal not found: %s", orig_path)
                continue
            try:
                original = np.load(orig_path).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted original %s: %s", orig_path, e)
                continue

            channel_arrays = []
            for d in CHANNEL_DIRS:
                ch_path = os.path.join(preprocess_dir, d, base)
                if not os.path.exists(ch_path):
                    logger.warning("Channel file not found: %s", ch_path)
                    break
                try:
                    ch_arr = np.load(ch_path).astype(np.float64)
                    channel_arrays.append(ch_arr)
                except (EOFError, ValueError) as e:
                    logger.warning("Skipping corrupted channel %s: %s", ch_path, e)
                    break

            if len(channel_arrays) != N_CHANNELS:
                continue

            ref_shape = channel_arrays[0].shape
            num_windows = ref_shape[0]

            all_match = True
            for arr in channel_arrays:
                if arr.shape[0] != ref_shape[0]:
                    logger.warning("Service %d: window count mismatch. Skipping.", svc_idx)
                    all_match = False
                    break
            if not all_match:
                continue

            idx_tr = int(num_windows * train_frac)
            idx_val = int(num_windows * (train_frac + val_frac))

            if idx_tr == 0 or idx_tr >= idx_val or idx_val >= num_windows:
                continue

            if split == "train":
                w_start, w_end = 0, idx_tr
            elif split == "val":
                w_start, w_end = idx_tr, idx_val
            else:
                w_start, w_end = idx_val, num_windows

            if w_start >= w_end:
                continue

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

                all_X.append(X)
                all_y.append(y)
                all_last.append(last_val)

        if not all_X:
            logger.warning(
                "CshvtbDataset[%s]: no valid windows found in %s",
                split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, N_CHANNELS), dtype=torch.float32)
            self.y = torch.empty((0, pred_horizon), dtype=torch.float32)
            self.last = torch.empty((0,), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.stack(all_X, axis=0))
            self.y = torch.from_numpy(np.stack(all_y, axis=0))
            self.last = torch.from_numpy(np.stack(all_last, axis=0))

        logger.info(
            "CshvtbDataset[%s]: %d windows from %d service files",
            split, len(self.X), len(service_files),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]


def _smoke_check(preprocess_dir: str, split: str) -> None:
    from experiments.cshvtb.config import CFG

    ds = CshvtbDataset(
        preprocess_dir,
        split,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC,
        val_frac=CFG.VAL_FRAC,
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
    print("CshvtbDataset smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check CshvtbDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split)

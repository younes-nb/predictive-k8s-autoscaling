
import argparse
import glob
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def build_windows(
    signal: np.ndarray, input_len: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:

    T = len(signal) - input_len
    if T <= 0:
        return (
            np.empty((0, input_len), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )
    indices = np.arange(0, T, stride)
    X = np.stack([signal[i: i + input_len] for i in indices]).astype(np.float32)
    y = signal[indices + input_len].reshape(-1, 1).astype(np.float32)
    return X, y

class CoImfDataset(Dataset):

    def __init__(
        self,
        preprocess_dir: str,
        co_imf_index: int,
        split: str,
        input_len: int = 30,
        stride: int = 1,
        test_size: int = 500,
        val_frac: float = 0.10,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split

        co_imf_dir = os.path.join(preprocess_dir, f"co_imf_{co_imf_index}")
        service_files = sorted(glob.glob(os.path.join(co_imf_dir, "service_*.npy")))

        if not service_files:
            raise FileNotFoundError(
                f"No Co-IMF files found in {co_imf_dir}. "
                "Run preprocess_services.py first."
            )

        all_X, all_y = [], []

        for sf in service_files:
            try:
                signal = np.load(sf).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted file %s: %s", sf, e)
                continue
            N = len(signal)

            if N < input_len + test_size + 1:
                continue

            test_start = N - test_size
            train_end = test_start - max(1, int((N - test_size) * val_frac))

            if split == "train":
                seg = signal[:train_end]
            elif split == "val":
                seg = signal[train_end:test_start]
            else:
                seg = signal[test_start:]

            if len(seg) < input_len + 1:
                continue

            X, y = build_windows(seg.astype(np.float32), input_len, stride)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            logger.warning(
                "CoImfDataset[co_imf_%d/%s]: no valid windows found in %s",
                co_imf_index, split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, 1), dtype=torch.float32)
            self.y = torch.empty((0, 1), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.concatenate(all_X, axis=0)).unsqueeze(-1)
            self.y = torch.from_numpy(np.concatenate(all_y, axis=0))

        logger.info(
            "CoImfDataset[co_imf_%d/%s]: %d windows from %d service files",
            co_imf_index, split, len(self.X), len(service_files),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.X[idx, -1, 0]

def _smoke_check(preprocess_dir: str, co_imf_index: int, split: str) -> None:
    from experiments.cvcbm.config import CFG

    ds = CoImfDataset(
        preprocess_dir,
        co_imf_index,
        split,
        input_len=CFG.INPUT_LEN,
        stride=CFG.STRIDE,
        test_size=CFG.TEST_SIZE,
        val_frac=CFG.VAL_FRAC,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    assert tuple(x.shape) == (CFG.INPUT_LEN, 1), f"Bad x shape: {tuple(x.shape)}"
    assert tuple(y.shape) == (1,), f"Bad y shape: {tuple(y.shape)}"
    assert last.dim() == 0, f"Bad last shape: {tuple(last.shape)}"
    print(f"Dataset windows: {len(ds)}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last_dim={last.dim()}")
    print("CoImfDataset smoke test passed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check CoImfDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--co_imf_index", type=int, default=0)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.co_imf_index, args.split)

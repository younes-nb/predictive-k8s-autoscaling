import argparse
import bisect
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)

MAX_IMFS = 3


class CskvDataset(Dataset):
    """Multi-channel dataset: stacks all Co-IMFs + VMD modes into one input matrix.

    Input shape:  (num_samples, input_len, total_channels)
    Target shape: (num_samples, pred_horizon)   <-- raw workload signal
    Last shape:   (num_samples,)                <-- last observed raw value

    Channel layout:
      [VMD mode 0, ..., VMD mode K-1, Co-IMF 1 (Medium), Co-IMF 2 (Low)]
    """

    def __init__(
        self,
        preprocess_dir: str,
        split: str,
        input_len: int = 128,
        pred_horizon: int = 1,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        t_start = time.time()

        x_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_X_{split}.npy")))
        y_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_y_{split}.npy")))
        sid_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_sid_{split}.npy")))
        last_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_last_{split}.npy")))

        if not x_files:
            raise FileNotFoundError(
                f"No decomposed shards found in {preprocess_dir} for split={split}. "
                "Run cskv/preprocess.py first."
            )

        n_shards = len(x_files)
        self._x_shards = []
        self._y_shards = []
        self._last_shards = []
        self._offsets = [0]
        n_windows = 0
        for xf in x_files:
            base = os.path.basename(xf).replace(f"_X_{split}.npy", "")
            yf = os.path.join(preprocess_dir, f"{base}_y_{split}.npy")
            sf = os.path.join(preprocess_dir, f"{base}_sid_{split}.npy")
            lf = os.path.join(preprocess_dir, f"{base}_last_{split}.npy")

            if not os.path.exists(yf) or not os.path.exists(sf):
                logger.warning("Missing y/sid for shard %s, skipping", base)
                continue

            X = np.load(xf, mmap_mode="r")
            y = np.load(yf, mmap_mode="r")

            if len(X) != len(y):
                logger.warning("X/y length mismatch in shard %s, skipping", base)
                continue

            last = np.load(lf, mmap_mode="r") if os.path.exists(lf) else None
            if last is not None and len(last) != len(X):
                logger.warning("last length mismatch in shard %s, skipping", base)
                last = None

            self._x_shards.append(X)
            self._y_shards.append(y)
            self._last_shards.append(last)
            n_windows += len(X)
            self._offsets.append(n_windows)

        self.n_windows = n_windows
        if self._x_shards:
            self.total_channels = self._x_shards[0].shape[-1]
        else:
            logger.warning("CskvDataset[%s]: no valid windows found in %s", split, preprocess_dir)
            self.total_channels = 1

        logger.info(
            "CskvDataset[%s]: %d windows, channels=%d from %d shards in %.1fs",
            split, n_windows, self.total_channels, n_shards, time.time() - t_start,
        )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        i = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[i]
        x = np.array(self._x_shards[i][local], copy=True, order="C")
        y = np.array(self._y_shards[i][local], copy=True, order="C")
        last = self._last_shards[i]
        if last is not None:
            last = np.array(last[local], copy=True, order="C")
        else:
            last = np.zeros((), dtype=np.float32)
        return x, y, last


def _smoke_check(preprocess_dir: str, split: str) -> None:
    from shared.config_preprocessing_defaults import PREPROCESSING

    ds = CskvDataset(
        preprocess_dir,
        split,
        input_len=PREPROCESSING.INPUT_LEN,
        pred_horizon=PREPROCESSING.PRED_HORIZON,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    expected_x_shape = (PREPROCESSING.INPUT_LEN, ds.total_channels)
    assert tuple(x.shape) == expected_x_shape, \
        f"Bad x shape: {tuple(x.shape)} expected {expected_x_shape}"
    expected_y_shape = (PREPROCESSING.PRED_HORIZON,)
    assert tuple(y.shape) == expected_y_shape, \
        f"Bad y shape: {tuple(y.shape)} expected {expected_y_shape}"
    assert last.dim() == 0, f"Bad last shape: {tuple(last.shape)}"
    print(f"Dataset windows: {len(ds)}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last={tuple(last.shape)} channels={ds.total_channels}")
    print("CskvDataset smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check CskvDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split)

import argparse
import bisect
import glob
import json
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

from shared.features import get_feature_set

logger = logging.getLogger(__name__)


def _load_meta(preprocess_dir: str) -> dict:
    meta_path = os.path.join(preprocess_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found at {meta_path}. "
            "Run swt/preprocess.py first to generate it."
        )
    with open(meta_path) as f:
        return json.load(f)


class SwtDataset(Dataset):
    def __init__(
        self,
        preprocess_dir: str,
        split: str,
        input_len: int = 128,
        pred_horizon: int = 5,
        feature_set: str = "cpu",
        swt_level: int = 5,
        mem_swt_level: int = 5,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        # Load metadata from preprocessing
        meta = _load_meta(preprocess_dir)
        self.features = meta["features"]
        self.target_features = meta["target_features"]
        self.target_indices = meta["target_indices"]
        self.channel_counts = meta["channel_counts"]
        self.n_channels = meta["total_channels"]
        self.has_mem = "memory_utilization" in self.target_features

        t_start = time.time()

        x_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_X_{split}.npy")))
        y_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_y_{split}.npy")))
        sid_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_sid_{split}.npy")))
        last_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_last_{split}.npy")))

        if not x_files:
            raise FileNotFoundError(
                f"No decomposed shards found in {preprocess_dir} for split={split}. "
                "Run swt/preprocess.py first."
            )

        self._x_shards = []
        self._y_shards = []
        self._last_shards = []
        self._offsets = [0]
        n_windows = 0
        n_shards = len(x_files)
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

        if not self._x_shards:
            logger.warning("SwtDataset[%s]: no valid windows found in %s", split, preprocess_dir)

        logger.info(
            "SwtDataset[%s]: %d windows, %d channels from %d shards in %.1fs",
            split, n_windows, self.n_channels, n_shards, time.time() - t_start,
        )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        i = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[i]
        x = np.array(self._x_shards[i][local], copy=True, order="C").astype(np.float16)
        y = np.array(self._y_shards[i][local], copy=True, order="C").astype(np.float16)
        last = self._last_shards[i]
        if last is not None:
            last = np.array(last[local], copy=True, order="C").astype(np.float16)
        else:
            last = np.zeros(2, dtype=np.float16) if self.has_mem else np.zeros((), dtype=np.float16)
        return x, y, last


def _smoke_check(preprocess_dir: str, split: str,
                  feature_set: str = "cpu", swt_level: int = 5,
                  mem_swt_level: int = 5) -> None:
    from shared.config_preprocessing_defaults import PREPROCESSING

    ds = SwtDataset(
        preprocess_dir,
        split,
        input_len=PREPROCESSING.INPUT_LEN,
        pred_horizon=PREPROCESSING.PRED_HORIZON,
        feature_set=feature_set,
        swt_level=swt_level,
        mem_swt_level=mem_swt_level,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    expected_x_shape = (PREPROCESSING.INPUT_LEN, ds.n_channels)
    assert tuple(x.shape) == expected_x_shape, \
        f"Bad x shape: {tuple(x.shape)} expected {expected_x_shape}"
    expected_y_shape = (PREPROCESSING.PRED_HORIZON, len(ds.target_indices))
    expected_last_shape = (len(ds.target_indices),)
    assert tuple(y.shape) == expected_y_shape, \
        f"Bad y shape: {tuple(y.shape)} expected {expected_y_shape}"
    assert tuple(last.shape) == expected_last_shape, \
        f"Bad last shape: {tuple(last.shape)} expected {expected_last_shape}"
    print(f"Dataset windows: {len(ds)}")
    print(f"Features: {ds.features}")
    print(f"Target features: {ds.target_features}")
    print(f"Target indices: {ds.target_indices}")
    print(f"Channel counts: {ds.channel_counts}")
    print(f"Total channels: {ds.n_channels}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last={tuple(last.shape)}")
    print("SwtDataset smoke test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check SwtDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument("--feature_set", default="cpu")
    ap.add_argument("--swt_level", type=int, default=5)
    ap.add_argument("--mem_swt_level", type=int, default=5)
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split,
                 feature_set=args.feature_set, swt_level=args.swt_level,
                 mem_swt_level=args.mem_swt_level)

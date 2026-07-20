import argparse
import glob
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.sv.config import CFG as SV_CFG, channel_dirs_for

logger = logging.getLogger(__name__)


class SvDataset(Dataset):
    def __init__(
        self,
        preprocess_dir: str,
        split: str,
        input_len: int = 60,
        pred_horizon: int = 5,
        feature_set: str = "cpu",
        swt_level: int = SV_CFG.SWT_LEVEL,
        mem_swt_level: int = SV_CFG.MEM_SWT_LEVEL,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon
        self.has_mem = feature_set == "cpu_mem_both"

        cpu_channel_dirs = channel_dirs_for(swt_level, SV_CFG.VMD_K, prefix="")
        mem_channel_dirs = channel_dirs_for(mem_swt_level, SV_CFG.VMD_K, prefix="mem_")
        self.channel_dirs = cpu_channel_dirs + (mem_channel_dirs if self.has_mem else [])
        self.n_channels = len(self.channel_dirs)

        t_start = time.time()

        x_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_X_{split}.npy")))
        y_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_y_{split}.npy")))
        sid_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_sid_{split}.npy")))
        last_files = sorted(glob.glob(os.path.join(preprocess_dir, f"part-*_last_{split}.npy")))

        if not x_files:
            raise FileNotFoundError(
                f"No decomposed shards found in {preprocess_dir} for split={split}. "
                "Run sv/preprocess.py first."
            )

        n_shards = len(x_files)
        all_X = []
        all_y = []
        all_last = []
        for i, xf in enumerate(x_files):
            base = os.path.basename(xf).replace(f"_X_{split}.npy", "")
            yf = os.path.join(preprocess_dir, f"{base}_y_{split}.npy")
            sf = os.path.join(preprocess_dir, f"{base}_sid_{split}.npy")
            lf = os.path.join(preprocess_dir, f"{base}_last_{split}.npy")

            if not os.path.exists(yf) or not os.path.exists(sf):
                logger.warning("Missing y/sid for shard %s, skipping", base)
                continue

            X = np.load(xf)
            y = np.load(yf)
            last = np.load(lf) if os.path.exists(lf) else None

            if len(X) != len(y):
                logger.warning("X/y length mismatch in shard %s, skipping", base)
                continue

            if last is not None and len(last) != len(X):
                logger.warning("last length mismatch in shard %s, skipping", base)
                last = None

            all_X.append(X)
            all_y.append(y)
            all_last.append(last)

        if not all_X:
            logger.warning("SvDataset[%s]: no valid windows found in %s", split, preprocess_dir)
            self.X = torch.empty((0, self.n_channels), dtype=torch.float32)
            if self.has_mem:
                self.y = torch.empty((0, 2), dtype=torch.float32)
                self.last = torch.empty((0, 2), dtype=torch.float32)
            else:
                self.y = torch.empty((0,), dtype=torch.float32)
                self.last = torch.empty((0,), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.concatenate(all_X, axis=0))
            self.y = torch.from_numpy(np.concatenate(all_y, axis=0))
            if all_last[0] is not None:
                self.last = torch.from_numpy(np.concatenate(all_last, axis=0))
            else:
                if self.has_mem:
                    self.last = torch.zeros(len(self.X), 2, dtype=torch.float32)
                else:
                    self.last = torch.zeros(len(self.X), dtype=torch.float32)

        logger.info(
            "SvDataset[%s]: %d windows, X=%s, y=%s from %d shards in %.1fs",
            split, len(self.X), tuple(self.X.shape), tuple(self.y.shape),
            n_shards, time.time() - t_start,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]


def _smoke_check(preprocess_dir: str, split: str,
                  feature_set: str = "cpu", swt_level: int = SV_CFG.SWT_LEVEL,
                  mem_swt_level: int = SV_CFG.MEM_SWT_LEVEL) -> None:
    from shared.config_preprocessing_defaults import PREPROCESSING

    ds = SvDataset(
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
    ap.add_argument("--feature_set", default="cpu")
    ap.add_argument("--swt_level", type=int, default=SV_CFG.SWT_LEVEL)
    ap.add_argument("--mem_swt_level", type=int, default=SV_CFG.MEM_SWT_LEVEL)
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split,
                 feature_set=args.feature_set, swt_level=args.swt_level,
                 mem_swt_level=args.mem_swt_level)

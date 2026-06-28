
import argparse
import glob
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CoImfDataset(Dataset):

    def __init__(
        self,
        preprocess_dir: str,
        co_imf_index: int,
        split: str,
        input_len: int = 30,
        pred_horizon: int = 1,
        stride: int = 1,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split

        co_imf_dir = os.path.join(preprocess_dir, f"co_imf_{co_imf_index}")
        original_dir = os.path.join(preprocess_dir, "original")
        service_files = sorted(glob.glob(os.path.join(co_imf_dir, "service_*.npy")))

        if not service_files:
            raise FileNotFoundError(
                f"No Co-IMF files found in {co_imf_dir}. "
                "Run preprocess_services.py first."
            )

        all_X, all_y, all_last = [], [], []

        for sf in service_files:
            try:
                co_imf_windows = np.load(sf).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted file %s: %s", sf, e)
                continue

            num_windows = co_imf_windows.shape[0]
            idx_tr = int(num_windows * train_frac)
            idx_val = int(num_windows * (train_frac + val_frac))

            if idx_tr == 0 or idx_tr >= idx_val or idx_val >= num_windows:
                continue

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

            if split == "train":
                w_start, w_end = 0, idx_tr
            elif split == "val":
                w_start, w_end = idx_tr, idx_val
            else:
                w_start, w_end = idx_val, num_windows

            if w_start >= w_end:
                continue

            # Each Co-IMF window corresponds to a sliding window of the original signal.
            # Target y = next PRED_HORIZON values of the original signal after the window.
            # last_val = last observed value in the window (used for persistence baseline).
            for j in range(w_start, w_end):
                pos = j * stride
                if pos + input_len + pred_horizon > len(original):
                    continue
                X = co_imf_windows[j]
                y = original[pos + input_len : pos + input_len + pred_horizon]
                last_val = original[pos + input_len - 1]
                all_X.append(X.astype(np.float32))
                all_y.append(y.astype(np.float32))
                all_last.append(last_val.astype(np.float32))

        if not all_X:
            logger.warning(
                "CoImfDataset[co_imf_%d/%s]: no valid windows found in %s",
                co_imf_index, split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, 1), dtype=torch.float32)
            self.y = torch.empty((0, pred_horizon), dtype=torch.float32)
            self.last = torch.empty((0,), dtype=torch.float32)
        else:
            self.X = torch.from_numpy(np.stack(all_X, axis=0)).unsqueeze(-1)
            self.y = torch.from_numpy(np.stack(all_y, axis=0))
            self.last = torch.from_numpy(np.stack(all_last, axis=0))

        logger.info(
            "CoImfDataset[co_imf_%d/%s]: %d windows from %d service files",
            co_imf_index, split, len(self.X), len(service_files),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]

def _smoke_check(preprocess_dir: str, co_imf_index: int, split: str) -> None:
    from experiments.cvcbm.config import CFG

    ds = CoImfDataset(
        preprocess_dir,
        co_imf_index,
        split,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC,
        val_frac=CFG.VAL_FRAC,
    )
    assert len(ds) > 0, "Dataset has no windows"
    x, y, last = ds[0]
    assert tuple(x.shape) == (CFG.INPUT_LEN, 1), f"Bad x shape: {tuple(x.shape)}"
    assert tuple(y.shape) == (CFG.PRED_HORIZON,), f"Bad y shape: {tuple(y.shape)}"
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

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ShardedWindowsDataset(Dataset):
    def __init__(
        self,
        windows_dir: str,
        split: str,
        input_len: int,
        horizon: int,
        use_weights: bool = False,
    ):
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.split = split
        self.use_weights = use_weights

        self.shard_paths = []
        self.cum_lengths = []

        pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(pattern), key=self._natural_key)

        if not x_files:
            print(f"[WARN] No shards found for split={split} in {windows_dir}")
            self.total_len = 0
            return

        total = 0
        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"
            sid_path = base + f"_sid_{split}.npy"
            w_path = base + f"_w_{split}.npy"

            if not (os.path.exists(y_path) and os.path.exists(sid_path)):
                continue

            x_mmap = np.load(x_path, mmap_mode="r")
            shard_size = x_mmap.shape[0]
            del x_mmap

            shard_info = {
                "X": x_path,
                "y": y_path,
                "sid": sid_path,
                "w": (
                    w_path
                    if (self.use_weights and split != "test" and os.path.exists(w_path))
                    else None
                ),
            }

            if self.use_weights and split != "test" and shard_info["w"] is None:
                raise RuntimeError(f"Weights requested but missing: {w_path}")

            self.shard_paths.append(shard_info)
            total += shard_size
            self.cum_lengths.append(total)

        self.cum_lengths = np.array(self.cum_lengths)
        self.total_len = total
        print(
            f"[{split}] Metadata loaded for {len(self.shard_paths)} shards, total: {self.total_len}"
        )

    def _natural_key(self, p):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", Path(p).name)
        ]

    def __len__(self):
        return getattr(self, "total_len", 0)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        shard_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        prev_cum = 0 if shard_idx == 0 else self.cum_lengths[shard_idx - 1]
        local_idx = idx - prev_cum

        paths = self.shard_paths[shard_idx]

        X_mmap = np.load(paths["X"], mmap_mode="r")
        Y_mmap = np.load(paths["y"], mmap_mode="r")
        SIDs_mmap = np.load(paths["sid"], mmap_mode="r")

        x_arr = np.array(X_mmap[local_idx], copy=True)
        y_arr = np.array(Y_mmap[local_idx], copy=True)
        sid_val = int(SIDs_mmap[local_idx])

        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]

        x_t = torch.from_numpy(x_arr).float()
        y_t = torch.from_numpy(y_arr).float()
        sid_t = torch.tensor(sid_val, dtype=torch.long)

        if paths["w"] is not None:
            W_mmap = np.load(paths["w"], mmap_mode="r")
            w_val = float(W_mmap[local_idx])
            w_t = torch.tensor(w_val, dtype=torch.float32)
            return x_t, y_t, w_t, sid_t

        return x_t, y_t, sid_t

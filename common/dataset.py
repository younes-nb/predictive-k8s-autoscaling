import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


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

        self.shards = []
        self.cum_lengths = []

        pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(pattern))

        if not x_files:
            print(f"[WARN] No shards found for split={split} in {windows_dir}")
            self.total_len = 0
            return

        total = 0
        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"

            X = np.load(x_path, mmap_mode="r")
            Y = np.load(y_path, mmap_mode="r")

            W = None
            if self.use_weights and split == "train":
                w_path = base + f"_w_{split}.npy"
                if os.path.exists(w_path):
                    W = np.load(w_path, mmap_mode="r")
                else:
                    raise RuntimeError(f"Weights requested but missing: {w_path}")

            self.shards.append((X, Y, W))
            total += X.shape[0]
            self.cum_lengths.append(total)

        self.total_len = total
        print(f"[{split}] Loaded {len(self.shards)} shards, total: {self.total_len}")

    def __len__(self):
        return getattr(self, "total_len", 0)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        for shard_idx, cum in enumerate(self.cum_lengths):
            if idx < cum:
                prev_cum = 0 if shard_idx == 0 else self.cum_lengths[shard_idx - 1]
                local_idx = idx - prev_cum

                X, Y, W = self.shards[shard_idx]

                x_arr = np.array(X[local_idx], copy=True)
                y_arr = np.array(Y[local_idx], copy=True)

                if x_arr.ndim == 1:
                    x_arr = x_arr[:, None]

                x_t = torch.from_numpy(x_arr).float()
                y_t = torch.from_numpy(y_arr).float()

                if W is not None:
                    w_val = float(W[local_idx])
                    return x_t, y_t, torch.tensor(w_val, dtype=torch.float32)

                return x_t, y_t

        raise IndexError(idx)

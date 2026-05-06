import os
import glob
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config.defaults import PATHS  # Added import


class ShardedWindowsDataset(Dataset):
    def __init__(
        self,
        windows_dir: str,
        split: str,
        input_len: int,
        horizon: int,
        use_weights: bool = False,
        archetype_id: int = None,
    ):
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.split = split
        self.use_weights = use_weights
        self.archetype_id = archetype_id

        self.shard_paths = []
        self.valid_indices = []

        self.archetype_map = None
        if self.archetype_id is not None:
            if not os.path.exists(PATHS.ARCHETYPE_MAPPING):
                raise FileNotFoundError(
                    f"Archetype mapping not found at {PATHS.ARCHETYPE_MAPPING}. Run clustering first."
                )
            with open(PATHS.ARCHETYPE_MAPPING, "r") as f:
                self.archetype_map = json.load(f)

        pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(pattern), key=self._natural_key)

        if not x_files:
            print(f"[WARN] No shards found for split={split} in {windows_dir}")
            return

        print(
            f"[{split}] Filtering shards for Archetype {archetype_id}..."
            if archetype_id is not None
            else f"[{split}] Loading all shards..."
        )

        for s_idx, x_path in enumerate(x_files):
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"
            sid_path = base + f"_sid_{split}.npy"
            w_path = base + f"_w_{split}.npy"

            if not (os.path.exists(y_path) and os.path.exists(sid_path)):
                continue

            sids = np.load(sid_path)

            if self.archetype_id is not None:
                mask = np.array(
                    [
                        self.archetype_map.get(str(s), -1) == self.archetype_id
                        for s in sids
                    ]
                )
                local_valid_rows = np.where(mask)[0]
            else:
                local_valid_rows = np.arange(len(sids))

            if len(local_valid_rows) > 0:
                shard_info = {
                    "X": x_path,
                    "y": y_path,
                    "sid": sid_path,
                    "w": (
                        w_path
                        if (
                            self.use_weights
                            and split != "test"
                            and os.path.exists(w_path)
                        )
                        else None
                    ),
                }

                if self.use_weights and split != "test" and shard_info["w"] is None:
                    raise RuntimeError(f"Weights requested but missing: {w_path}")

                shard_list_idx = len(self.shard_paths)
                self.shard_paths.append(shard_info)

                for l_idx in local_valid_rows:
                    self.valid_indices.append((shard_list_idx, l_idx))

        self.total_len = len(self.valid_indices)
        print(
            f"[{split}] Filtered to {self.total_len} samples across {len(self.shard_paths)} shards."
        )

    def _natural_key(self, p):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", Path(p).name)
        ]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        shard_list_idx, local_idx = self.valid_indices[idx]
        paths = self.shard_paths[shard_list_idx]

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

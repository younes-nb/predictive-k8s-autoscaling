import os
import glob
import re
from pathlib import Path

import numpy as np


def natural_key(p: str):
    return [
        int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", Path(p).name)
    ]


def find_shards(windows_dir: str, split: str) -> list:
    pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
    x_files = sorted(glob.glob(pattern), key=natural_key)
    shards = []
    for x_path in x_files:
        base = x_path.replace(f"_X_{split}.npy", "")
        y_path = base + f"_y_{split}.npy"
        sid_path = base + f"_sid_{split}.npy"
        if os.path.exists(y_path) and os.path.exists(sid_path):
            shards.append((x_path, y_path, sid_path, base))
    return shards


def hist_update(hist, values, bins: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    values = np.clip(values, 0.0, 1.0)
    indices = (values * (bins - 1)).astype(np.int32)
    np.add.at(hist, indices, 1)


def hist_quantile(hist, tau: float) -> float:
    total = hist.sum()
    if total <= 0:
        return 0.5
    cdf = np.cumsum(hist)
    target = tau * total
    bin_idx = np.searchsorted(cdf, target, side="left")
    return float(bin_idx / (len(hist) - 1))

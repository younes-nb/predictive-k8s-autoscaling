
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

class CvcbmDataset(Dataset):
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
        input_len: int = 30,
        pred_horizon: int = 1,
        stride: int = 1,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        co_imf_dirs = [
            os.path.join(preprocess_dir, f"co_imf_{k}") for k in range(3)
        ]
        original_dir = os.path.join(preprocess_dir, "original")

        service_files = sorted(glob.glob(os.path.join(co_imf_dirs[0], "service_*.npy")))

        if not service_files:
            raise FileNotFoundError(
                f"No Co-IMF files found in {co_imf_dirs[0]}. "
                "Run preprocess_services.py first."
            )

        all_X, all_y, all_last = [], [], []

        for sf in service_files:
            try:
                co_imf0_windows = np.load(sf).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted file %s: %s", sf, e)
                continue

            num_windows = co_imf0_windows.shape[0]
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

            # Load Co-IMF 1 (Medium)
            co_imf1_path = os.path.join(co_imf_dirs[1], base)
            if not os.path.exists(co_imf1_path):
                logger.warning("Co-IMF 1 not found for service %d: %s", svc_idx, co_imf1_path)
                continue
            try:
                co_imf1_windows = np.load(co_imf1_path).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted Co-IMF 1 %s: %s", co_imf1_path, e)
                continue

            # Load Co-IMF 2 (Low)
            co_imf2_path = os.path.join(co_imf_dirs[2], base)
            if not os.path.exists(co_imf2_path):
                logger.warning("Co-IMF 2 not found for service %d: %s", svc_idx, co_imf2_path)
                continue
            try:
                co_imf2_windows = np.load(co_imf2_path).astype(np.float64)
            except (EOFError, ValueError) as e:
                logger.warning("Skipping corrupted Co-IMF 2 %s: %s", co_imf2_path, e)
                continue

            # Load VMD modes from co_imf_0 directory
            vmd_windows_list = []
            vmd_mode_file = os.path.join(co_imf_dirs[0], f"vmd_mode_0_{base}")
            if os.path.exists(vmd_mode_file):
                mode_idx = 0
                while True:
                    vmd_path = os.path.join(co_imf_dirs[0], f"vmd_mode_{mode_idx}_{base}")
                    if not os.path.exists(vmd_path):
                        break
                    try:
                        vmd_windows_list.append(np.load(vmd_path).astype(np.float64))
                    except (EOFError, ValueError) as e:
                        logger.warning("Skipping corrupted VMD mode %s: %s", vmd_path, e)
                    mode_idx += 1
            else:
                # Fall back to summed co_imf_0 as single channel
                vmd_windows_list = [co_imf0_windows]

            # Validate window count consistency
            ref_shape = co_imf0_windows.shape
            for arr_name, arr in [
                ("co_imf1", co_imf1_windows),
                ("co_imf2", co_imf2_windows),
            ]:
                if arr.shape[0] != ref_shape[0]:
                    logger.warning(
                        "Service %d: %s has %d windows, expected %d. Skipping.",
                        svc_idx, arr_name, arr.shape[0], ref_shape[0],
                    )
                    break
            else:
                for vm in vmd_windows_list:
                    if vm.shape[0] != ref_shape[0]:
                        logger.warning(
                            "Service %d: VMD window count mismatch. Skipping.", svc_idx,
                        )
                        break
                else:
                    if split == "train":
                        w_start, w_end = 0, idx_tr
                    elif split == "val":
                        w_start, w_end = idx_tr, idx_val
                    else:
                        w_start, w_end = idx_val, num_windows

                    if w_start >= w_end:
                        continue

                    channel_arrays = vmd_windows_list + [co_imf1_windows, co_imf2_windows]
                    total_channels_svc = len(channel_arrays)

                    for j in range(w_start, w_end):
                        pos = j * stride
                        if pos + input_len + pred_horizon > len(original):
                            continue

                        # Build multi-channel input: (input_len, total_channels)
                        X = np.stack(
                            [ch[j] for ch in channel_arrays],
                            axis=1,
                        ).astype(np.float32)

                        y = original[pos + input_len : pos + input_len + pred_horizon].astype(np.float32)
                        last_val = original[pos + input_len - 1].astype(np.float32)

                        all_X.append(X)
                        all_y.append(y)
                        all_last.append(last_val)

        if not all_X:
            logger.warning(
                "CvcbmDataset[%s]: no valid windows found in %s",
                split, preprocess_dir,
            )
            self.X = torch.empty((0, input_len, 1), dtype=torch.float32)
            self.y = torch.empty((0, pred_horizon), dtype=torch.float32)
            self.last = torch.empty((0,), dtype=torch.float32)
            self.total_channels = 1
        else:
            self.X = torch.from_numpy(np.stack(all_X, axis=0))
            self.y = torch.from_numpy(np.stack(all_y, axis=0))
            self.last = torch.from_numpy(np.stack(all_last, axis=0))
            self.total_channels = total_channels_svc

        logger.info(
            "CvcbmDataset[%s]: %d windows, %d channels from %d service files",
            split, len(self.X), self.total_channels, len(service_files),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.last[idx]


def _smoke_check(preprocess_dir: str, split: str) -> None:
    from experiments.cvcbm.config import CFG

    ds = CvcbmDataset(
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
    expected_x_shape = (CFG.INPUT_LEN, ds.total_channels)
    assert tuple(x.shape) == expected_x_shape, f"Bad x shape: {tuple(x.shape)} expected {expected_x_shape}"
    assert tuple(y.shape) == (CFG.PRED_HORIZON,), f"Bad y shape: {tuple(y.shape)}"
    assert last.dim() == 0, f"Bad last shape: {tuple(last.shape)}"
    print(f"Dataset windows: {len(ds)}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} last_dim={last.dim()} channels={ds.total_channels}")
    print("CvcbmDataset smoke test passed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-check CvcbmDataset shapes.")
    ap.add_argument("--preprocess_dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = ap.parse_args()
    _smoke_check(args.preprocess_dir, args.split)

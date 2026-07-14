import os
import sys
import argparse
import glob
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.utils import moving_average


def smooth_array(arr, window_size):
    smoothed = np.empty_like(arr)
    for i in range(arr.shape[-1]):
        result = moving_average(arr[:, i], window_size)
        if result is None:
            smoothed[:, i] = arr[:, i]
        else:
            smoothed[:, i] = result
    return smoothed


def main():
    p = argparse.ArgumentParser(
        description="Post-processing: apply moving-average smoothing to raw window .npy files."
    )
    p.add_argument("--windows_dir", required=True, help="Directory with raw window .npy files.")
    p.add_argument("--out_dir", required=True, help="Output directory for smoothed windows.")
    p.add_argument("--smoothing_window", type=int, default=5, help="Moving-average window size.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        x_pattern = os.path.join(args.windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(x_pattern))

        if not x_files:
            print(f"No X files for split={split} in {args.windows_dir}, skipping.")
            continue

        print(f"\nSmoothing split={split}: {len(x_files)} shard files")

        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"
            sid_path = base + f"_sid_{split}.npy"

            if not (os.path.exists(y_path) and os.path.exists(sid_path)):
                print(f"  Skipping {os.path.basename(x_path)}: missing y or sid file.")
                continue

            X = np.load(x_path)
            y = np.load(y_path)
            sid = np.load(sid_path)

            if X.ndim == 2:
                X = X[:, :, np.newaxis]

            n_samples, input_len, n_features = X.shape

            X_smooth = np.empty_like(X)
            for s in range(n_samples):
                X_smooth[s] = smooth_array(X[s], args.smoothing_window)

            if X_smooth.shape[-1] == 1:
                X_smooth = X_smooth[:, :, 0]

            out_base = os.path.join(args.out_dir, os.path.basename(base))
            np.save(f"{out_base}_X_{split}.npy", X_smooth)
            np.save(f"{out_base}_y_{split}.npy", y)
            np.save(f"{out_base}_sid_{split}.npy", sid)

            print(f"  {os.path.basename(x_path)}: {n_samples} samples smoothed")

    print(f"\nSmoothed windows saved to {args.out_dir}")


if __name__ == "__main__":
    main()

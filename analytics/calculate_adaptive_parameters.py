import os
import glob
import numpy as np
from tqdm import tqdm

DATA_DIR = "./data/windows"
THETA_MIN = 0.60
THETA_MAX = 0.90


def calculate_adaptive_parameters():
    print(f"📊 Analyzing Action Zone ({THETA_MIN} - {THETA_MAX})...")

    y_files = sorted(glob.glob(os.path.join(DATA_DIR, "part-*_y_train.npy")))

    all_action_zone_values = []
    total_samples = 0
    action_zone_count = 0

    for y_path in tqdm(y_files, desc="Processing Shards"):
        y = np.load(y_path, mmap_mode="r")
        y_target = y[:, -1]
        total_samples += len(y_target)

        mask = (y_target >= THETA_MIN) & (y_target <= THETA_MAX)
        action_samples = y_target[mask]

        action_zone_count += len(action_samples)
        all_action_zone_values.extend(action_samples.tolist())

    if not all_action_zone_values:
        print("❌ Action Zone is empty. Check your data scaling.")
        return

    delta = np.std(all_action_zone_values)

    normal_samples = total_samples - action_zone_count
    gamma = (normal_samples / action_zone_count) - 1.0

    print("=" * 45)
    print(f"Operational Range:  {THETA_MIN} <--> {THETA_MAX}")
    print(f"Action Zone Density: {(action_zone_count/total_samples)*100:.2f}% of data")
    print("-" * 45)
    print(f"✅ DELTA (Sigma):    {delta:.6f}")
    print(f"✅ GAMMA (Imbalance): {gamma:.2f}")
    print("-" * 45)


if __name__ == "__main__":
    calculate_adaptive_parameters()

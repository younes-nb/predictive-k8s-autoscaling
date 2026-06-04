"""Compute Ground-Truth Boundary Weights for training data."""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import glob
import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
import torch

# Config imports from shared submodules
from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_training_defaults import TRAINING

# Extracted utilities
from tooling.utils import find_shards, hist_update, hist_quantile


def main():
    ap = argparse.ArgumentParser(description="Compute Ground-Truth Boundary Weights.")
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument(
        "--archetype_id",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--theta_mode", default=TRAINING.THETA_MODE, choices=["adaptive", "static"]
    )
    ap.add_argument("--theta_base", type=float, default=TRAINING.THETA_BASE)
    ap.add_argument("--gamma", type=float, default=TRAINING.GAMMA)
    ap.add_argument("--delta", type=float, default=TRAINING.DELTA)
    ap.add_argument("--theta_min", type=float, default=TRAINING.THETA_MIN)
    ap.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)

    args = ap.parse_args()
    shards = find_shards(args.windows_dir, args.split)

    sid_to_cluster = {}
    if args.archetype_id is not None:
        if os.path.exists(PATHS.ARCHETYPE_MAPPING):
            print(
                f"Resolving from {PATHS.ARCHETYPE_MAPPING}..."
            )
            with open(PATHS.ARCHETYPE_MAPPING, "r") as f:
                name_to_cluster = json.load(f)

            pq_dir = DATASET_TABLES["msresource"]["parquet_dir"]
            all_parts = sorted(glob.glob(os.path.join(pq_dir, "*.parquet")))
            all_services = set()
            for p in all_parts:
                try:
                    all_services.update(
                        pl.scan_parquet(p)
                        .select("msname")
                        .unique()
                        .collect()["msname"]
                        .to_list()
                    )
                except:
                    continue

            sorted_names = sorted(list(all_services))
            sid_to_cluster = {
                idx: name_to_cluster[name]
                for idx, name in enumerate(sorted_names)
                if name in name_to_cluster
            }
        else:
            print(
                f"[ERROR] Archetype {args.archetype_id} requested but service_clusters.npy not found."
            )
            sys.exit(1)

    theta_base_dict = {}
    if args.theta_mode == "adaptive":
        hists: Dict[int, np.ndarray] = {}
        bins = 1000
        print(
            f"Pass 1: Building service histograms for Adaptive Mode ({len(shards)} shards)..."
        )
        for x_path, _, sid_path, _ in shards:
            X = np.load(x_path, mmap_mode="r")
            sid = np.load(sid_path, mmap_mode="r")

            u_now = X[:, -1, -1] if X.ndim == 3 else X[:, -1]
            for s in np.unique(sid):
                if (
                    args.archetype_id is not None
                    and sid_to_cluster.get(int(s)) != args.archetype_id
                ):
                    continue

                if s not in hists:
                    hists[s] = np.zeros(bins, dtype=np.int64)
                hist_update(hists[s], u_now[sid == s], bins)

        theta_base_dict = {
            int(s): hist_quantile(h, args.theta_base) for s, h in hists.items()
        }
    else:
        print(f"Static Mode: Using global base threshold {args.theta_base}")

    print(f"Pass 2: Computing {args.theta_mode} weights...")
    for x_path, y_path, sid_path, base in shards:
        Y = np.load(y_path, mmap_mode="r")
        sid = np.load(sid_path, mmap_mode="r")

        suffix = f"_w_{args.split}.npy"
        if args.archetype_id is not None:
            suffix = f"_arch{args.archetype_id}_w_{args.split}.npy"
        out_w_path = base + suffix

        W = np.lib.format.open_memmap(
            out_w_path, mode="w+", dtype=np.float32, shape=(Y.shape[0],)
        )

        W[:] = 1.0

        for i in range(0, Y.shape[0], args.batch_size):
            end = i + args.batch_size
            y_target = torch.from_numpy(Y[i:end, -1].copy()).float()
            sid_batch = sid[i:end]

            if args.archetype_id is not None:
                cluster_mask = torch.tensor(
                    [sid_to_cluster.get(int(s)) == args.archetype_id for s in sid_batch]
                )
            else:
                cluster_mask = torch.ones(len(sid_batch), dtype=torch.bool)

            if not cluster_mask.any():
                continue

            if args.theta_mode == "adaptive":
                tb = torch.tensor(
                    [theta_base_dict.get(int(s), args.theta_base) for s in sid_batch]
                )
                dist = torch.max(
                    torch.zeros_like(y_target),
                    torch.max(args.theta_min - y_target, y_target - tb),
                )
                w = 1.0 + args.gamma * torch.exp(-(dist**2) / (2.0 * (args.delta**2)))
            else:
                dist_sq = (y_target - args.theta_base) ** 2
                w = 1.0 + args.gamma * torch.exp(-dist_sq / (2.0 * (args.delta**2)))

            final_w = torch.ones_like(w)
            final_w[cluster_mask] = w[cluster_mask]
            W[i : i + len(final_w)] = final_w.numpy()

        W.flush()
        print(f"  Saved weights: {Path(out_w_path).name}", end="\r")

    print("\nWeight computation complete.")


if __name__ == "__main__":
    main()

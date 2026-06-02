from __future__ import annotations

import glob
import os
import shutil
import tempfile
import time
from typing import List, Optional

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors


def list_parquet_parts(parquet_dir: str):
    return sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))


def build_table_agg(
    df_or_lazy,
    time_col: str,
    id_cols: list,
    freq: str,
    feature_exprs: list,
    agg_exprs: Optional[List[pl.Expr]] = None,
):
    if isinstance(df_or_lazy, pl.DataFrame):
        df_or_lazy = df_or_lazy.lazy()

    df_or_lazy = df_or_lazy.with_columns(pl.col(time_col).cast(pl.Datetime))

    if agg_exprs is None:
        agg_exprs = [
            pl.col(raw_col).last().alias(feat_name)
            for feat_name, raw_col in feature_exprs
        ]

    out = (
        df_or_lazy.with_columns(pl.col(time_col).dt.truncate(freq).alias("_t"))
        .group_by(["_t"] + id_cols)
        .agg(agg_exprs)
        .sort(["_t"] + id_cols)
    )
    return out


def save_chunk(out_dir, shard_idx, chunk_idx, shard_data):
    base_name = f"part-{shard_idx:04d}_chunk-{chunk_idx:04d}"
    saved_any = False

    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp_dir:
            tmp_base = os.path.join(tmp_dir, base_name)

            for split, (Xs, Ys, Ss) in shard_data.items():
                if Xs:
                    np.save(f"{tmp_base}_X_{split}.npy", np.concatenate(Xs))
                    np.save(f"{tmp_base}_y_{split}.npy", np.concatenate(Ys))
                    np.save(f"{tmp_base}_sid_{split}.npy", np.concatenate(Ss))
                    saved_any = True

            if saved_any:
                files_to_move = glob.glob(f"{tmp_base}*")
                for src_file in files_to_move:
                    file_name = os.path.basename(src_file)
                    dest_file = os.path.join(out_dir, file_name)
                    shutil.move(src_file, dest_file)
                    os.sync()
                    time.sleep(2.0)

    except OSError as e:
        print(f"\nStaging Error: {e}")
        raise

    return saved_any


def _combine_split_arrays(split_data):
    Xs, Ys, Ss = split_data
    if not Xs:
        return None
    X = np.concatenate(Xs).astype(np.float32, copy=False)
    Y = np.concatenate(Ys).astype(np.float32, copy=False)
    S = np.concatenate(Ss).astype(np.int32, copy=False)
    return X, Y, S


def apply_smote_tomek(split_name, split_data, threshold, rng, k_neighbors=5):
    combined = _combine_split_arrays(split_data)
    if combined is None:
        return split_data

    X, Y, S = combined
    n = X.shape[0]
    if n == 0:
        return split_data

    y_last = Y[:, -1]
    labels = (y_last >= threshold).astype(np.int8)
    counts = np.bincount(labels, minlength=2)

    if counts.min() == 0:
        print(
            f"[SMOTE-Tomek] {split_name}: skipped — one class absent "
            f"(label-0={counts[0]}, label-1={counts[1]})"
        )
        return ([X], [Y], [S])

    majority_label = int(np.argmax(counts))
    minority_label = 1 - majority_label

    x_shape = X.shape
    X_flat = X.reshape(n, -1)

    num_to_add = int(counts[majority_label] - counts[minority_label])

    if num_to_add > 0 and counts[minority_label] > 1:
        k = min(k_neighbors, counts[minority_label] - 1)
        minority_idx = np.where(labels == minority_label)[0]

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_flat[minority_idx])
        nbrs = nn.kneighbors(return_distance=False)

        base_pos = rng.integers(0, len(minority_idx), size=num_to_add)
        nbr_col = rng.integers(0, k, size=num_to_add)
        nbr_pos = nbrs[base_pos, nbr_col]

        idx_i = minority_idx[base_pos]
        idx_j = minority_idx[nbr_pos]
        gap = rng.random(size=(num_to_add, 1)).astype(np.float32)

        synth_Xf = X_flat[idx_i] + gap * (X_flat[idx_j] - X_flat[idx_i])
        synth_Y = Y[idx_i] + gap * (Y[idx_j] - Y[idx_i])
        synth_S = S[idx_i]

        X_flat = np.vstack([X_flat, synth_Xf])
        Y = np.vstack([Y, synth_Y])
        S = np.concatenate([S, synth_S])
        labels = np.concatenate(
            [
                labels,
                np.full(num_to_add, minority_label, dtype=labels.dtype),
            ]
        )
    else:
        num_to_add = 0

    removed = 0
    n_aug = X_flat.shape[0]

    if n_aug > 1:
        nn_all = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_all.fit(X_flat)
        nn_idx = nn_all.kneighbors(return_distance=False)[:, 0]

        mutual = nn_idx[nn_idx] == np.arange(n_aug)
        is_tomek = mutual & (labels != labels[nn_idx])
        remove = is_tomek & (labels == majority_label)

        if remove.any():
            keep = ~remove
            X_flat = X_flat[keep]
            Y = Y[keep]
            S = S[keep]
            labels = labels[keep]
            removed = int(remove.sum())

    X_new = X_flat.reshape(-1, *x_shape[1:])
    n_min_new = int((labels == minority_label).sum())
    print(
        f"[SMOTE-Tomek] {split_name}: {n} → {X_new.shape[0]} samples | "
        f"minority {counts[minority_label]} → {n_min_new} "
        f"(+{num_to_add} synthetic, -{removed} Tomek majority)"
    )
    return ([X_new], [Y], [S])

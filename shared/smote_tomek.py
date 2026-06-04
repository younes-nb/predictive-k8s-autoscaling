from typing import List, Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _combine_split_arrays(
    split_data: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    Xs, Ys, Ss = split_data
    if not Xs:
        return None
    X = np.concatenate(Xs).astype(np.float32, copy=False)
    Y = np.concatenate(Ys).astype(np.float32, copy=False)
    S = np.concatenate(Ss).astype(np.int32, copy=False)
    return X, Y, S


def _apply_smote_tomek(
    split_name: str,
    split_data: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    threshold: float,
    rng: np.random.Generator,
    k_neighbors: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
        labels = np.concatenate([
            labels,
            np.full(num_to_add, minority_label, dtype=labels.dtype),
        ])
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

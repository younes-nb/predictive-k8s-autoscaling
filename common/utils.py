import numpy as np


def windowize_multivariate(
    x_feat: np.ndarray, y_target: np.ndarray, in_len: int, horizon: int, stride: int
):
    X, Y, S = [], [], []
    T = x_feat.shape[0]

    if T < in_len + horizon:
        return (
            np.empty((0, in_len, x_feat.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    for i in range(0, T - in_len - horizon + 1, stride):
        X.append(x_feat[i : i + in_len, :])
        Y.append(y_target[i + in_len : i + in_len + horizon])
        S.append(x_feat[i + in_len - 1, -1])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(S, dtype=np.int32),
    )


def moving_average(a: np.ndarray, window: int):
    n = len(a)
    if window <= 1 or n == 0:
        return a
    if window > n:
        window = n
    kernel = np.ones(window, dtype=a.dtype) / float(window)
    return np.convolve(a, kernel, mode="same")

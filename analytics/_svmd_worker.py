"""Phase 2 worker: run SVMD on D1 windows for a single service, return mode counts."""

import json
import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def svmd(
    signal: np.ndarray,
    alpha: float,
    tau: float,
    tol: float,
    max_modes: int,
    stop_power_ratio: float,
) -> np.ndarray:
    from vmdpy import VMD
    import warnings as _warnings

    signal = np.asarray(signal, dtype=np.float64)
    orig_len = len(signal)
    if orig_len < 4:
        return signal[np.newaxis, :]

    pad = orig_len % 2
    if pad:
        signal = np.append(signal, signal[-1])

    eps = np.finfo(np.float64).eps
    signal_power = float(np.sum(signal ** 2))
    residual = signal.copy()
    modes_list = []

    for k in range(max_modes):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                u, _, _ = VMD(residual, alpha, tau, K=1, DC=0, init=1, tol=tol)
            except Exception:
                break

        mode = u[0]
        if pad:
            mode = mode[:-1]

        mode_power = float(np.sum(mode ** 2))
        if k > 0 and mode_power / (signal_power + eps) < stop_power_ratio:
            break

        modes_list.append(mode)
        residual = residual - u[0]

    if not modes_list:
        return signal[:orig_len][np.newaxis, :]
    return np.stack(modes_list, axis=0)


def main() -> None:
    svc_idx = int(sys.argv[1])
    d1_dir = sys.argv[2]        # e.g. /dataset/decomp_analysis/cpu/d1_64
    input_size = int(sys.argv[3])
    alpha = float(sys.argv[4])
    tau = float(sys.argv[5])
    tol = float(sys.argv[6])
    max_modes = int(sys.argv[7])
    stop_power_ratio = float(sys.argv[8])

    _t0 = _time.time()

    d1_path = os.path.join(d1_dir, f"service_{svc_idx:05d}.npy")
    if not os.path.exists(d1_path):
        print(f"RESULT:False:ERROR: D1 file not found at {d1_path}")
        sys.exit(1)

    d1_arr = np.load(d1_path).astype(np.float64)
    mode_counts = []
    for wi in range(d1_arr.shape[0]):
        d1 = d1_arr[wi]
        if np.std(d1) < 1e-12:
            continue
        modes = svmd(d1, alpha, tau, tol, max_modes, stop_power_ratio)
        mode_counts.append(int(modes.shape[0]))

    _elapsed = _time.time() - _t0
    _log(f"[svc_{svc_idx:05d}/is{input_size}] {len(mode_counts)} windows, {_elapsed:.1f}s")

    # Output counts as JSON on stdout for the parent to collect
    result = {"idx": svc_idx, "counts": mode_counts}
    print(f"RESULT:True:{json.dumps(result)}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

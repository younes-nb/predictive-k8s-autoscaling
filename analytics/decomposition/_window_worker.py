import json
import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    svc_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    input_sizes_json = sys.argv[4]
    stride = int(sys.argv[5])

    input_sizes = json.loads(input_sizes_json)

    _t0 = _time.time()

    signal_path = os.path.join(out_dir, "original", f"service_{idx:05d}.npy")
    if not os.path.exists(signal_path):
        print(f"RESULT:False:ERROR: signal not found at {signal_path}")
        sys.exit(1)

    signal = np.load(signal_path).astype(np.float32)

    for input_size in input_sizes:
        win_dir = os.path.join(out_dir, f"windows_{input_size}")
        os.makedirs(win_dir, exist_ok=True)

        win_path = os.path.join(win_dir, f"service_{idx:05d}.npy")
        if os.path.exists(win_path):
            continue

        if len(signal) < input_size:
            continue

        T = len(signal) - input_size + 1
        windows = []
        for i in range(0, T, stride):
            w = signal[i: i + input_size].astype(np.float32)
            windows.append(w)

        if not windows:
            continue

        np.save(win_path, np.stack(windows, axis=0))

    _elapsed = _time.time() - _t0
    _log(f"[{svc_name}] {_elapsed:.0f}s")
    print(f"RESULT:True:ok")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

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
    swt_levels_json = sys.argv[5]

    input_sizes = json.loads(input_sizes_json)
    swt_levels = json.loads(swt_levels_json)

    import pywt

    _t0 = _time.time()

    for input_size in input_sizes:
        win_path = os.path.join(out_dir, f"windows_{input_size}", f"service_{idx:05d}.npy")
        if not os.path.exists(win_path):
            continue

        win_arr = np.load(win_path).astype(np.float64)
        n_windows = win_arr.shape[0]

        for level in swt_levels:
            out_dir_swt = os.path.join(out_dir, f"swt_{input_size}_lv{level}")
            os.makedirs(out_dir_swt, exist_ok=True)
            out_path = os.path.join(out_dir_swt, f"service_{idx:05d}.npy")

            if os.path.exists(out_path):
                continue

            n_comp = level + 1
            all_components = np.zeros((n_windows, n_comp, input_size), dtype=np.float32)

            skip = False
            for wi in range(n_windows):
                window = win_arr[wi]
                if np.std(window) < 1e-12:
                    skip = True
                    break

                coeffs = pywt.swt(window, "sym4", level=level, norm=True, trim_approx=True)
                for ci, c in enumerate(coeffs):
                    all_components[wi, ci] = c.astype(np.float32)

            if skip:
                continue

            np.save(out_path, all_components)

    _elapsed = _time.time() - _t0
    _log(f"[{svc_name}] {_elapsed:.1f}s")
    print(f"RESULT:True:ok")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

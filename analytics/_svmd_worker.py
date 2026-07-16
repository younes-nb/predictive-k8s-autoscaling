import json
import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.svmd import svmd


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    svc_idx = int(sys.argv[1])
    out_dir = sys.argv[2]
    input_size = int(sys.argv[3])
    max_alpha = float(sys.argv[4])
    tau = float(sys.argv[5])
    tol = float(sys.argv[6])
    stop_criteria = int(sys.argv[7])
    init_omega = int(sys.argv[8])
    max_modes = int(sys.argv[9])
    max_inner_iter = int(sys.argv[10]) if len(sys.argv) > 10 else 300

    _t0 = _time.time()

    d1_path = os.path.join(out_dir, f"swt_{input_size}_lv1", f"service_{svc_idx:05d}.npy")
    if not os.path.exists(d1_path):
        print(f"RESULT:False:ERROR: D1 file not found at {d1_path}")
        sys.exit(1)

    components = np.load(d1_path).astype(np.float64)
    mode_counts = []
    for wi in range(components.shape[0]):
        d1 = components[wi, 1]
        if np.std(d1) < 1e-12:
            continue

        modes, _, _ = svmd(
            d1,
            max_alpha=max_alpha,
            tau=tau,
            tol=tol,
            stop_criteria=stop_criteria,
            init_omega=init_omega,
            max_modes=max_modes,
            max_inner_iter=max_inner_iter,
        )
        mode_counts.append(int(modes.shape[0]))

    _elapsed = _time.time() - _t0
    _log(f"[svc_{svc_idx:05d}/is{input_size}] {len(mode_counts)} windows, {_elapsed:.1f}s")

    result = {"idx": svc_idx, "counts": mode_counts}
    print(f"RESULT:True:{json.dumps(result)}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

"""Phase 1 worker: compute SWT energy and sample entropy for one service."""

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


def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2,
                    max_samples: int = 1000) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) > max_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(x), size=max_samples, replace=False))
        x = x[idx]
    N = len(x)
    if N < 2 * (m + 1) + 2:
        return float("nan")
    r = r_frac * float(np.std(x, ddof=1))
    if r == 0.0:
        return float("nan")

    def _count_matches(template_len: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(x, template_len)
        n_templates = len(templates)
        count = 0
        for i in range(n_templates):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            dists[i] = np.inf
            count += int(np.sum(dists < r))
        return count

    Bm = _count_matches(m)
    Am = _count_matches(m + 1)
    if Bm == 0 or Am == 0:
        return float("nan")
    return float(-np.log(Am / Bm))


def main() -> None:
    svc_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    input_sizes_json = sys.argv[4]   # e.g. "[16, 32, 64, 128]"
    swt_levels_json = sys.argv[5]    # e.g. "[1, 2, 3, 4]"

    input_sizes = json.loads(input_sizes_json)
    swt_levels = json.loads(swt_levels_json)

    import pywt

    _t0 = _time.time()

    # {(input_size, level): {"energies": ndarray, "total": float,
    #                         "se_sums": ndarray, "se_counts": ndarray, "count": int}}
    result: dict[tuple[int, int], dict] = {}

    for input_size in input_sizes:
        win_path = os.path.join(out_dir, f"windows_{input_size}", f"service_{idx:05d}.npy")
        if not os.path.exists(win_path):
            continue

        win_arr = np.load(win_path).astype(np.float64)

        for level in swt_levels:
            n_comp = level + 1
            key = (input_size, level)
            if key not in result:
                result[key] = {
                    "energies": np.zeros(n_comp, dtype=np.float64),
                    "total": 0.0,
                    "se_sums": np.zeros(n_comp, dtype=np.float64),
                    "se_counts": np.zeros(n_comp, dtype=np.int64),
                    "count": 0,
                }

            for wi in range(win_arr.shape[0]):
                window = win_arr[wi]
                if np.std(window) < 1e-12:
                    continue

                coeffs = pywt.swt(window, "sym4", level=level, norm=True, trim_approx=True)
                energies = np.array([float(np.sum(c ** 2)) for c in coeffs])
                e_total = float(np.sum(energies))

                result[key]["energies"] += energies
                result[key]["total"] += e_total
                result[key]["count"] += 1

                for ci, c in enumerate(coeffs):
                    se_val = _sample_entropy(c)
                    if not np.isnan(se_val):
                        result[key]["se_sums"][ci] += se_val
                        result[key]["se_counts"][ci] += 1

    # Serialize result as JSON
    out_json = {}
    for (inp_sz, lvl), data in result.items():
        key = f"{inp_sz}_{lvl}"
        se_avgs = np.where(
            data["se_counts"] > 0,
            data["se_sums"] / data["se_counts"],
            float("nan"),
        )
        out_json[key] = {
            "energies": data["energies"].tolist(),
            "total": data["total"],
            "se_avgs": se_avgs.tolist(),
            "se_counts": data["se_counts"].tolist(),
            "count": data["count"],
        }

    _elapsed = _time.time() - _t0
    _log(f"[{svc_name}] {_elapsed:.1f}s")
    print(f"RESULT:True:{json.dumps(out_json)}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

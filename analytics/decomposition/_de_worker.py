import gc
import json
import os
import sys
import time as _time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from EntropyHub import DispEn


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _dispersion_entropy(x: np.ndarray, m: int = 2, c: int = 6) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 10:
        return float("nan")
    if np.ptp(x) == 0.0:
        return float("nan")
    de, _ = DispEn(x, m=m, tau=1, c=c, Typex="ncdf", Norm=True)
    return float(de)


def main() -> None:
    svc_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    input_sizes_json = sys.argv[4]
    swt_levels_json = sys.argv[5]

    input_sizes = json.loads(input_sizes_json)
    swt_levels = json.loads(swt_levels_json)

    cache_dir = os.path.join(out_dir, "phase2_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"service_{idx:05d}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = f.read()
        print(f"RESULT:True:{cached}")
        sys.exit(0)

    _t0 = _time.time()

    result: dict[tuple[int, int], dict] = {}

    for input_size in input_sizes:
        for level in swt_levels:
            swt_path = os.path.join(
                out_dir, f"swt_{input_size}_lv{level}",
                f"service_{idx:05d}.npy",
            )
            if not os.path.exists(swt_path):
                continue

            n_comp = level + 1
            key = (input_size, level)
            if key not in result:
                result[key] = {
                    "de_sums": np.zeros(n_comp, dtype=np.float64),
                    "de_counts": np.zeros(n_comp, dtype=np.int64),
                    "count": 0,
                }

            components = np.load(swt_path)
            for wi in range(components.shape[0]):
                comps = components[wi]
                result[key]["count"] += 1

                for ci, c in enumerate(comps):
                    de_val = _dispersion_entropy(c)
                    if not np.isnan(de_val):
                        result[key]["de_sums"][ci] += de_val
                        result[key]["de_counts"][ci] += 1
            del components
        gc.collect()

    out_json = {}
    for (inp_sz, lvl), data in result.items():
        key = f"{inp_sz}_{lvl}"
        n_comp = lvl + 1
        de_avgs = np.where(
            data["de_counts"] > 0,
            data["de_sums"] / data["de_counts"],
            float("nan"),
        )
        out_json[key] = {
            "de_avgs": de_avgs.tolist(),
            "de_counts": data["de_counts"].tolist(),
            "count": data["count"],
        }

    _elapsed = _time.time() - _t0
    _log(f"[{svc_name}] {_elapsed:.1f}s")
    payload = json.dumps(out_json)
    with open(cache_path, "w") as f:
        f.write(payload)
    print(f"RESULT:True:{payload}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}", file=sys.stdout, flush=True)
        sys.exit(1)

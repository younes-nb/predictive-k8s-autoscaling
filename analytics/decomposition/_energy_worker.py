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
from statsmodels.tsa.stattools import kpss, mackinnonp


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


def _adf(x: np.ndarray) -> float:
    try:
        y = np.asarray(x, dtype=np.float64).ravel()
        N = len(y)
        if N < 6:
            return float("nan")

        dy = np.diff(y)
        n = len(dy)
        maxlag = int(np.ceil(12.0 * (N / 100.0) ** 0.25))
        maxlag = min(maxlag, n - 3)

        best_aic = np.inf
        best_tstat = 0.0

        for p in range(0, maxlag + 1):
            k = n - p
            if k < p + 3:
                break

            dy_t = dy[p:]
            y_lag = y[p : N - 1]

            if p == 0:
                X = np.column_stack([np.ones(k), y_lag])
            else:
                lag_cols = np.column_stack(
                    [dy[p - j : p - j + k] for j in range(1, p + 1)]
                )
                X = np.column_stack([np.ones(k), y_lag, lag_cols])

            XtX = X.T @ X
            Xty = X.T @ dy_t
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue

            resid = dy_t - X @ beta
            rss = float(np.sum(resid ** 2))
            m = X.shape[1]

            aic = k * np.log(rss / k + 1e-30) + 2 * m

            if aic < best_aic:
                best_aic = aic
                try:
                    sigma2 = rss / (k - m)
                    se = np.sqrt(sigma2 * np.linalg.inv(XtX)[1, 1])
                    best_tstat = beta[1] / se if se > 0 else 0.0
                except np.linalg.LinAlgError:
                    best_tstat = 0.0

        return float(mackinnonp(best_tstat, regression="c", N=1))
    except Exception:
        return float("nan")


def _kpss_test(x: np.ndarray) -> float:
    try:
        result = kpss(x, regression="c", nlags="auto")
        return float(result[1])
    except Exception:
        return float("nan")


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
                    "energies": np.zeros(n_comp, dtype=np.float64),
                    "total": 0.0,
                    "de_sums": np.zeros(n_comp, dtype=np.float64),
                    "de_counts": np.zeros(n_comp, dtype=np.int64),
                    "adf_sums": np.zeros(n_comp, dtype=np.float64),
                    "adf_counts": np.zeros(n_comp, dtype=np.int64),
                    "kpss_sums": np.zeros(n_comp, dtype=np.float64),
                    "kpss_counts": np.zeros(n_comp, dtype=np.int64),
                    "count": 0,
                }

            components = np.load(swt_path)
            for wi in range(components.shape[0]):
                comps = components[wi]
                energies = np.sum(comps ** 2, axis=1)
                e_total = float(np.sum(energies))

                result[key]["energies"] += energies
                result[key]["total"] += e_total
                result[key]["count"] += 1

                for ci, c in enumerate(comps):
                    de_val = _dispersion_entropy(c)
                    if not np.isnan(de_val):
                        result[key]["de_sums"][ci] += de_val
                        result[key]["de_counts"][ci] += 1

                    adf_val = _adf(c)
                    if not np.isnan(adf_val):
                        result[key]["adf_sums"][ci] += adf_val
                        result[key]["adf_counts"][ci] += 1

                    kpss_val = _kpss_test(c)
                    if not np.isnan(kpss_val):
                        result[key]["kpss_sums"][ci] += kpss_val
                        result[key]["kpss_counts"][ci] += 1
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
        adf_avgs = np.where(
            data["adf_counts"] > 0,
            data["adf_sums"] / data["adf_counts"],
            float("nan"),
        )
        kpss_avgs = np.where(
            data["kpss_counts"] > 0,
            data["kpss_sums"] / data["kpss_counts"],
            float("nan"),
        )
        out_json[key] = {
            "energies": data["energies"].tolist(),
            "total": data["total"],
            "de_avgs": de_avgs.tolist(),
            "de_counts": data["de_counts"].tolist(),
            "adf_avgs": adf_avgs.tolist(),
            "adf_counts": data["adf_counts"].tolist(),
            "kpss_avgs": kpss_avgs.tolist(),
            "kpss_counts": data["kpss_counts"].tolist(),
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

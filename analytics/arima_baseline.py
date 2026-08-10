import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import (
    FEATURE_SETS,
    feature_names_for_feature_set,
    target_features_for_feature_set,
)
from core.architectures.arima import ArimaForecaster
from training.metrics import compute_metrics, METRIC_NAMES

_CTX = {}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_service_arrays(windows_dir):
    arrays_path = os.path.join(windows_dir, "_service_arrays.npy")
    index_path = os.path.join(windows_dir, "_service_index.json")
    if not (os.path.exists(arrays_path) and os.path.exists(index_path)):
        raise FileNotFoundError(
            f"Service arrays not found in {windows_dir}. Rebuild windows with "
            "`python preprocessing/build_windows.py` first."
        )
    big = np.load(arrays_path)
    with open(index_path, "r") as f:
        index = json.load(f)["index"]
    index = {k: (int(v[0]), int(v[1])) for k, v in index.items()}
    return big, index


def _parse_ints(value):
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_order(value):
    parts = _parse_ints(value)
    if len(parts) != 3:
        raise SystemExit("--order must be 'p,d,q' (e.g. '1,1,0')")
    return tuple(parts)


def _forecast_target(series, cfg):
    H = cfg["horizon"]
    n = int(series.shape[0])
    idx_tr = int(n * cfg["train_frac"])
    idx_val = int(n * (cfg["train_frac"] + cfg["val_frac"]))

    if idx_tr < cfg["min_train_len"] or idx_val < 1 or idx_val + H >= n:
        return None

    train = np.asarray(series[:idx_tr], dtype=float)
    test = np.asarray(series[idx_val:], dtype=float)
    L = len(test)
    N = L - H + 1
    if N < 1:
        return None

    trend = "auto"
    order = cfg["order"]

    preds = np.empty((N, H), dtype=np.float64)
    truths = np.empty((N, H), dtype=np.float64)
    y_last = np.empty(N, dtype=np.float64)

    if cfg["protocol"] == "one_shot":
        try:
            fore = ArimaForecaster(order=order, trend=trend).fit(train).forecast(L)
        except Exception:
            return None
        for t in range(N):
            for h in range(1, H + 1):
                preds[t, h - 1] = fore[t + h - 1]
                truths[t, h - 1] = test[t + h - 1]
            y_last[t] = series[idx_val + t - 1]
    else:
        refit_every = cfg["refit_every"]
        fit_window = cfg["fit_window"]
        t0 = 0
        while t0 < N:
            block = min(refit_every, N - t0)
            window = series[max(0, idx_val + t0 - fit_window): idx_val + t0]
            try:
                fore = ArimaForecaster(order=order, trend=trend).fit(window).forecast(block + H - 1)
            except Exception:
                return None
            for t in range(block):
                for h in range(1, H + 1):
                    preds[t0 + t, h - 1] = fore[t + h - 1]
                    truths[t0 + t, h - 1] = test[t0 + t + h - 1]
                y_last[t0 + t] = series[idx_val + t0 + t - 1]
            t0 += block

    return preds, truths, y_last


def _worker(job):
    cfg, ms_id, off, length = job
    big = _CTX["big"]
    block = np.asarray(big[off:off + length])
    results = {}
    for ti, tname in zip(cfg["target_idxs"], cfg["target_names"]):
        r = _forecast_target(block[:, ti], cfg)
        if r is not None:
            results[tname] = r
    return ms_id, results


def main():
    ap = argparse.ArgumentParser(
        description="Classical ARIMA(p,d,q) statistical baseline: fit per service "
                    "on its train segment, multi-step forecast the test segment. "
                    "Metrics mirror training/evaluate.py."
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR,
                    help="Windows dir containing _service_arrays.npy (default: %(default)s)")
    ap.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET,
                    choices=list(FEATURE_SETS.keys()),
                    help="Feature set; target channels are forecast per-service (default: %(default)s)")
    ap.add_argument("--order", default="1,1,0",
                    help="Fixed 'p,d,q' order applied to every service (default: %(default)s)")
    ap.add_argument("--protocol", choices=["one_shot", "rolling"], default="one_shot",
                    help="one_shot: fit on train, recursive forecast of test. "
                         "rolling: refit every --refit_every steps on a trailing --fit_window.")
    ap.add_argument("--refit_every", type=int, default=25,
                    help="Steps between ARIMA refits for --protocol rolling (default: %(default)s)")
    ap.add_argument("--fit_window", type=int, default=PREPROCESSING.INPUT_LEN,
                    help="Trailing window used when refitting for rolling (default: %(default)s)")
    ap.add_argument("--train_frac", type=float, default=PREPROCESSING.TRAIN_FRAC)
    ap.add_argument("--val_frac", type=float, default=PREPROCESSING.VAL_FRAC)
    ap.add_argument("--horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    ap.add_argument("--max_services", type=int, default=0,
                    help="Limit number of services (0 = all)")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="Worker processes (default: %(default)s)")
    ap.add_argument("--out_csv", default=None,
                    help="Optional CSV path for the per-target metric table")
    args = ap.parse_args()

    if args.protocol == "rolling" and args.horizon > args.refit_every:
        raise SystemExit("--refit_every must be >= --horizon for the rolling protocol.")

    feature_names = feature_names_for_feature_set(args.feature_set)
    target_names = target_features_for_feature_set(args.feature_set)
    target_idxs = [feature_names.index(f) for f in target_names]
    log(f"Feature set: {args.feature_set} | targets: {target_names} | "
        f"feature cols: {feature_names}")

    t0_all = datetime.now()
    big, index = _load_service_arrays(args.windows_dir)
    log(f"Loaded service arrays: {big.shape} | services: {len(index)}")

    if args.max_services and args.max_services > 0:
        ids = list(index.keys())[: args.max_services]
    else:
        ids = list(index.keys())

    cfg = {
        "horizon": args.horizon,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "min_train_len": max(12, args.horizon * 3),
        "protocol": args.protocol,
        "refit_every": args.refit_every,
        "fit_window": args.fit_window,
        "order": _parse_order(args.order),
        "target_names": target_names,
        "target_idxs": target_idxs,
    }

    jobs = [(cfg, ms_id, index[ms_id][0], index[ms_id][1]) for ms_id in ids]
    per_target = {t: {"preds": [], "truths": [], "y_last": [], "n_services": 0}
                  for t in target_names}

    if args.num_workers <= 1:
        pool = None
        _CTX["big"] = big
        results = [_worker(job) for job in jobs]
    else:
        pool = mp.Pool(processes=args.num_workers, initializer=_worker_init, initargs=(big,))
        results = pool.imap_unordered(_worker, jobs, chunksize=1)

    n_ok = 0
    pbar = tqdm(
        total=len(jobs), desc="ARIMA services", unit="svc",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )
    for _ms_id, res in results:
        if res:
            n_ok += 1
        for tname, r in res.items():
            preds, truths, y_last = r
            per_target[tname]["preds"].append(preds)
            per_target[tname]["truths"].append(truths)
            per_target[tname]["y_last"].append(y_last)
            per_target[tname]["n_services"] += 1
        pbar.update(1)
    pbar.close()
    if pool is not None:
        pool.close()
        pool.join()

    log(f"Services evaluated: {n_ok}/{len(jobs)} "
        f"({(datetime.now() - t0_all).total_seconds() / 60.0:.1f} min)")

    all_results = {}
    for tname in target_names:
        bucket = per_target[tname]
        if not bucket["preds"]:
            log(f"\n=== ARIMA: {tname} ===  no services produced forecasts")
            continue
        y_pred = np.concatenate(bucket["preds"], axis=0)
        y_true = np.concatenate(bucket["truths"], axis=0)
        y_last = np.concatenate(bucket["y_last"], axis=0)
        log(f"  {tname}: samples={y_pred.shape[0]} services={bucket['n_services']}")
        results = compute_metrics(
            y_pred, y_true, y_last, args.horizon, y_pred.shape[0], log,
            target_name=f"ARIMA {tname}",
        )
        all_results[tname] = results

    if args.out_csv:
        rows = []
        for tname, results in all_results.items():
            for metric in METRIC_NAMES:
                if metric not in results:
                    continue
                rows.append({
                    "target": tname,
                    "metric": metric,
                    "last_step": results[metric]["last_step"],
                    "avg_steps": results[metric]["avg_steps"],
                    "naive": results[metric]["naive"],
                    "delta_pct": results[metric]["delta_pct"],
                })
        if rows:
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            log(f"Wrote CSV: {args.out_csv}")


def _worker_init(big):
    global _CTX
    _CTX["big"] = big


if __name__ == "__main__":
    main()

"""§4.2 performance profiling: linear CPU/memory vs per-container CPM.

    g^CPU_i(x) = a^CPU_i * x + b^CPU_i      (Eq.9)
    g^Mem_i(x) = a^Mem_i * x + b^Mem_i      (Eq.10)
with x = total_CPM / n_containers for microservice i. Inputs: the CPM
series (ms_call_counts) and the per-minute util aggregation
(madu/aggregate_util.py). Fit by OLS on train-split minutes only.
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from madu.data import load_cpm_series


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-12:
        return 0.0, float(np.mean(y)) if len(y) else 0.0, 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(a), float(b), float(r2)


def main():
    p = argparse.ArgumentParser(description="Fit §4.2 linear CPU/mem profiling.")
    p.add_argument("--ms_list", required=True)
    p.add_argument("--cpm_dir", default="/dataset/ms_call_counts/per_minute")
    p.add_argument("--util_parquet",
                   default="/proj/k8sautoscaledl-PG0/madu/util_per_minute.parquet")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--out", default="/proj/k8sautoscaledl-PG0/madu/profiling.json")
    args = p.parse_args()

    with open(args.ms_list) as f:
        msnames = json.load(f)["msnames"]
    util = pl.read_parquet(args.util_parquet).filter(
        pl.col("msname").is_in(msnames))
    out = {}
    for ms in msnames:
        minutes, cpm = load_cpm_series(ms, args.cpm_dir)
        cpm_map = dict(zip(minutes.tolist(), cpm.tolist()))
        u = util.filter(pl.col("msname") == ms).sort("minute")
        umin = u["minute"].to_numpy()
        x = np.array([cpm_map.get(int(m), 0.0) /
                      max(int(ni), 1) for m, ni in
                      zip(umin, u["n_inst"].to_numpy())])
        cpu = u["cpu"].to_numpy()
        mem = u["mem"].to_numpy()
        n = len(umin)
        n_tr = int(n * args.train_frac)
        a_cpu, b_cpu, r2_cpu = fit_line(x[:n_tr], cpu[:n_tr])
        a_mem, b_mem, r2_mem = fit_line(x[:n_tr], mem[:n_tr])
        out[ms] = {"a_cpu": a_cpu, "b_cpu": b_cpu, "r2_cpu": r2_cpu,
                   "a_mem": a_mem, "b_mem": b_mem, "r2_mem": r2_mem,
                   "n_minutes": n}
        print(f"{ms}: cpu a={a_cpu:.3e} b={b_cpu:.4f} R2={r2_cpu:.3f} | "
              f"mem a={a_mem:.3e} b={b_mem:.4f} R2={r2_mem:.3f} (n={n})",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} profiles -> {args.out}")


if __name__ == "__main__":
    main()

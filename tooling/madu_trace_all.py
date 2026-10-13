#!/usr/bin/env python3
"""Batch Madu CPU/memory traces + plots for all checkpoints.

Phases (run separately; each skip-if-done):
  rollout: serial GPU loop over checkpoints -> per-MS npz
           (test_starts, pred_next, full_cpm, grid, msname).
  plot:    32-worker CPU pool over npz files -> per-MS trace CSV/JSON
           (CPU/memory metrics) + full/zoom PNGs (CPU, Memory, CPM panels).
  summary: aggregate per-MS trace JSONs -> cross-MS means of CPU/memory
           MSE/MAE (plus R2, beat-persistence) -> summary JSON/CSV.

Usage:
  python tooling/madu_trace_all.py --phase rollout --ckpt_dir ... --npz_dir ...
  python tooling/madu_trace_all.py --phase plot --npz_dir ... --plots_dir ... --profile_json ... --util_parquet ...
  python tooling/madu_trace_all.py --phase summary --plots_dir ... --out ...
"""
import argparse
import glob
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MADU_HOURS = 63.0  # covers the full test split (20% of ~18720 min)
THRESHOLD = 0.80

_SUMMARY_KEYS = [
    "cpu_MSE_last_step", "cpu_MAE_last_step",
    "memory_MSE_last_step", "memory_MAE_last_step",
    "cpu_R²_last_step", "memory_R²_last_step",
    "cpu_beat_persistence", "memory_beat_persistence",
    "cpu_corr_pred_true", "memory_corr_pred_true",
    "n_evaluation_minutes",
]


def _args_ns(ckpt, plots_dir):
    import argparse as _ap
    ns = _ap.Namespace(
        madu_ckpt=ckpt, cpm_dir="/dataset/ms_call_counts/per_minute",
        profile_json=None, util_parquet=None, madu_hours=MADU_HOURS,
        plots_dir=plots_dir, threshold=THRESHOLD, device=None,
    )
    return ns


def phase_rollout(ckpt_dir, npz_dir, device):
    from analytics.evaluate_trace import madu_rollout
    os.makedirs(npz_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "madu_*.pt")))
    print(f"rollout: {len(ckpts)} checkpoints", flush=True)
    import numpy as np
    done, failed = 0, []
    for c in ckpts:
        ms = os.path.basename(c)[len("madu_"):-len(".pt")]
        out = os.path.join(npz_dir, f"roll_{ms}.npz")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            done += 1
            continue
        try:
            ro = madu_rollout(_args_ns(c, ""), device)
            np.savez_compressed(out, msname=ro["msname"],
                                test_starts=ro["test_starts"],
                                pred_next=ro["pred_next"],
                                full_cpm=ro["full_cpm"], grid=ro["grid"],
                                n=ro["cfg"]["n"])
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append((ms, repr(e)))
        if (done + len(failed)) % 50 == 0:
            print(f"rollout: {done} done, {len(failed)} failed", flush=True)
    print(f"rollout complete: {done} done, {len(failed)} failed", flush=True)
    for ms, e in failed:
        print(f"  FAILED {ms}: {e}", flush=True)


_UTIL = {}


def _plot_one(job):
    npz_path, plots_dir, prof, util_minutes = job
    import numpy as np
    import pandas as pd
    from analytics.evaluate_trace import compute_metrics, plot_results
    z = np.load(npz_path, allow_pickle=True)
    msname = str(z["msname"])
    stamp = None
    csv_path = os.path.join(plots_dir, f"madu_trace_{msname}.csv")
    json_path = os.path.join(plots_dir, f"madu_trace_{msname}.json")
    if (os.path.isfile(json_path) and os.path.getsize(json_path) > 0
            and glob.glob(os.path.join(
                plots_dir, f"trace_eval_{msname}_full_*.png"))):
        return (msname, "skipped")
    test_starts = z["test_starts"].astype(int)
    pred_next = z["pred_next"].astype(float)
    full_cpm = z["full_cpm"].astype(float)
    grid = z["grid"].astype(int)
    n = int(z["n"])
    p = prof[msname]
    um = util_minutes[msname]
    results, cpm_a, cpm_p = [], [], []
    for s, fc in zip(test_starts, pred_next):
        m_now, m_tgt = int(grid[s + n - 1]), int(grid[s + n])
        if m_now not in um or m_tgt not in um:
            continue
        cpu_now, mem_now, _ = um[m_now]
        _, _, n_inst_tgt = um[m_tgt]
        per_cont = float(fc) / max(n_inst_tgt, 1)
        results.append({"timestamp": None, "cpu": cpu_now, "memory": mem_now,
                        "pred_cpu": max(p["a_cpu"] * per_cont + p["b_cpu"], 0.0),
                        "pred_mem": max(p["a_mem"] * per_cont + p["b_mem"], 0.0),
                        "lower_cpu": 0.0, "upper_cpu": 0.0,
                        "lower_mem": 0.0, "upper_mem": 0.0,
                        "inference_time_s": 0.0})
        cpm_a.append(float(full_cpm[s + n - 1]))
        cpm_p.append(float(fc))
    if not results:
        return (msname, "empty")
    metrics = compute_metrics(results, 1, THRESHOLD,
                              target_features=["cpu_utilization",
                                               "memory_utilization"])
    metrics.update({"msname": msname, "hours": MADU_HOURS})
    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    plot_results(results, msname, plots_dir, 1, THRESHOLD, num_targets=2,
                 target_features=["cpu_utilization", "memory_utilization"],
                 feature_names=["cpu_utilization", "memory_utilization"],
                 cpm_actual=np.asarray(cpm_a), cpm_pred=np.asarray(cpm_p))
    return (msname, "ok")


def phase_plot(npz_dir, plots_dir, profile_json, util_parquet, workers):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(plots_dir, exist_ok=True)
    prof = json.load(open(profile_json))
    print("loading util parquet once...", flush=True)
    udf = pd.read_parquet(util_parquet)
    util_minutes = {}
    for ms, g in udf.groupby("msname"):
        util_minutes[str(ms)] = {int(r.minute): (float(r.cpu), float(r.mem),
                                                 int(r.n_inst))
                                 for r in g.itertuples()}
    del udf
    print(f"util: {len(util_minutes)} MSs", flush=True)
    npzs = sorted(glob.glob(os.path.join(npz_dir, "roll_*.npz")))
    print(f"plot: {len(npzs)} rollouts, {workers} workers", flush=True)
    jobs = [(p, plots_dir, prof, util_minutes) for p in npzs
            if os.path.basename(p)[len("roll_"):-len(".npz")] in prof
            and os.path.basename(p)[len("roll_"):-len(".npz")] in util_minutes]
    print(f"jobs with profile+util: {len(jobs)}", flush=True)
    ok = skip = empty = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_plot_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            try:
                _, st = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"PLOT FAILED {futs[fut]}: {e!r}", flush=True)
                empty += 1
                continue
            ok += st == "ok"
            skip += st == "skipped"
            empty += st == "empty"
            if (ok + skip + empty) % 200 == 0:
                print(f"plot: {ok} ok, {skip} skipped, {empty} empty/fail",
                      flush=True)
    print(f"plot complete: {ok} ok, {skip} skipped, {empty} empty/fail",
          flush=True)


def phase_summary(plots_dir, out):
    import numpy as np
    paths = sorted(glob.glob(os.path.join(plots_dir, "madu_trace_*.json")))
    rows = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        row = {"msname": d.get("msname", os.path.basename(p))}
        for k in _SUMMARY_KEYS:
            v = d.get(k)
            row[k] = float(v) if isinstance(v, (int, float)) else float("nan")
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out.replace(".json", ".csv"), index=False)
    summ = {"n_ms": len(df)}
    for k in _SUMMARY_KEYS:
        a = df[k].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        summ[k + "_mean"] = float(a.mean()) if len(a) else float("nan")
        summ[k + "_n"] = int(len(a))
    with open(out, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"summary: {summ['n_ms']} MSs -> {out}", flush=True)
    for k in ["cpu_MSE_last_step_mean", "cpu_MAE_last_step_mean",
              "memory_MSE_last_step_mean", "memory_MAE_last_step_mean"]:
        print(f"  {k} = {summ[k]:.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Batch Madu traces/plots/summary")
    ap.add_argument("--phase", required=True,
                    choices=["rollout", "plot", "summary"])
    ap.add_argument("--ckpt_dir", default="/proj/k8sautoscaledl-PG0/madu_all/checkpoints")
    ap.add_argument("--npz_dir", default="/proj/k8sautoscaledl-PG0/madu_all/rollouts")
    ap.add_argument("--plots_dir", default="/proj/k8sautoscaledl-PG0/madu_all_plots")
    ap.add_argument("--profile_json",
                    default="/proj/k8sautoscaledl-PG0/madu_all/profiling.json")
    ap.add_argument("--util_parquet",
                    default="/proj/k8sautoscaledl-PG0/madu_all/util_per_minute.parquet")
    ap.add_argument("--out", default="/proj/k8sautoscaledl-PG0/madu_all/cpu_mem_summary.json")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.phase == "rollout":
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device}", flush=True)
        phase_rollout(args.ckpt_dir, args.npz_dir, device)
    elif args.phase == "plot":
        phase_plot(args.npz_dir, args.plots_dir, args.profile_json,
                   args.util_parquet, args.workers)
    else:
        phase_summary(args.plots_dir, args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch Madu CPU/memory traces + plots for all checkpoints.

Uses the FULL decoder horizon (m heads, like training/evaluate.py whose
errors pool all (window, step) samples): rollout saves yhat_full (N, m);
metrics pool every (window, step) sample for CPM/CPU/memory; CSVs carry
per-minute actuals + per-head CPM preds + h=m CPU/mem; PNGs show CPU and
Memory h=m heads plus all CPM heads.

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

# Pooled full-horizon (all m heads) metric keys from per-MS trace JSONs.
_SUMMARY_KEYS = [
    "cpm.mse", "cpm.mae", "cpm.r2", "cpm.beat",
    "cpu.mse", "cpu.mae", "cpu.r2", "cpu.beat",
    "memory.mse", "memory.mae", "memory.r2", "memory.beat",
    "n_samples",
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
                                yhat_full=ro["yhat_full"],
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


_G_PROF = {}
_G_UTIL = {}
_G_PNGSET = None
_G_PLOTSDIR = ""


def _pooled_metrics(y_true, y_pred, y_last):
    """Pooled MSE/MAE/R2/beat-persistence over all (window, step) samples,
    mirroring training/evaluate.py (ravelled multi-step errors)."""
    import numpy as _np
    y_true = _np.asarray(y_true, dtype=float)
    y_pred = _np.asarray(y_pred, dtype=float)
    y_last = _np.asarray(y_last, dtype=float)
    err = y_pred - y_true
    mse = float(_np.mean(err ** 2))
    mae = float(_np.mean(_np.abs(err)))
    ss_res = float(_np.sum(err ** 2))
    ss_tot = float(_np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    base = _np.abs(y_last - y_true)
    beat = float(_np.mean(_np.abs(err) < base)) if len(err) else 0.0
    return {"mse": mse, "mae": mae, "r2": r2, "beat": beat,
            "n": int(len(err))}


def _plot_h5(msname, plots_dir, minutes, actual_cpu, actual_mem, actual_cpm,
             pred_cpu_h5, pred_mem_h5, cpm_heads, m_head, metrics):
    """Madu trace figure in the evaluate_trace style: CPU (actual + h5
    head), Memory (actual + h5 head), CPM (actual + all m heads shifted
    +k). Full + highest-std zoom. Returns (full_path, zoom_path)."""
    import time as _time
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np
    import os as _os
    _os.makedirs(plots_dir, exist_ok=True)
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    x = _np.asarray(minutes)
    cpm_mse = metrics["cpm"]["mse"]
    cpm_mae = metrics["cpm"]["mae"]

    def _draw(xv, ac, pc, am, pm, cpm_a, heads, suffix, zs=None, ze=None,
              fname=None):
        def _slice(a):
            a = _np.asarray(a, dtype=float)
            return a[zs:ze] if zs is not None else a
        xv, ac, pc = _slice(xv), _slice(ac), _slice(pc)
        am, pm = _slice(am), _slice(pm)
        cpm_a = _slice(cpm_a)
        heads = [(k, _slice(mv)) for k, mv in heads]
        fig, axes = _plt.subplots(3, 1, figsize=(22, 12), sharex=True)
        fig.suptitle(f"{msname} — {suffix} (horizon m={m_head}; h={m_head} heads shown, errors pooled over all heads)\n"
                     f"CPM pooled: MSE={cpm_mse:.6f}, MAE={cpm_mae:.6f}",
                     fontsize=13, fontweight="bold", y=0.99)
        ax = axes[0]
        ax.plot(xv, ac, color="#1976D2", linewidth=1.5, label="Actual CPU")
        ax.plot(xv, pc, color="#FF5722", linewidth=1.5, alpha=0.8,
                label=f"Predicted CPU (h={m_head})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("CPU")
        ax.set_title(f"Predicted vs Actual CPU (h={m_head} head)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax = axes[1]
        ax.plot(xv, am, color="#4CAF50", linewidth=1.5, label="Actual Memory")
        ax.plot(xv, pm, color="#FF9800", linewidth=1.5, alpha=0.8,
                label=f"Predicted Memory (h={m_head})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Memory")
        ax.set_title(f"Predicted vs Actual Memory (h={m_head} head)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax = axes[2]
        ax.plot(xv, cpm_a, color="#00695C", linewidth=1.5, label="Actual CPM")
        for k, mv in heads:
            kk = int(k) + 1
            ax.plot(xv, mv, color="#FF6F00", linewidth=1.5, alpha=0.8,
                    label=f"Predicted CPM (h={kk})")
        ax.set_ylabel("CPM")
        ax.set_title(f"Predicted vs Actual CPM (h={m_head} head)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Minute")
        _plt.tight_layout(rect=[0, 0, 1, 0.96])
        fp = _os.path.join(plots_dir, fname)
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        _plt.close(fig)
        return fp

    heads_full = [(m_head - 1, cpm_heads[m_head - 1])]
    full = _draw(x, actual_cpu, pred_cpu_h5, actual_mem, pred_mem_h5,
                 actual_cpm, heads_full, "Full Test Set",
                 fname=f"trace_eval_{msname}_full_{stamp}.png")
    w = 60
    best, center = -1.0, w
    a = _np.asarray(actual_cpu, dtype=float)
    for i in range(w, len(a) - w):
        s = float(_np.std(a[i - w:i + w]))
        if s > best:
            best, center = s, i
    zs, ze = max(0, center - w), min(len(a), center + w)
    zoom = _draw(x, actual_cpu, pred_cpu_h5, actual_mem, pred_mem_h5,
                 actual_cpm, heads_full,
                 f"Zoomed (min {zs}-{ze}, highest-std window)",
                 zs=zs, ze=ze,
                 fname=f"trace_eval_{msname}_zoom_{stamp}.png")
    return full, zoom


def _plot_one(npz_path):
    """Full-horizon trace outputs for one MS.

    Uses every decoder head k=1..m (like training/evaluate.py, whose
    errors pool all (window, step) samples). Metrics pool all samples;
    CSV carries per-minute actuals + per-head CPM preds + h=m CPU/mem;
    PNGs (png_set members only) show CPU/Memory h=m heads and all CPM
    heads. Returns (msname, status).
    """
    plots_dir = _G_PLOTSDIR
    prof = _G_PROF
    util_minutes = _G_UTIL
    png_set = _G_PNGSET
    import glob as _glob
    import numpy as _np
    import pandas as _pd
    import os as _os
    z = _np.load(npz_path, allow_pickle=True)
    msname = str(z["msname"])
    csv_path = _os.path.join(plots_dir, f"madu_trace_{msname}.csv")
    json_path = _os.path.join(plots_dir, f"madu_trace_{msname}.json")
    want_png = png_set is None or msname in png_set
    has_json = _os.path.isfile(json_path) and _os.path.getsize(json_path) > 0
    has_png = bool(_glob.glob(_os.path.join(
        plots_dir, f"trace_eval_{msname}_full_*.png")))
    if has_json and (has_png or not want_png):
        return (msname, "skipped")
    if "yhat_full" not in z:
        return (msname, "no-h5")
    test_starts = z["test_starts"].astype(int)
    yhat_full = z["yhat_full"].astype(float)
    full_cpm = z["full_cpm"].astype(float)
    grid = z["grid"].astype(int)
    n = int(z["n"])
    m = int(yhat_full.shape[1])
    p = prof[msname]
    arr = util_minutes[msname]  # numpy [minute, cpu, mem, n_inst], sorted
    mins = arr[:, 0]

    def _util_at(minute):
        i = int(_np.searchsorted(mins, minute))
        if i < len(arr) and arr[i, 0] == minute:
            return float(arr[i, 1]), float(arr[i, 2]), int(arr[i, 3])
        return None

    # pooled samples over all (window, step), mirroring ravelled errors
    yt_cpm, yp_cpm, yl_cpm = [], [], []
    yt_cpu, yp_cpu, yl_cpu = [], [], []
    yt_mem, yp_mem, yl_mem = [], [], []
    # per-minute aligned series for CSV/plots
    per_min = {}
    for s, yf in zip(test_starts, yhat_full):
        for k in range(m):
            t = int(s + n + k)
            minute = int(grid[t])
            u = _util_at(minute)
            if u is None:
                continue
            cpu_a, mem_a, n_inst = u
            fc = float(yf[k])
            per_cont = fc / max(n_inst, 1)
            cpu_p = max(p["a_cpu"] * per_cont + p["b_cpu"], 0.0)
            mem_p = max(p["a_mem"] * per_cont + p["b_mem"], 0.0)
            cpm_a = float(full_cpm[t])
            yt_cpm.append(cpm_a)
            yp_cpm.append(fc)
            yl_cpm.append(float(full_cpm[t - 1]))
            yt_cpu.append(cpu_a)
            yp_cpu.append(cpu_p)
            um = _util_at(minute - 1)
            yl_cpu.append(um[0] if um is not None else cpu_a)
            yt_mem.append(mem_a)
            yp_mem.append(mem_p)
            yl_mem.append(um[1] if um is not None else mem_a)
            d = per_min.setdefault(minute, {"cpu": cpu_a, "memory": mem_a,
                                            "cpm": cpm_a})
            d[f"cpm_h{k + 1}"] = fc
            if k == m - 1:
                d["cpu_h5"] = cpu_p
                d["mem_h5"] = mem_p
    if not yt_cpm:
        return (msname, "empty")
    metrics = {"msname": msname, "hours": MADU_HOURS, "horizon": m,
               "n_windows": int(len(test_starts)),
               "n_samples": int(len(yt_cpm)),
               "cpm": _pooled_metrics(yt_cpm, yp_cpm, yl_cpm),
               "cpu": _pooled_metrics(yt_cpu, yp_cpu, yl_cpu),
               "memory": _pooled_metrics(yt_mem, yp_mem, yl_mem)}
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    frame = _pd.DataFrame.from_dict(per_min, orient="index").sort_index()
    frame.index.name = "minute"
    frame.to_csv(csv_path)
    if png_set is not None and msname not in png_set:
        return (msname, "ok-nopng")
    minutes = frame.index.to_numpy()
    get = lambda c, fill: frame[c].to_numpy(dtype=float) if c in frame else _np.full(len(frame), fill)
    _plot_h5(msname, plots_dir, minutes,
             get("cpu", _np.nan), get("memory", _np.nan), get("cpm", _np.nan),
             get("cpu_h5", _np.nan), get("mem_h5", _np.nan),
             {k: get(f"cpm_h{k + 1}", _np.nan) for k in range(m)},
             m, metrics)
    return (msname, "ok")


def phase_plot(npz_dir, plots_dir, profile_json, util_parquet, workers,
               png_list=None, ms_allow=None):
    import numpy as np
    import polars as pl
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(plots_dir, exist_ok=True)
    prof = json.load(open(profile_json))
    print("loading util parquet once (compact numpy partitions)...",
          flush=True)
    udf = pl.read_parquet(util_parquet,
                          columns=["msname", "minute", "cpu", "mem", "n_inst"])
    # compact per-MS float64 arrays (fork-shared, read-only); ~1.5 GB total
    # for 87M rows — never expanded to per-minute Python dicts.
    util_minutes = {}
    for ms, g in udf.group_by("msname"):
        key = ms[0] if isinstance(ms, tuple) else ms
        a = np.asarray(
            g.select(["minute", "cpu", "mem", "n_inst"]).to_numpy(),
            dtype=np.float64)
        util_minutes[str(key)] = a[a[:, 0].argsort()]
    del udf
    import gc
    gc.collect()
    print(f"util: {len(util_minutes)} MSs", flush=True)
    npzs = sorted(glob.glob(os.path.join(npz_dir, "roll_*.npz")))
    print(f"plot: {len(npzs)} rollouts, {workers} workers", flush=True)
    png_set = None
    if png_list:
        png_set = set(json.load(open(png_list))["msnames"])
        print(f"PNGs restricted to {len(png_set)} MSs", flush=True)
    allowed = set(ms_allow) if ms_allow else None
    global _G_PROF, _G_UTIL, _G_PNGSET, _G_PLOTSDIR
    _G_PROF, _G_UTIL, _G_PNGSET, _G_PLOTSDIR = prof, util_minutes, png_set, plots_dir
    jobs = []
    for p in npzs:
        ms = os.path.basename(p)[len("roll_"):-len(".npz")]
        if ms not in prof or ms not in util_minutes:
            continue
        if allowed is not None and ms not in allowed:
            continue
        jobs.append(p)  # paths only: bulk payloads ride fork-shared globals
    print(f"jobs with profile+util: {len(jobs)}", flush=True)
    ok = skip = empty = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_plot_one, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                _, st = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"PLOT FAILED {futs[fut]}: {e!r}", flush=True)
                empty += 1
                continue
            ok += st in ("ok", "ok-nopng")
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
            v = d
            for part in k.split("."):
                v = v.get(part) if isinstance(v, dict) else None
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
    for k in ["cpm.mse_mean", "cpm.mae_mean",
              "cpu.mse_mean", "cpu.mae_mean",
              "memory.mse_mean", "memory.mae_mean"]:
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
    ap.add_argument("--png_list", default=None,
                    help="JSON {msnames:[...]}: PNGs only for these MSs "
                         "(metrics+CSV still for all jobs)")
    ap.add_argument("--ms_list", default=None,
                    help="JSON {msnames:[...]}: restrict jobs to these MSs")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.phase == "rollout":
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device}", flush=True)
        phase_rollout(args.ckpt_dir, args.npz_dir, device)
    elif args.phase == "plot":
        ms_allow = (json.load(open(args.ms_list))["msnames"]
                    if args.ms_list else None)
        phase_plot(args.npz_dir, args.plots_dir, args.profile_json,
                   args.util_parquet, args.workers, args.png_list, ms_allow)
    else:
        phase_summary(args.plots_dir, args.out)


if __name__ == "__main__":
    main()

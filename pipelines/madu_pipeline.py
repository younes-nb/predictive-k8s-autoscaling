"""Madu pipeline: select -> aggregate -> train -> profile -> evaluate -> plot.

Adopts the Madu workload predictor (madu/) into pipelines/ following the
full_pipeline.py conventions (argparse stages, shared.subprocess_utils.run,
per-stage timing). Trains one individual model per microservice (paper
§6.2.1) on the 100 highest-CPM-variance MSs, fits the §4.2 linear CPU/mem
profiles, evaluates, and plots predicted CPM/CPU/memory per MS via
analytics/evaluate_trace.py (Madu trace mode).
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

from shared.subprocess_utils import run

MADU_DIR = "/proj/k8sautoscaledl-PG0/madu"


def main():
    ap = argparse.ArgumentParser(description="Madu end-to-end pipeline")
    ap.add_argument("--madu_dir", default=MADU_DIR)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--reselect", action="store_true",
                    help="Re-run high-variance MS selection")
    ap.add_argument("--skip_select", action="store_true")
    ap.add_argument("--skip_aggregate", action="store_true")
    ap.add_argument("--recompute_util", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_profile", action="store_true")
    ap.add_argument("--skip_testing", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--attention", default="stochastic",
                    choices=["stochastic", "deterministic", "none"])
    ap.add_argument("--no_uncertainty", action="store_true")
    ap.add_argument("--arima", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--arima_refit", type=int, default=288)
    ap.add_argument("--madu_hours", type=float, default=24.0)
    ap.add_argument("--max_plot", type=int, default=0,
                    help="Plot only first N MSs (0 = all)")
    ap.add_argument("--plots_dir", default="/proj/k8sautoscaledl-PG0/madu_plots")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.madu_dir, exist_ok=True)
    ms_list = os.path.join(args.madu_dir, "ms_list.json")
    util_pq = os.path.join(args.madu_dir, "util_per_minute.parquet")
    prof_json = os.path.join(args.madu_dir, "profiling.json")
    ckpt_dir = os.path.join(args.madu_dir, "checkpoints")
    total_times = {}

    if not args.skip_select and (args.reselect or not os.path.exists(ms_list)):
        total_times["select"] = run(
            [sys.executable, "madu/select_ms.py", "--top_k", str(args.top_k),
             "--out", ms_list],
            "Step 1: Select high-variance MSs")
    with open(ms_list) as f:
        msnames = json.load(f)["msnames"]
    print(f"Selected {len(msnames)} MSs")

    if not args.skip_aggregate and (args.recompute_util or not os.path.exists(util_pq)):
        total_times["aggregate"] = run(
            [sys.executable, "madu/aggregate_util.py", "--ms_list", ms_list,
             "--out", util_pq],
            "Step 2: Aggregate per-minute util")

    if not args.skip_training:
        cmd = ([sys.executable, "madu/train.py"]
               + sum((["--msname", m] for m in msnames), [])
               + ["--epochs", str(args.epochs), "--n", str(args.n),
                  "--m", str(args.m), "--hidden", str(args.hidden),
                  "--batch_size", str(args.batch_size), "--lr", str(args.lr),
                  "--attention", args.attention,
                  "--checkpoint_dir", ckpt_dir, "--seed", str(args.seed)])
        if args.no_uncertainty:
            cmd.append("--no_uncertainty")
        if args.cpu:
            cmd.append("--cpu")
        total_times["training"] = run(cmd, "Step 3: Train Madu per MS")

    if not args.skip_profile:
        total_times["profile"] = run(
            [sys.executable, "madu/profiling.py", "--ms_list", ms_list,
             "--util_parquet", util_pq, "--out", prof_json],
            "Step 4: Fit CPU/mem profiles (Eq.9-10)")

    if not args.skip_testing:
        cmd = [sys.executable, "madu/evaluate.py", "--checkpoint_dir", ckpt_dir,
               "--out_json", os.path.join(args.madu_dir, "eval_all.json"),
               "--arima_refit", str(args.arima_refit)]
        cmd.append("--arima" if args.arima else "--no-arima")
        if args.cpu:
            cmd.append("--cpu")
        total_times["testing"] = run(cmd, "Step 5: Evaluate all MSs")

    if not args.skip_plots:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "madu_*.pt")))
        if args.max_plot:
            ckpts = ckpts[:args.max_plot]
        for c in ckpts:
            total_times.setdefault("plots", 0.0)
            total_times["plots"] += run(
                [sys.executable, "analytics/evaluate_trace.py",
                 "--madu_ckpt", c, "--profile_json", prof_json,
                 "--util_parquet", util_pq, "--madu_hours", str(args.madu_hours),
                 "--plots_dir", args.plots_dir]
                + (["--device", "cpu"] if args.cpu else []),
                f"Step 6: Plot {os.path.basename(c)}")

    print("\n========== MADU PIPELINE COMPLETE ==========")
    for stage, t in total_times.items():
        print(f"{stage:>20}: {t:.2f}s")
    print("===========================================")


if __name__ == "__main__":
    main()

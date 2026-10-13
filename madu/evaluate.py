"""Evaluate Madu predictor(s) (§5.1.1): Eq.(15) accuracy by CPM percentile,
±10% accurate fraction (Fig.9a), overestimation accuracy Eq.(16)
(Fig.9b), plus ARIMA / persistence baselines on identical test windows.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from madu.baselines import arima_rolling_forecast, naive_forecast
from madu.data import prepare_ms
from madu.metrics import evaluate_by_percentile
from madu.model import MaduSeq2Seq
from shared.logging_utils import setup_logging


def invert_norm(arr, norm):
    """Map normalized predictions back to raw CPM (per checkpoint norm)."""
    if norm.get("mode", "max") == "log1p" or norm.get("transform") == "log1p":
        out = np.expm1(np.asarray(arr, dtype=float) * norm["std"] + norm["mean"])
    else:
        out = np.asarray(arr, dtype=float) * norm["scale"]
    return np.maximum(out, 0.0)


def predict_madu(model, loader, device, norm, mu_only=False):
    pocket = []
    with torch.no_grad():
        for xb, _ in loader:
            yhat, mu, _ = model.predict(xb.float().to(device))
            pocket.append((mu if mu_only else yhat).cpu().numpy())
    return invert_norm(np.concatenate(pocket, axis=0), norm)


def evaluate_ckpt(ckpt_path, args, device, log_info):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, norm, msname = ckpt["config"], ckpt["norm"], ckpt["msname"]
    prep = prepare_ms(msname, cfg["n"], cfg["m"], ckpt.get("parquet_dir", args.parquet_dir),
                      norm["train_frac"], norm["val_frac"], norm["max_minutes"],
                      norm.get("transform", "max"), norm.get("feat", "paper"),
                      norm.get("resample", 1), norm.get("resample_how", "mean"))
    if norm.get("mode", norm.get("transform", "max")) == "max":
        assert abs(prep["norm"].get("scale", 0) - norm["scale"]) < 1e-9, \
            "train scale mismatch"
    model = MaduSeq2Seq(input_size=prep["test"][0][0].shape[-1],
                        hidden_size=cfg["hidden"],
                        num_layers=cfg["layers"], horizon=cfg["m"],
                        attention=cfg["attention"],
                        uncertainty=cfg["uncertainty"],
                        dropout=cfg.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    loader = DataLoader(prep["test"], batch_size=args.batch_size, shuffle=False)
    y_true = prep["test_raw"]
    results = {"msname": msname}
    y_madu = predict_madu(model, loader, device, prep["norm"],
                        mu_only=args.mu_only)
    results["madu"] = evaluate_by_percentile(y_true, y_madu)
    log_info(f"\n=== {msname}: Madu (n={cfg['n']}, m={cfg['m']}, "
             f"attn={cfg['attention']}, unc={cfg['uncertainty']}) ===")
    _print_table(results["madu"], y_true, y_madu, log_info)

    if args.baselines:
        # identical test windows: history end = last observed index
        n, m = cfg["n"], cfg["m"]
        n_win = len(prep["test_raw"])
        # window starts in raw-series coordinates: test windows follow train+val
        full = prep["series_cpm"]
        n_total_win = len(full) - n - m + 1
        test_starts = np.arange(n_total_win - n_win, n_total_win)
        last_obs = np.array([full[s + n - 1] for s in test_starts])
        y_naive = naive_forecast(last_obs, m)
        results["naive"] = evaluate_by_percentile(y_true, y_naive)
        log_info(f"\n=== {msname}: Naive (persistence) ===")
        _print_table(results["naive"], y_true, y_naive, log_info)
        if args.arima:
            y_arima = arima_rolling_forecast(
                full, test_starts, n, m, order=tuple(args.arima_order),
                refit_every=args.arima_refit)
            results["arima"] = evaluate_by_percentile(y_true, y_arima)
            log_info(f"\n=== {msname}: ARIMA{args.arima_order} "
                     f"(refit every {args.arima_refit}) ===")
            _print_table(results["arima"], y_true, y_arima, log_info)
    return results


SUMMARY_GROUPS = ("0-50%", "50-95%", "95-100%")


def _nanmean(vals):
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def summarize_results(all_results):
    """Table-2 style cross-MS summary (paper §5.1.1).

    Group cells = mean over MSs of the per-MS group accuracy ("quantify
    the prediction accuracy of each group and take the average across all
    microservices"). The 'avg' row = mean over MSs of the per-MS pooled
    ('overall') accuracy — this is what Table 2's Avg row reports (it
    equals the sample-pooled value, not the mean of the group cells).
    NaN groups (e.g. empty percentile bins) are skipped.
    """
    methods = sorted({k for r in all_results for k in r if k != "msname"})
    summary = {}
    for meth in methods:
        have = [r[meth] for r in all_results if meth in r]
        g = {}
        for grp in SUMMARY_GROUPS:
            accs = [h[grp]["accuracy"] for h in have if grp in h]
            meds = [h[grp]["median_accuracy"] for h in have if grp in h]
            ns = [h[grp]["n"] for h in have if grp in h and "n" in h[grp]]
            g[grp] = {"accuracy": _nanmean(accs),
                      "median_accuracy": _nanmean(meds),
                      "n_ms": len(accs), "n": int(sum(ns))}
        ov = [h["overall"] for h in have if "overall" in h]
        g["avg"] = {"accuracy": _nanmean([o["accuracy"] for o in ov]),
                    "median_accuracy": _nanmean([o["median_accuracy"] for o in ov]),
                    "n_ms": len(ov),
                    "n": int(sum(o["n"] for o in ov))}
        g["accurate_frac_10pct"] = _nanmean(
            [h["accurate_frac_10pct"] for h in have if "accurate_frac_10pct" in h])
        g["overestimation_accuracy"] = _nanmean(
            [h["overestimation_accuracy"] for h in have if "overestimation_accuracy" in h])
        g["mse"] = _nanmean([h["mse"] for h in have if "mse" in h])
        g["mae"] = _nanmean([h["mae"] for h in have if "mae" in h])
        g["n_ms_total"] = len(have)
        summary[meth] = g
    return summary


def _print_summary(summary, log_info):
    for meth, g in summary.items():
        log_info(f"--- cross-MS summary: {meth} "
                 f"(n_ms={g['n_ms_total']}) ---")
        log_info(f"{'group':<10s} {'accuracy':>10s} {'med-acc':>10s} "
                 f"{'n_ms':>6s} {'n':>10s}")
        for grp in (*SUMMARY_GROUPS, "avg"):
            log_info(f"{grp:<10s} {g[grp]['accuracy']:>10.4f} "
                     f"{g[grp]['median_accuracy']:>10.4f} "
                     f"{g[grp]['n_ms']:>6d} {g[grp]['n']:>10d}")
        log_info(f"accurate±10% (mean over MSs): {g['accurate_frac_10pct']:.4f} | "
                 f"overest_acc (mean over MSs): {g['overestimation_accuracy']:.4f} | "
                 f"MSE (mean over MSs): {g['mse']:.4f} | "
                 f"MAE (mean over MSs): {g['mae']:.4f}")


def _print_table(res, y_true, y_pred, log_info):
    import numpy as np
    log_info(f"{'group':<10s} {'accuracy':>10s} {'med-acc':>10s} {'n':>10s}")
    for g in ("0-50%", "50-95%", "95-100%", "avg", "overall"):
        log_info(f"{g:<10s} {res[g]['accuracy']:>10.4f} "
                 f"{res[g]['median_accuracy']:>10.4f} {res[g]['n']:>10d}")
    log_info(f"accurate±10%: {res['accurate_frac_10pct']:.4f} | "
             f"overest_acc: {res['overestimation_accuracy']:.4f} | "
             f"MAE: {np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))):.1f}")


def main():
    p = argparse.ArgumentParser(description="Evaluate Madu predictors (SoCC'22 §5.1.1).")
    p.add_argument("--checkpoint", action="append", default=[],
                   help="Checkpoint(s) from madu/train.py (repeatable).")
    p.add_argument("--checkpoint_dir", default="/proj/k8sautoscaledl-PG0/madu",
                   help="Evaluate all checkpoints in this dir if --checkpoint omitted.")
    p.add_argument("--parquet_dir", default="/dataset/ms_call_counts/per_minute")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arima", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arima_order", type=int, nargs=3, default=[5, 1, 0])
    p.add_argument("--arima_refit", type=int, default=288)
    p.add_argument("--mu_only", action="store_true",
                   help="Ablation: predict with the mu head only (no +sigma).")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    setup_logging("madu_evaluate")
    log_info = logging.info
    ckpts = list(args.checkpoint)
    if not ckpts:
        ckpts = sorted(
            os.path.join(args.checkpoint_dir, f)
            for f in os.listdir(args.checkpoint_dir)
            if f.endswith(".pt")
        )
    if not ckpts:
        raise SystemExit("No checkpoints found.")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    all_results = [evaluate_ckpt(c, args, device, log_info) for c in ckpts]
    summary = summarize_results(all_results)
    _print_summary(summary, log_info)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"per_ms": all_results, "summary": summary},
                      f, indent=2, default=float)
        log_info(f"results -> {args.out_json}")


if __name__ == "__main__":
    main()

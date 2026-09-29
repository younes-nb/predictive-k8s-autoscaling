#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.dataset import ShardedWindowsDataset
from training.sfoa_configs import get_config
from types import SimpleNamespace


def log(msg):
    print(msg, flush=True)


def build_model(checkpoint, input_size, device):
    from core.models import ChangeHeadForecaster
    ckpt_args = checkpoint.get("args", {})
    model_type = checkpoint.get("model_type", "lstm")
    from shared.features import target_features_for_feature_set
    from shared.config_preprocessing_defaults import PREPROCESSING
    num_targets = len(target_features_for_feature_set(
        ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)))
    cfg = get_config(model_type)
    hyperparams = checkpoint.get("hyperparams", cfg.DEFAULTS)
    model = cfg.build_model(hyperparams, input_size,
                            SimpleNamespace(**ckpt_args), num_targets, device)
    if ckpt_args.get("change_head", False) or ckpt_args.get("change_head_mem", False):
        inject_mask = None
        if ckpt_args.get("change_head_mem", False):
            inject_mask = [False] * num_targets
            inject_mask[-1 if num_targets > 1 else 0] = True
        model = ChangeHeadForecaster(model, inject_mask)
    return model, ckpt_args


def pr_at_recall(y_true, scores, target_recall=0.5):
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    tp = np.cumsum(y_true[order] == 1)
    rec = tp / max(1, y_true.sum())
    prec = tp / (np.arange(len(y_true)) + 1)
    pm = prec[rec >= target_recall]
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return ap, (round(float(pm.max()), 4) if len(pm) else 0.0)


def evaluate_task(name, y_true, scores, pred_level, last, truth_last):
    y_true = np.asarray(y_true, dtype=int)
    n_pos = int(y_true.sum())
    base = float(y_true.mean())
    out = {"n_events": n_pos, "base_rate": round(base, 5)}
    if n_pos == 0:
        out.update({"pr_auc": None, "prec_at_rec50": None,
                    "mae_model": None, "mae_persist": None, "mae_ratio": None})
        return out
    ap, p50 = pr_at_recall(y_true, scores)
    out["pr_auc"] = round(ap, 4)
    out["prec_at_rec50"] = p50
    m = y_true == 1
    mae_m = float(np.abs(pred_level[m] - truth_last[m]).mean())
    mae_p = float(np.abs(last[m] - truth_last[m]).mean())
    out["mae_model"] = round(mae_m, 5)
    out["mae_persist"] = round(mae_p, 5)
    out["mae_ratio"] = round(mae_m / (mae_p + 1e-12), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--delta", type=float, default=0.2,
                    help="Absolute cpu jump defining a cold transition")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    input_len = ckpt_args.get("input_len")
    horizon = ckpt_args.get("pred_horizon", 5)
    feature_set = ckpt_args.get("feature_set")
    pre = checkpoint.get("preprocess_approach", "none")
    if pre != "none":
        raise SystemExit("transition_metric currently supports approach 'none' only")
    log(f"model={checkpoint.get('model_type')} features={feature_set} "
        f"L={input_len} H={horizon} delta={args.delta}")

    with open(os.path.join(args.windows_dir, "_service_index.json")) as f:
        feat_names = json.load(f)["features"]
    cpu_idx = feat_names.index("cpu_utilization")
    vel_back = min(horizon, input_len - 1)
    rps_idx = feat_names.index("rps_total") if "rps_total" in feat_names else None
    slope_idx = (feat_names.index("rps_slope5") if "rps_slope5" in feat_names
                 else None)

    ds = ShardedWindowsDataset(args.windows_dir, args.split, input_len, horizon)
    if len(ds) == 0:
        raise SystemExit("empty dataset split")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    first_x, *_ = ds[0]
    model, _ = build_model(checkpoint, first_x.shape[-1], device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    P, T, L, V, R, S = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].float(), batch[1].float()
            p = model(x)
            if p.dim() == 3:
                p = p[:, :, 0] if p.shape[-1] == 1 else p[:, :, 0]
            if y.dim() == 3:
                y = y[:, :, 0]
            P.append(p[:, -1].numpy())
            T.append(y[:, -1].numpy())
            xn = x.numpy()
            L.append(xn[:, -1, cpu_idx])
            V.append(xn[:, -1, cpu_idx] - xn[:, -1 - vel_back, cpu_idx])
            if rps_idx is not None and slope_idx is not None:
                R.append(xn[:, -1, slope_idx] * horizon)
                S.append(xn[:, -1, rps_idx])
    pred, truth, last = (np.concatenate(a) for a in (P, T, L))
    vel = np.concatenate(V)
    n = len(pred)
    log(f"test windows: {n}")

    jump_true = truth - last
    jump_pred = pred - last
    spike = (jump_true >= args.delta).astype(int)
    drop = (jump_true <= -args.delta).astype(int)

    res = {"checkpoint": args.checkpoint, "split": args.split,
           "n": n, "delta": args.delta, "horizon": horizon,
           "services_note": "pooled over services (csv windows carry no sid)",
           "has_event_head": bool(hasattr(model, "event_logits"))}
    if hasattr(model, "event_logits"):
        EP = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].float()
                e = model.event_logits(x)
                if e.dim() == 2:
                    e = e.unsqueeze(-1).expand(-1, -1, 2)
                EP.append(torch.sigmoid(e[:, -1, :]).numpy())
        eprob = np.concatenate(EP)
        for ename, lab, col in (("spike", spike, 0), ("drop", drop, 1)):
            ap, p50 = pr_at_recall(lab, eprob[:, col])
            res[f"event_{ename}"] = {"n_events": int(lab.sum()),
                                     "pr_auc": round(ap, 4),
                                     "prec_at_rec50": p50}
    res["spike"] = evaluate_task("spike", spike, jump_pred, pred, last, truth)
    res["drop"] = evaluate_task("drop", drop, -jump_pred, pred, last, truth)
    res["spike_vel"] = evaluate_task("spike_vel", spike, vel, last + vel, last, truth)
    if rps_idx is not None and slope_idx is not None:
        rslope = np.concatenate(R)
        r = evaluate_task("spike_rpsslope", spike, rslope, last, last, truth)
        r["mae_ratio"] = 1.0
        res["spike_rpsslope"] = r
    mae_m = float(np.abs(pred - truth).mean())
    mae_p = float(np.abs(last - truth).mean())
    res["guard"] = {"mae_model": round(mae_m, 5),
                    "mae_persist": round(mae_p, 5),
                    "mae_ratio": round(mae_m / (mae_p + 1e-12), 4),
                    "bias": round(float((pred - truth).mean()), 5),
                    "pass": bool(mae_m <= mae_p)}
    cheat = np.full(n, last + 1.0)
    res["cheater"] = {
        "spike_recall": round(float(((cheat - last) >= args.delta)[spike == 1].mean())
                              if spike.sum() else 0.0, 4),
        "spike_precision": round(float(spike[(cheat - last) >= args.delta].mean())
                                 if ((cheat - last) >= args.delta).sum() else 0.0, 4),
        "global_mae_ratio": round(float(np.abs(cheat - truth).mean()) / (mae_p + 1e-12), 2),
    }

    for task in ("spike", "drop", "spike_vel", "spike_rpsslope"):
        if task in res:
            r = res[task]
            log(f"{task:>14}: n={r['n_events']} base={r['base_rate']} "
                f"PR-AUC={r['pr_auc']} P@R50={r['prec_at_rec50']} "
                f"transMAE_ratio={r['mae_ratio']}")
    for task in ("event_spike", "event_drop"):
        if task in res:
            r = res[task]
            log(f"{task:>14}: n={r['n_events']} "
                f"PR-AUC={r['pr_auc']} P@R50={r['prec_at_rec50']}")
    log(f"GUARD global MAE ratio={res['guard']['mae_ratio']} "
        f"bias={res['guard']['bias']} pass={res['guard']['pass']}")
    log(f"CHEATER spike recall={res['cheater']['spike_recall']} "
        f"prec={res['cheater']['spike_precision']} "
        f"globalMAE_ratio={res['cheater']['global_mae_ratio']} (must be >>1)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log(f"saved -> {args.out}")


if __name__ == "__main__":
    main()


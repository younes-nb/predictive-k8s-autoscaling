#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd

H = 5
LAGS_FULL = list(range(0, 13))
LAGS_SHORT = [0, 1, 2, 3]
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


def pr_scores(y_true, scores):
    order = np.argsort(-scores)
    tp = np.cumsum(np.asarray(y_true, dtype=float)[order] == 1)
    rec = tp / max(1, np.asarray(y_true).sum())
    prec = tp / (np.arange(len(y_true)) + 1)
    pm = prec[rec >= 0.5]
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return ap, (round(float(pm.max()), 4) if len(pm) else 0.0)


def make_stage1(kind, seed=42):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=63, min_samples_leaf=50,
                                         random_state=seed)


def make_stage2(kind, seed=42):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if kind == "hgb_q90":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.9,
                                             max_iter=200, learning_rate=0.06,
                                             max_leaf_nodes=31,
                                             min_samples_leaf=100,
                                             random_state=seed)
    if kind == "hurdle":
        return None
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                         max_leaf_nodes=31, min_samples_leaf=100,
                                         random_state=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--s1", default="ridge", choices=["ridge", "hgb"])
    ap.add_argument("--s2", default="ridge",
                    choices=["ridge", "hgb", "hgb_q90", "hurdle"],
                    help="hurdle: balanced event gates x event-only magnitude "
                         "specialists (zero-inflated style level forecast)")
    ap.add_argument("--s2-weight", default="none", choices=["none", "jump"],
                    help="jump: sample-weight stage-2 by (1+5*|cpu jump|) to "
                         "emphasize transitions (cost-sensitive, no resampling)")
    ap.add_argument("--target", default="level", choices=["level", "peak"],
                    help="peak: forecast max cpu over next H steps instead of "
                         "level at H (peak-capture task)")
    ap.add_argument("--s1-lags", default="full", choices=["full", "short"],
                    help="short: stage-1 uses only lags 0..3 (more reactive, "
                         "less smoothing lag)")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--s1-feats", default="basic", choices=["basic", "wide"],
                    help="wide adds caller/mesh/frontend rps lags to stage 1")
    ap.add_argument("--s2-in", default="rps", choices=["rps", "full", "jvm"],
                    help="full adds current cpu; jvm adds current jvm_heap / "
                         "nonheap / gc / threads (needs --aux-csv)")
    ap.add_argument("--s1-rps-col", default="rps_total",
                    choices=["rps_total", "rps_sharp"],
                    help="rps_sharp: 30s-rate RPS from --aux-csv (sharper "
                         "timing, noisier)")
    ap.add_argument("--aux-csv", default=None,
                    help="auxiliary CSV (timestamp, msname, jvm_*, rps_sharp) "
                         "merged on identical grid; jvm cols train-zone "
                         "z-scored per service, NaN (no agent) -> 0 post-scale")
    ap.add_argument("--s2-clf", action="store_true",
                    help="additionally fit balanced HGB spike/drop event "
                         "classifiers on [rps_hat, delta_hat, cpu_now] and "
                         "report their detection PR (levels/guard unchanged)")
    ap.add_argument("--dump-preds", default=None,
                    help="CSV path for per-row test predictions "
                         "(timestamp, msname, actual, predicted, last)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    delta = args.delta

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    JVM_COLS = ["jvm_heap", "jvm_nonheap", "jvm_gc", "jvm_threads"]
    if args.aux_csv:
        aux = pd.read_csv(args.aux_csv, parse_dates=["timestamp"])
        aux = aux[["timestamp", "msname"] + [c for c in aux.columns
                                             if c in JVM_COLS + ["rps_sharp"]]]
        df = df.merge(aux, on=["msname", "timestamp"], how="left")
        df = df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
        for svc, idx in df.groupby("msname").groups.items():
            ii = np.asarray(list(idx))
            ntr = int(len(ii) * 0.70)
            for c in JVM_COLS:
                if c not in df.columns:
                    continue
                v = df.loc[ii, c].to_numpy(float)
                mu = np.nanmean(v[:ntr])
                sd = np.nanstd(v[:ntr])
                if not np.isfinite(mu):
                    mu = 0.0
                if not np.isfinite(sd) or sd < 1e-12:
                    sd = 1.0
                df.loc[ii, c] = np.nan_to_num((v - mu) / sd, nan=0.0,
                                              posinf=0.0, neginf=0.0)
    if args.s2_in == "jvm" and not all(
            c in df.columns for c in JVM_COLS):
        raise SystemExit("--s2-in jvm needs --aux-csv with jvm_* columns")
    if args.s1_rps_col != "rps_total" and args.s1_rps_col not in df.columns:
        raise SystemExit(f"--s1-rps-col {args.s1_rps_col} needs --aux-csv")
    P = {"jump_pred": [], "truth": [], "last": []}
    META = {"timestamp": [], "msname": []}
    CFtr, CFte, SLtr, DLtr = [], [], [], []
    rps_mae, rps_n = 0.0, 0

    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 3000:
            continue
        rps_col = (args.s1_rps_col if args.s1_rps_col in g.columns
                   else "rps_total")
        rps = g[rps_col].ffill().bfill().to_numpy(float)
        cpu = g["cpu_utilization"].ffill().bfill().to_numpy(float)
        tod_s = g["tod_sin"].to_numpy(float)
        tod_c = g["tod_cos"].to_numpy(float)
        lags = LAGS_SHORT if args.s1_lags == "short" else LAGS_FULL
        parts = [np.roll(rps, l) for l in lags] + [tod_s, tod_c]
        if args.s1_feats == "wide":
            for c in ("caller_sum", "mesh_rps", "frontend_rps"):
                if c in g.columns:
                    v = g[c].ffill().bfill().to_numpy(float)
                    parts += [np.roll(v, l) for l in (0, 3, 6, 12)]
        F1 = np.stack(parts, axis=1)
        ok = np.isfinite(F1).all(axis=1)
        ok[:max(lags) + 1] = False
        cut_tr = int(n * 0.70)
        cut_te = int(n * 0.80)
        te_idx = np.nonzero(ok & (np.arange(n) >= cut_te)
                            & (np.arange(n) + H < n))[0]
        tr_idx = np.nonzero(ok & (np.arange(n) < cut_tr)
                            & (np.arange(n) + H < n))[0]
        if len(te_idx) < 200 or len(tr_idx) < 800:
            continue
        if args.target == "peak":
            YTR = np.stack([cpu[tr_idx + h] for h in range(1, H + 1)]).max(axis=0)
            YTE = np.stack([cpu[te_idx + h] for h in range(1, H + 1)]).max(axis=0)
        else:
            YTR = cpu[tr_idx + H]
            YTE = cpu[te_idx + H]
        s1 = make_stage1(args.s1, args.seed)
        s1.fit(F1[tr_idx], rps[tr_idx + H])
        rps_hat_te = s1.predict(F1[te_idx])
        rps_mae += float(np.abs(rps_hat_te - rps[te_idx + H]).sum())
        rps_n += len(te_idx)
        r_now_tr, r_fut_tr = rps[tr_idx], rps[tr_idx + H]
        M2 = np.stack([r_fut_tr, r_fut_tr - r_now_tr], axis=1)
        if args.s2_in in ("full", "jvm"):
            M2 = np.stack([r_fut_tr, r_fut_tr - r_now_tr, cpu[tr_idx]], axis=1)
        if args.s2_in == "jvm":
            J = np.stack([g[c].ffill().bfill().fillna(0.0).to_numpy(float)[tr_idx]
                          for c in JVM_COLS], axis=1)
            M2 = np.concatenate([M2, J], axis=1)
        s2 = make_stage2(args.s2, args.seed)
        if args.s2 == "hurdle":
            from sklearn.ensemble import (HistGradientBoostingClassifier,
                                          HistGradientBoostingRegressor)
            sl = (YTR - cpu[tr_idx] >= delta).astype(int)
            dl = (YTR - cpu[tr_idx] <= -delta).astype(int)
            gs = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                min_samples_leaf=100, class_weight="balanced",
                random_state=args.seed).fit(M2, sl)
            gd = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                min_samples_leaf=100, class_weight="balanced",
                random_state=args.seed).fit(M2, dl)
            ms = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=20, random_state=args.seed)
            md = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=20, random_state=args.seed)
            if sl.sum() >= 10:
                ms.fit(M2[sl == 1], YTR[sl == 1])
            if dl.sum() >= 10:
                md.fit(M2[dl == 1], YTR[dl == 1])
            r_now_te = rps[te_idx]
            Mte = np.stack([rps_hat_te, rps_hat_te - r_now_te], axis=1)
            if args.s2_in in ("full", "jvm"):
                Mte = np.stack([rps_hat_te, rps_hat_te - r_now_te,
                                cpu[te_idx]], axis=1)
            if args.s2_in == "jvm":
                Jte = np.stack([g[c].ffill().bfill().fillna(0.0).to_numpy(float)[te_idx]
                                for c in JVM_COLS], axis=1)
                Mte = np.concatenate([Mte, Jte], axis=1)
            PGS = (gs.predict_proba(Mte)[:, 1] if sl.sum() >= 10
                   else np.zeros(len(Mte)))
            PGD = (gd.predict_proba(Mte)[:, 1] if dl.sum() >= 10
                   else np.zeros(len(Mte)))
            MS = ms.predict(Mte) if sl.sum() >= 10 else cpu[te_idx]
            MD = md.predict(Mte) if dl.sum() >= 10 else cpu[te_idx]
            w0 = np.clip(1.0 - PGS - PGD, 0.0, 1.0)
            cpu_hat = PGS * MS + PGD * MD + w0 * cpu[te_idx]
            P.setdefault("gate_spike", []).append(PGS)
            P.setdefault("gate_drop", []).append(PGD)
            P["jump_pred"].append(cpu_hat - cpu[te_idx])
            P["truth"].append(YTE)
            P["last"].append(cpu[te_idx])
            META["timestamp"].append(g["timestamp"].iloc[te_idx + H].to_numpy())
            META["msname"].append(np.full(len(te_idx), svc))
            continue
        if args.s2_weight == "jump":
            sw = 1.0 + 5.0 * np.abs(YTR - cpu[tr_idx])
            s2.fit(M2, YTR, sample_weight=sw)
        else:
            s2.fit(M2, YTR)
        r_now_te = rps[te_idx]
        Mte = np.stack([rps_hat_te, rps_hat_te - r_now_te], axis=1)
        if args.s2_in in ("full", "jvm"):
            Mte = np.stack([rps_hat_te, rps_hat_te - r_now_te,
                            cpu[te_idx]], axis=1)
        if args.s2_in == "jvm":
            Jte = np.stack([g[c].ffill().bfill().fillna(0.0).to_numpy(float)[te_idx]
                            for c in JVM_COLS], axis=1)
            Mte = np.concatenate([Mte, Jte], axis=1)
        cpu_hat = s2.predict(Mte)
        P["jump_pred"].append(cpu_hat - cpu[te_idx])
        P["truth"].append(YTE)
        P["last"].append(cpu[te_idx])
        META["timestamp"].append(g["timestamp"].iloc[te_idx + H].to_numpy())
        META["msname"].append(np.full(len(te_idx), svc))
        if args.s2_clf:
            r_hat_tr = s1.predict(F1[tr_idx])
            CFtr.append(np.stack([r_hat_tr, r_hat_tr - rps[tr_idx],
                                  cpu[tr_idx]], axis=1))
            SLtr.append((YTR - cpu[tr_idx] >= delta).astype(int))
            DLtr.append((YTR - cpu[tr_idx] <= -delta).astype(int))
            CFte.append(np.stack([rps_hat_te, rps_hat_te - r_now_te,
                                  cpu[te_idx]], axis=1))

    J = np.concatenate(P["jump_pred"])
    Y = np.concatenate(P["truth"])
    L = np.concatenate(P["last"])
    log(f"test rows: {len(Y)}  stage1 rps MAE: {rps_mae / max(1, rps_n):.4f}")
    jump_true = Y - L
    spike = (jump_true >= delta).astype(int)
    drop = (jump_true <= -delta).astype(int)
    res = {"s1": args.s1, "s2": args.s2, "s1_feats": args.s1_feats,
           "s2_in": args.s2_in, "s2_weight": args.s2_weight,
           "target": args.target, "s1_lags": args.s1_lags,
           "s1_rps_col": getattr(args, "s1_rps_col", "rps_total"),
           "n": len(Y), "delta": delta,
           "horizon": H, "stage1_rps_mae": round(rps_mae / max(1, rps_n), 4)}
    for name, lab, score in (("spike", spike, J), ("drop", drop, -J)):
        m = lab == 1
        ap, p50 = pr_scores(lab, score)
        mae_m = float(np.abs((L + J)[m] - Y[m]).mean()) if m.sum() else None
        mae_p = float(np.abs(L[m] - Y[m]).mean()) if m.sum() else None
        res[name] = {"n_events": int(lab.sum()),
                     "base_rate": round(float(lab.mean()), 5),
                     "pr_auc": round(ap, 4), "prec_at_rec50": p50,
                     "mae_model": round(mae_m, 5) if mae_m is not None else None,
                     "mae_persist": round(mae_p, 5) if mae_p is not None else None,
                     "mae_ratio": round(mae_m / (mae_p + 1e-12), 4)
                     if mae_m is not None else None}
    mae_m = float(np.abs((L + J) - Y).mean())
    mae_p = float(np.abs(L - Y).mean())
    res["guard"] = {"mae_ratio": round(mae_m / (mae_p + 1e-12), 4),
                    "bias": round(float(((L + J) - Y).mean()), 5),
                    "pass": bool(mae_m <= mae_p)}
    cheat = L + 1.0
    res["cheater"] = {
        "spike_recall": round(float(((cheat - L) >= delta)[spike == 1].mean()), 4),
        "spike_precision": round(float(spike[(cheat - L) >= delta].mean()), 4),
        "global_mae_ratio": round(float(np.abs(cheat - Y).mean()) / (mae_p + 1e-12), 2)}
    for t in ("spike", "drop"):
        r = res[t]
        log(f"{t:>6}: n={r['n_events']} base={r['base_rate']} "
            f"PR-AUC={r['pr_auc']} P@R50={r['prec_at_rec50']} "
            f"transMAE_ratio={r['mae_ratio']}")
    if "gate_spike" in P:
        GS = np.concatenate(P["gate_spike"])
        GD = np.concatenate(P["gate_drop"])
        for ename, probs in (("spike", GS), ("drop", GD)):
            lab = (jump_true >= delta).astype(int) if ename == "spike" else (
                jump_true <= -delta).astype(int)
            ap, p50 = pr_scores(lab, probs)
            res[f"gate_{ename}"] = {"n_events": int(lab.sum()),
                                    "pr_auc": round(ap, 4),
                                    "prec_at_rec50": p50}
            log(f"gate_{ename}: n={int(lab.sum())} PR-AUC={ap:.4f} "
                f"P@R50={p50} (hurdle gates)")
    if args.s2_clf and CFtr:
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtr = np.concatenate(CFtr)
        Xte = np.concatenate(CFte)
        for ename, Ltr in (("spike", np.concatenate(SLtr)),
                           ("drop", np.concatenate(DLtr))):
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                min_samples_leaf=100, class_weight="balanced",
                random_state=args.seed)
            clf.fit(Xtr, Ltr)
            pr = clf.predict_proba(Xte)[:, 1]
            lab = (jump_true >= delta).astype(int) if ename == "spike" else (
                jump_true <= -delta).astype(int)
            ap, p50 = pr_scores(lab, pr)
            res[f"event_{ename}"] = {"n_events": int(lab.sum()),
                                     "pr_auc": round(ap, 4),
                                     "prec_at_rec50": p50}
            log(f"event_{ename}: n={int(lab.sum())} PR-AUC={ap:.4f} "
                f"P@R50={p50} (balanced HGB on predicted workload)")
    log(f"GUARD {res['guard']}  CHEATER {res['cheater']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log(f"saved -> {args.out}")
    if args.dump_preds:
        pd.DataFrame({
            "timestamp": np.concatenate(META["timestamp"]),
            "msname": np.concatenate(META["msname"]),
            "actual": Y,
            "predicted": L + J,
            "last": L,
        }).sort_values(["msname", "timestamp"]).to_csv(args.dump_preds,
                                                       index=False)
        log(f"saved predictions -> {args.dump_preds}")


if __name__ == "__main__":
    main()


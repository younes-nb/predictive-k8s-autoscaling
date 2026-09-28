#!/usr/bin/env python3
"""What leads big transitions? Spike-triggered-average analysis.

For each large cpu jump (|jump over H steps| >= BIG), align all events at
onset t=0 and average each feature's z-scored trajectory over [-60, +10]
steps (spike-triggered average, STA). Features that ramp/dip BEFORE t=0
are leaders; coincident movers peak at 0; laggards after.

Also per feature: quiet-then-cross hit rate, median lead, and alarm
precision (crossing -> P(event within 12 steps)).

No models, no labels to fit — pure measurement. Answers "what leads"
before any new model is built.

Run from repo root:
    python analytics/transition_leaders.py --csv <10s-tier0> \\
        --out-dir analytics/data/leaders
"""

import argparse
import os

import numpy as np
import pandas as pd

H = 6
PRE, POST = 60, 10
BIG_CANDIDATES = [0.3, 0.5, 0.75]

CANDS = ["rps_total", "http_mcr", "providerrpc_mcr", "frontend_rps",
         "mesh_rps", "upstream_rps_sum", "caller_rps_max", "from_frontend",
         "n_callers", "active_in", "active_for", "queue_in", "queue_for",
         "throttle_ratio", "p99_latency", "req_byte_rate", "resp_byte_rate",
         "req_bytes_per_req", "resp_bytes_per_req", "err_frac", "flag_frac",
         "net_rx", "rps_slope5", "cpu_slope3", "mem_delta5", "rps_z30",
         "ewma_gap", "vol_rps", "vol_cpu", "concurrency", "scale_recency",
         "cpu_lim", "mem_lim", "pgfault", "pgmajfault", "restart_rate",
         "replicas", "desired_replicas", "unavailable",
         "neigh_cpu_mean", "neigh_rps_z30_mean", "tod_sin"]


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/leaders")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    feats = [c for c in CANDS if c in df.columns]

    # event scan at several thresholds (pooled app services)
    app = df[~df.msname.str.contains("mongo|mysql")
             & (df.msname != "ts-voucher-service")].reset_index(drop=True)
    counts = {}
    for big in BIG_CANDIDATES:
        n = 0
        for _, g in app.groupby("msname"):
            cpu = g["cpu_utilization"].to_numpy(float)
            n += int((((cpu[H:] - cpu[:-H]) >= big).sum()))
        counts[big] = n
        log(f"spike events at |jump|>={big}: {n}")
    big = max([b for b, n in counts.items() if n >= 200],
              default=max(BIG_CANDIDATES))
    log(f"selected BIG={big}")

    # NOTE: STA is descriptive statistics (no model is fit), so events come
    # from the FULL window — no train/test split needed, no leak possible.

    # collect aligned windows per event (spike + drop separately)
    STA, META = {"spike": {}, "drop": {}}, []
    for svc, g in app.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 1000:
            continue
        cpu = g["cpu_utilization"].to_numpy(float)
        F = g[feats].ffill(limit=12).bfill().to_numpy(float)
        mu = np.nanmean(F, axis=0)
        sd = np.nanstd(F, axis=0) + 1e-12
        Z = (F - mu) / sd
        for direction, cond in (("spike", cpu[H:] - cpu[:-H] >= big),
                                ("drop", cpu[H:] - cpu[:-H] <= -big)):
            onsets = np.nonzero(cond)[0]
            # refractory: keep onsets 30+ apart, margins for the window
            keep = []
            last = -10 ** 9
            for o in onsets:
                if o - last > 30 and o - PRE >= 0 and o + POST < n:
                    keep.append(o)
                    last = o
            for o in keep:
                STA[direction].setdefault("zs", []).append(Z[o - PRE:o + POST])
                META.append((direction, svc, o))
    for direction in STA:
        STA[direction]["zs"] = np.stack(STA[direction]["zs"])
    n_sp = len([m for m in META if m[0] == "spike"])
    n_dr = len([m for m in META if m[0] == "drop"])
    log(f"events (full window): spikes={n_sp} drops={n_dr}")
    if n_sp < 10 or n_dr < 10:
        raise SystemExit("too few events for STA; lower BIG")

    rows = []
    for direction in ("spike", "drop"):
        Z = STA[direction]["zs"]  # (E, PRE+POST, F)
        E = Z.shape[0]
        if E < 10:
            continue
        for j, name in enumerate(feats):
            curve = np.nanmean(Z[:, :, j], axis=0)
            se = np.nanstd(Z[:, :, j], axis=0) / np.sqrt(max(1, E))
            pre = curve[:PRE]
            # leader score: max |mean| in [-30, -2] vs null band
            lead_win = np.abs(pre[-30:-2])
            lead_score = float(lead_win.max())
            lead_at = int(np.argmax(np.abs(pre[-30:-2])) - 30)  # rel to onset
            # quiet-then-cross: fraction with |z|>2 first crossing in [-24,-1]
            # after quiet [-48,-24]
            hits, leads = 0, []
            for e in range(E):
                z = Z[e, :, j]
                if np.nanmax(np.abs(z[PRE - 48:PRE - 24])) > 1.0:
                    continue
                over = np.nonzero(np.abs(z[PRE - 24:PRE]) > 2.0)[0]
                if len(over):
                    hits += 1
                    leads.append(24 - over[0])
            rows.append(dict(direction=direction, feature=name,
                             n_events=E,
                             lead_score=round(lead_score, 3),
                             lead_at_steps=lead_at,
                             hit_rate=round(hits / E, 3),
                             median_lead=round(float(np.median(leads)), 1)
                             if leads else None,
                             at_onset=round(float(curve[PRE]), 3)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out_dir}/leaders.csv", index=False)

    # alarm precision: crossings -> event within 12 (pooled, per feature)
    prec_rows = []
    for direction in ("spike", "drop"):
        for j, name in enumerate(feats):
            alarms, hits = 0, 0
            for svc, g in app.groupby("msname"):
                g = g.reset_index(drop=True)
                n = len(g)
                if n < 1000:
                    continue
                x = pd.Series(g[name].to_numpy(float)).ffill().bfill()
                xz = ((x - x.mean()) / (x.std() + 1e-12)).to_numpy()
                cpu = g["cpu_utilization"].to_numpy(float)
                te = np.arange(int(n * 0.80), n - 12)
                last = -10 ** 9
                for t in te:
                    if abs(xz[t]) > 2.0 and t - last > 30:
                        alarms += 1
                        last = t
                        f = cpu[t + 1:t + 13] - cpu[t]
                        if ((f >= big).any() if direction == "spike"
                                else (f <= -big).any()):
                            hits += 1
            prec_rows.append(dict(direction=direction, feature=name,
                                  alarms=alarms,
                                  precision=round(hits / max(1, alarms), 3)))
    pd.DataFrame(prec_rows).to_csv(f"{args.out_dir}/alarm_precision.csv",
                                   index=False)
    for direction in ("spike", "drop"):
        r = res[res.direction == direction].sort_values("lead_score",
                                                        ascending=False)
        log(f"--- {direction} leaders (by pre-onset STA excursion) ---")
        for _, row in r.head(12).iterrows():
            log(f"  {row['feature']:22} lead_score={row['lead_score']:.2f} "
                f"at={row['lead_at_steps']:>4} hit={row['hit_rate']:.2f} "
                f"med_lead={row['median_lead']} onset={row['at_onset']:.2f}")
    log(f"wrote -> {args.out_dir}")

    # STA plot data + figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for direction in ("spike", "drop"):
        Z = STA[direction]["zs"]
        E = Z.shape[0]
        if E < 10:
            continue
        r = res[res.direction == direction].sort_values("lead_score",
                                                        ascending=False)
        top = r.head(8)["feature"].tolist()
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        t = np.arange(-PRE, POST) * 10
        for ax, name in zip(axes.flatten(), top):
            j = feats.index(name)
            curve = np.nanmean(Z[:, :, j], axis=0)
            se = np.nanstd(Z[:, :, j], axis=0) / np.sqrt(E)
            ax.plot(t, curve, "b-", lw=1.5)
            ax.fill_between(t, curve - 2 * se, curve + 2 * se, alpha=0.2)
            ax.axvline(0, color="r", ls="--", lw=1)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{name} (n={E})", fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("seconds to onset")
        fig.suptitle(f"Spike-triggered average — {direction} events "
                     f"(|jump|>={big}, H={H})")
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/sta_{direction}.png", dpi=120)
    log("saved STA plots")


if __name__ == "__main__":
    main()

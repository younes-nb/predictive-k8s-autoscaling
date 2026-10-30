#!/usr/bin/env python3
"""Same-day 4h A/B rerun analyzer: HPA Run A vs PBScaler-v1.2.5 Run B.

Produces (in --outdir):
  - ab4h_comprehensive_report.txt  (full report: viol rates, avg response time,
    measured module timing for the four PBScaler modules incl. GA, scale
    decisions per service, clamps, replica tables)
  - module_timing_table.csv        (per-module count/mean/med/p90/p95/max/sum)
  - replica_table_{runA,runB}.csv  (per-service replica distribution, 5s-weighted)
  - avg_response_{runA,runB}.csv   (front + per-service avg response time)
  - offered_series_{runA,runB}.csv / delivered_series_{runA,runB}.csv
  - loadshape_{runA,runB}.png      (per-window min-max x7600 + delivered only)
  - replica_plot_{runA,runB}.png   (top hot services A-vs-B overlay)

Run A's front_avg must be pre-joined: pass --runA-avgcsv (output of
reconstruct_front_avg.py) or the analyzer falls back to NaN for A avg columns.
"""
import argparse
import csv
import datetime as dt
import glob
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

MODULE_LABELS = [
    ('slo_detection', 'SLO Violation Detection'),
    ('redundancy_check', 'Redundancy Checking'),
    ('bottleneck_analysis', 'Performance Bottleneck Analysis'),
    ('decision_maker_ga', 'Decision Maker (GA)'),
    ('action_commit', 'Action Commit (k8s write, aux)'),
]
TIMING_RE = re.compile(r'\[timing\] (\S+) module=(\S+) seconds=([0-9.eE+-]+)')
SCALE_RE = re.compile(r'^scale (\S+) to (\d+)$')
CLAMP_RE = re.compile(r'clamped out-of-band action for (\S+): (-?\d+) -> (-?\d+) \(band \[(-?\d+),(-?\d+)\]\)')
ANOM_RE = re.compile(r'\[anomaly\] SLO-violating calls: (\d+)')
K6_RE = re.compile(r'running \((\d+)h(\d+)m([\d.]+)s\), (\d+)/560 VUs, ([\d,]+) complete and (\d+) interrupted')
SNAP_TS_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def parse_timing_log(path, t_lo=None, t_hi=None):
    """-> per-module list of (iso_ts, seconds) and cycle annotations.

    t_lo/t_hi (ISO strings): restrict to the collector window. Timing lines carry
    their own UTC ts; scale/clamp/anomaly lines come from a stream without
    --timestamps, so they inherit the ts of the most recent preceding [timing]
    line (chronological file => exact). Lines before the first in-window timing
    line (idle pre-warmup cycles) are excluded.
    """
    mods = defaultdict(list)
    anomalies = []
    scales = []          # (ts, svc, n)
    clamps = []          # (ts, svc, act, clamped, lo, hi)
    noopscale_out = noopscale_in = 0
    cur_ts = None
    for line in open(path, errors='ignore'):
        line = line.rstrip()
        m = TIMING_RE.search(line)
        if m:
            cur_ts = m.group(1)
            if t_lo and cur_ts < t_lo:
                continue
            if t_hi and cur_ts > t_hi:
                continue
            mods[m.group(2)].append((cur_ts, float(m.group(3))))
            continue
        if t_lo and (cur_ts is None or cur_ts < t_lo):
            continue
        if t_hi and cur_ts is not None and cur_ts > t_hi:
            continue
        m = ANOM_RE.search(line)
        if m:
            anomalies.append(int(m.group(1)))
            continue
        m = CLAMP_RE.search(line)
        if m:
            clamps.append((cur_ts, m.group(1), int(m.group(2)), int(m.group(3)),
                           int(m.group(4)), int(m.group(5))))
            continue
        m = SCALE_RE.match(line)
        if m:
            scales.append((cur_ts, m.group(1), int(m.group(2))))
            continue
        if 'begin scale out' in line:
            noopscale_out += 1
        elif 'begin scale in' in line:
            noopscale_in += 1
    return mods, anomalies, scales, clamps, noopscale_out, noopscale_in


def snap_log_to_df(path):
    """Replica snapshot log -> wide DataFrame indexed by ts (spec.replicas)."""
    blocks = []
    cur_ts, cur = None, {}
    for line in open(path, errors='ignore'):
        line = line.strip()
        if not line:
            continue
        if SNAP_TS_RE.match(line):
            if cur_ts is not None:
                blocks.append((cur_ts, cur))
            cur_ts, cur = line, {}
        else:
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                cur[parts[0]] = int(parts[1])
    if cur_ts is not None:
        blocks.append((cur_ts, cur))
    df = pd.DataFrame({t: d for t, d in blocks}).T.fillna(np.nan).astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def delivered_series(k6_path):
    """k6 completed-iterations/s at 1s cadence -> pandas Series (index min, req/s)."""
    el, comp = [], []
    for line in open(k6_path, errors='ignore'):
        m = K6_RE.search(line)
        if m:
            h, mi, s, _, cc, _it = m.groups()
            el.append(int(h) * 3600 + int(mi) * 60 + float(s))
            comp.append(int(cc.replace(',', '')))
    if len(el) < 2:
        return pd.Series(dtype=float)
    el, comp = np.array(el), np.array(comp)
    keep = np.diff(el) > 0
    S = pd.Series(np.diff(comp)[keep] / np.diff(el)[keep], index=el[1:][keep] / 60.0)
    return S


def offered_from_trace(mcr_csv, start_min, n_min=255, compress=8):
    tr = pd.read_csv(mcr_csv)['http_mcr'].values
    MCR = np.round(np.clip(tr / tr.max(), 0, None) * 7600).astype(int)
    idx = ((start_min + np.arange(n_min) * compress) % len(tr)).astype(int)
    return pd.Series(MCR[idx] * compress / 60.0)  # req/s per wall-minute


def dataset_stats(csv_path, avgcsv=None):
    df = pd.read_csv(csv_path)
    df['ts'] = pd.to_datetime(df['timestamp'])
    lab = df['slo_reward'].isin([0, 1])
    n, viol, ok = len(df), int((df['slo_reward'] == 0).sum()), int((df['slo_reward'] == 1).sum())
    out = {
        'rows': n, 'viol': viol, 'ok': ok, 'unlabeled': n - viol - ok,
        'viol_rate': viol / max(viol + ok, 1),
        'fp90_med': df.loc[lab, 'front_p90'].median(),
        'fp90_p95': df.loc[lab, 'front_p90'].quantile(0.95),
        'fp90_max': df.loc[lab, 'front_p90'].max(),
        'df': df,
    }
    if 'front_avg' in df.columns:
        fa = df.loc[lab, 'front_avg']
        out['favg_med'] = fa.median()
        out['favg_mean'] = fa.mean()
        out['favg_p95'] = fa.quantile(0.95)
        out['favg_max'] = fa.max()
        out['favg_avail'] = df['front_avg'].notna().mean()
    if avgcsv:  # join reconstructed Run-A front_avg
        rec = pd.read_csv(avgcsv)
        rec['ts'] = pd.to_datetime(rec['timestamp'])
        df = df.merge(rec[['ts', 'front_avg_ms']].rename(columns={'ts': 'ts'}), on='ts', how='left')
        fa = df.dropna(subset=['front_avg_ms'])
        falm = df.loc[lab].dropna(subset=['front_avg_ms'])['front_avg_ms'] if len(fa) else pd.Series(dtype=float)
        out['favg_med'] = falm.median()
        out['favg_mean'] = falm.mean()
        out['favg_p95'] = falm.quantile(0.95)
        out['favg_max'] = falm.max()
        out['favg_repr_n'] = int(df['front_avg_ms'].notna().sum())
        out['df'] = df
    return out


def replica_table_coll(df, svcs):
    """Per-service 5s-weighted replica (pod-count) distribution from &count cols."""
    rows = []
    for s in svcs:
        col = f'{s}&count'
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(v) == 0:
            continue
        rows.append(dict(svc=s, min=v.min(), med=v.median(), mean=v.mean(),
                         p95=v.quantile(0.95), max=v.max(),
                         frac_at_max=(v == v.max()).mean(),
                         n=len(v)))
    return pd.DataFrame(rows).set_index('svc')


def module_timing_table(mods):
    rows = []
    for key, label in MODULE_LABELS:
        vals = [s for _, s in mods.get(key, [])]
        if not vals:
            rows.append(dict(module=label, key=key, count=0))
            continue
        a = np.array(vals)
        rows.append(dict(module=label, key=key, count=len(vals), sum=a.sum(),
                         mean=a.mean(), med=float(np.median(a)),
                         p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                         max=a.max()))
    return pd.DataFrame(rows)


def fmt(x, nd=1):
    return f'{x:.{nd}f}' if x == x else 'NA'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runA', required=True)
    ap.add_argument('--runB', required=True)
    ap.add_argument('--runB-ctrllog', default='/tmp/opencode/runB_pbscaler_controller_raw.log')
    ap.add_argument('--runA-k6', default='/tmp/opencode/runA_hpa_jul14_4h_k6.log')
    ap.add_argument('--runB-k6', default='/tmp/opencode/runB_pbscaler_jul14_4h_k6.log')
    ap.add_argument('--runA-avgcsv', default=None, help='reconstructed Run-A front_avg CSV')
    ap.add_argument('--mcr', default='/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv')
    ap.add_argument('--outdir', default='/proj/k8sautoscaledl-PG0/logs/pbscaler/ab4h_rerun')
    ap.add_argument('--start-min', type=int, default=1932)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    R = []  # report lines

    def say(line=''):
        R.append(line)
        print(line)

    csvA = glob.glob(os.path.join(args.runA, 'train_data.csv'))[0]
    csvB = glob.glob(os.path.join(args.runB, 'train_data.csv'))[0]
    A = dataset_stats(csvA, args.runA_avgcsv)
    B = dataset_stats(csvB)
    dfA, dfB = A['df'], B['df']
    svcs = [c[:-6] for c in dfA.columns if c.endswith('&count')]

    # per-service avg response time from each run's metrics.csv (<svc>&avg)
    def svc_avg_table(dsdir):
        mp = glob.glob(os.path.join(dsdir, 'metrics.csv'))
        if not mp:
            return pd.DataFrame()
        m = pd.read_csv(mp[0], low_memory=False)
        avgcols = [c for c in m.columns if c.endswith('&avg')]
        if not avgcols:
            return pd.DataFrame()
        out = {}
        for c in avgcols:
            v = pd.to_numeric(m[c], errors='coerce').dropna()
            if len(v):
                out[c[:-4]] = dict(med=v.median(), mean=v.mean(), p95=v.quantile(.95), max=v.max())
        return pd.DataFrame(out).T

    svcAvgA, svcAvgB = svc_avg_table(args.runA), svc_avg_table(args.runB)
    for tag, df in [('runA', dfA), ('runB', dfB)]:
        col = 'front_avg' if 'front_avg' in df.columns else 'front_avg_ms'
        sub = df[['timestamp', 'front_p90'] + ([col] if col in df.columns else [])]
        sub.to_csv(os.path.join(args.outdir, f'avg_response_{tag}.csv'), index=False)

    # ---------- violations ----------
    say('=' * 88)
    say('SAME-DAY 4h A/B RERUN — Run A (HPA-only) vs Run B (PBScaler v1.2.5, instrumented)')
    say('Phase: START_MIN=1932, TRACE_COMPRESS=8, MAX_REQUESTS=7600, USER_POOL=560, 15m warmup + 4h collect')
    say('=' * 88)
    say('')
    say('## 1. SLO violation rates')
    say(f"{'metric':26s} {'Run A (HPA)':>14s} {'Run B (PBScaler)':>18s} {'delta (PB−HPA)':>16s}")
    vrA, vrB = A['viol_rate'], B['viol_rate']
    say(f"{'violations (ticks)':26s} {A['viol']:>14d} {B['viol']:>18d} {B['viol']-A['viol']:>+16d}")
    say(f"{'labeled-ok ticks':26s} {A['ok']:>14d} {B['ok']:>18d}")
    say(f"{'unlabeled (-1) ticks':26s} {A['unlabeled']:>14d} {B['unlabeled']:>18d}")
    say(f"{'violation rate (%)':26s} {100*vrA:>14.2f} {100*vrB:>18.2f} {100*(vrB-vrA):>+16.2f}")
    say(f"{'relative change (%)':26s} {'':>14s} {'':>18s} {100*(vrB-vrA)/max(vrA,1e-9):>+16.1f}")
    say('')
    say('## 2. Front response latency (ms) — label basis: istio gateway->front P90, completion semantics')
    say(f"{'metric':26s} {'Run A (HPA)':>14s} {'Run B (PBScaler)':>18s}")
    for k, fmtk in [('fp90_med', 'front P90 med'), ('fp90_p95', 'front P90 p95'), ('fp90_max', 'front P90 max')]:
        say(f"{fmtk:26s} {fmt(A[k],1):>14s} {fmt(B[k],1):>18s}")
    if 'favg_med' in A and 'favg_med' in B:
        say(f"{'front AVG med':26s} {fmt(A['favg_med'],1):>14s} {fmt(B['favg_med'],1):>18s}")
        say(f"{'front AVG mean':26s} {fmt(A['favg_mean'],1):>14s} {fmt(B['favg_mean'],1):>18s}")
        say(f"{'front AVG p95':26s} {fmt(A['favg_p95'],1):>14s} {fmt(B['favg_p95'],1):>18s}")
        say(f"{'front AVG max':26s} {fmt(A['favg_max'],1):>14s} {fmt(B['favg_max'],1):>18s}")
        if 'favg_repr_n' in A:
            say(f"(Run A front AVG reconstructed from Mimir range queries, n={A.get('favg_repr_n','?')} pts; "
                f"Run B recorded in-flight by collector; front_avg col availability B={100*B.get('favg_avail',0):.1f}%)")
    say('')

    say('## 2b. Per-service avg response time (ms; med over the run, from each run\'s metrics.csv &avg)')
    if len(svcAvgA) and len(svcAvgB):
        j = svcAvgA.join(svcAvgB, how='inner', lsuffix='_A', rsuffix='_B')
        j = j.sort_values('med_B', ascending=False)
        say(f"   {'service':30s} {'A med':>8s} {'A p95':>9s} {'B med':>8s} {'B p95':>9s}")
        for s, r in j.head(20).iterrows():
            say(f"   {s:30s} {fmt(r['med_A'],1):>8s} {fmt(r['p95_A'],1):>9s} {fmt(r['med_B'],1):>8s} {fmt(r['p95_B'],1):>9s}")
        say(f"   ({len(svcAvgA)} svcs in A, {len(svcAvgB)} in B; top-20 by B med shown; full via avg_response_*.csv front cols)")
    else:
        say('   (metrics.csv &avg columns unavailable for at least one run)')
    say('')

    # ---------- module timing ----------
    # restrict controller-log parsing to the collector window (excludes the
    # idle pre-warmup cycles that precede the measured regime)
    t_lo = dfB['ts'].min().strftime('%FT%TZ')
    t_hi = dfB['ts'].max().strftime('%FT%TZ')
    mods, anomalies, scales, clamps, noopscale_out, noopscale_in = parse_timing_log(
        args.runB_ctrllog, t_lo, t_hi)
    all_lines = sum(len(v) for v in mods.values())
    mt = module_timing_table(mods)
    mt.to_csv(os.path.join(args.outdir, 'module_timing_table.csv'), index=False)
    say('## 3. Measured PBScaler module time costs (Run B, real recorded values — NOT paper numbers)')
    say('   Source: [timing] lines streamed from the in-cluster v1.2.5 controller, time.perf_counter().')
    say(f'   Window: collector ticks only ({t_lo} .. {t_hi}); {all_lines} timing samples.')
    say(f"   {'Module (user naming)':38s} {'count':>7s} {'mean_s':>9s} {'med_s':>9s} {'p90_s':>9s} {'p95_s':>9s} {'max_s':>9s} {'sum_s':>9s}")
    for _, r in mt.iterrows():
        if r['count'] == 0:
            say(f"   {r['module']:38s} {0:>7d} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s}")
            continue
        say(f"   {r['module']:38s} {r['count']:>7d} {fmt(r['mean'],3):>9s} {fmt(r['med'],3):>9s} "
            f"{fmt(r['p90'],3):>9s} {fmt(r['p95'],3):>9s} {fmt(r['max'],3):>9s} {fmt(r['sum'],1):>9s}")
    cyc = len(anomalies) + noopscale_out + noopscale_in
    say(f"   anomaly cycles={len(anomalies)} (15s window), no-action cycles: add={noopscale_out} reduce={noopscale_in}")
    say('')

    # ---------- scale decisions ----------
    say('## 4. Scale decisions (Run B controller)')
    per_svc = defaultdict(lambda: defaultdict(int))
    for _, svc, _n in scales:
        per_svc[svc]['writes'] += 1
    for _, svc, _a, _c, _lo, _hi in clamps:
        per_svc[svc]['clamps'] += 1
    say(f"   total spec-replica writes: {len(scales)}; clamped out-of-band GA outputs: {len(clamps)}")
    ncommit = len(mods.get('action_commit', []))
    if ncommit:
        say(f"   writes per GA commit cycle: {len(scales)/ncommit:.1f}  (n_commits={ncommit}) "
            f"— echo-patch storm: execute_task re-writes the whole action map each cycle")
    say(f"   cycles with >=1 SLO-violating call: {sum(1 for a in anomalies if a > 0)}/{len(anomalies)} "
        f"(anomaly cycle = 15 s); waste cycles with no action: add={noopscale_out} reduce={noopscale_in}")
    say(f"   {'service':26s} {'writes':>7s} {'clamps':>7s}")
    for svc in sorted(per_svc, key=lambda s: -per_svc[s]['writes']):
        say(f"   {svc:26s} {per_svc[svc]['writes']:>7d} {per_svc[svc]['clamps']:>7d}")
    say('')

    # ---------- replicas ----------
    tabA = replica_table_coll(dfA, svcs)
    tabB = replica_table_coll(dfB, svcs)
    tabA.to_csv(os.path.join(args.outdir, 'replica_table_runA.csv'))
    tabB.to_csv(os.path.join(args.outdir, 'replica_table_runB.csv'))
    say('## 5. Replicas per service (5s-weighted; collector &count = pod count)')
    say(f"{'service':26s} {'A med':>6s} {'A max':>6s} {'B med':>6s} {'B max':>6s} {'dMed':>6s}")
    for s in tabA.index:
        a, b = tabA.loc[s], tabB.loc[s] if s in tabB.index else None
        dm = '-' if b is None else fmt(b['med'] - a['med'], 0)
        bs = '-' if b is None else f"{b['med']:>6.0f} {b['max']:>6.0f}"
        say(f"{s:26s} {a['med']:>6.0f} {a['max']:>6.0f} {bs} {dm:>6s}")
    both = tabA.join(tabB, how='inner', lsuffix='_A', rsuffix='_B')
    say(f"   mean replicas over all svcs: A={both['mean_A'].mean():.2f} B={both['mean_B'].mean():.2f}; "
        f"time-avg total: A={dfA[[s+'&count' for s in svcs if s+'&count' in dfA.columns]].sum(axis=1).mean():.0f} "
        f"B={dfB[[s+'&count' for s in svcs if s+'&count' in dfB.columns]].sum(axis=1).mean():.0f}")
    say('')

    # ---------- load shape + plots ----------
    off = offered_from_trace(args.mcr, args.start_min)
    offN = (off - off.min()) / (off.max() - off.min()) * 7600
    delA, delB = delivered_series(args.runA_k6), delivered_series(args.runB_k6)
    for tag, d, s in [('runA', dfA, delA), ('runB', dfB, delB)]:
        d['wm'] = 30.0 + (d['ts'] - d['ts'].min()).dt.total_seconds() / 60.0
        d.to_csv(os.path.join(args.outdir, f'dataset_{tag}_window.csv'), index=False)
        s.to_csv(os.path.join(args.outdir, f'delivered_series_{tag}.csv'))
    off.to_csv(os.path.join(args.outdir, 'offered_series.csv'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tag, D, dl, lab in [('runA', dfA, delA, 'Run A (HPA-only)'), ('runB', dfB, delB, 'Run B (PBScaler v1.2.5)')]:
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax[0].plot(off.index, offN, 'k-', lw=1.4)
        ax[0].set_ylabel('offered (window min−max → peak 7600)')
        ax[0].set_title(f'{lab} — load shape: offered + delivered')
        ax[1].plot(dl.index, dl.values, 'b-', lw=0.9, label='delivered (k6 iter/s)')
        ax[1].set_ylabel('delivered (k6 iter/s)'); ax[1].set_xlabel('run minute (incl. 15m warmup)')
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f'loadshape_{tag}.png'), dpi=130)
        plt.close(fig)
    # combined
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(off.index, offN, 'k-', lw=1.3, label='offered (window min−max → peak 7600)')
    ax[0].set_ylabel('offered'); ax[0].legend(); ax[0].set_title('Same-day A/B: offered identical; delivered compared')
    ax[1].plot(delA.index, delA.values, 'b-', lw=0.9, label='delivered A (HPA)')
    ax[1].plot(delB.index, delB.values, 'r-', lw=0.9, label='delivered B (PBScaler)')
    ax[1].set_ylabel('delivered (k6 iter/s)'); ax[1].legend(); ax[1].set_xlabel('run minute')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'loadshape_AB_compare.png'), dpi=130)
    plt.close(fig)

    # replica overlay plots
    hot = tabA.sort_values('mean', ascending=False).index.tolist()
    hot = [s for s in hot if f'{s}&count' in dfA.columns and f'{s}&count' in dfB.columns][:12]
    fig, axes = plt.subplots(min(6, len(hot)), 2, figsize=(14, 3 * min(6, len(hot))), sharex=False)
    axes = np.atleast_2d(axes)
    for i, s in enumerate(hot[:6]):
        for j, (D, lab, col) in enumerate([('A', 'HPA', 'b'), ('B', 'PBScaler', 'r')]):
            dfx = dfA if D == 'A' else dfB
            axes[i, j].plot(dfx['wm'], dfx[f'{s}&count'], col, lw=0.5)
            axes[i, j].set_title(f'{s} — {lab}')
            axes[i, j].set_ylabel('replicas')
            axes[i, j].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'replica_plot_AB.png'), dpi=130)
    plt.close(fig)

    say('## 6. Load-shape & artifacts')
    say(f"   offered window min−max: [{off.min():.0f},{off.max():.0f}] req/s (raw) → normalized peak=7600")
    say(f"   delivered peaks: A={delA.max():.0f} B={delB.max():.0f} iter/s; medians A={delA.median():.0f} B={delB.median():.0f}")
    say(f"   outputs → {args.outdir}")

    with open(os.path.join(args.outdir, 'ab4h_comprehensive_report.txt'), 'w') as f:
        f.write('\n'.join(R) + '\n')

    # summary
    with open(os.path.join(args.outdir, 'ab4h_summary.txt'), 'w') as f:
        f.write(f"A/B: viol rate {100*vrA:.2f}% -> {100*vrB:.2f}% ({100*(vrB-vrA):+.2f} pts)\n")
        f.write(f"front P90 med {A['fp90_med']:.1f} -> {B['fp90_med']:.1f} ms; "
                f"front AVG med {fmt(A.get('favg_med', float('nan')), 1)} -> {fmt(B.get('favg_med', float('nan')), 1)} ms\n")
        ga = mt[mt['key'] == 'decision_maker_ga'].iloc[0] if len(mt) else None
        if ga is not None and ga['count']:
            f.write(f"GA(measured): n={ga['count']} mean={ga['mean']:.3f}s med={ga['med']:.3f}s p95={ga['p95']:.3f}s\n")
        clamps_n = len(clamps)
        f.write(f"spec writes={len(scales)} clamps={clamps_n}\n")


if __name__ == '__main__':
    main()

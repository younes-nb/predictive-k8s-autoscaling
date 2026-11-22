"""Harvest 5 s telemetry from Mimir into a training corpus.

Windows are discovered from the orchestrator's own ``params.json`` files, so the
corpus covers exactly the regimes the benchmark has exercised (plus idle periods
for healthy examples). Nothing is hand-listed.

Output: parquet with one row per (service, 5 s tick) and one column per raw
feature. Missing samples are NaN, never zero.
"""

import argparse
import glob
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                        # noqa: E402
from features.groups import queries                              # noqa: E402
from telemetry.mimir import MimirClient, MimirError              # noqa: E402

DEFAULT_RUNS_GLOB = "../stage7_28/exp13jul/*/params.json"


def discover_windows(pattern):
    """(start, end, label) from params.json launch_epoch + k6_run_s."""
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        run = os.path.basename(os.path.dirname(p))
        # launch epoch lives in a sibling file, not in params.json
        lf = os.path.join(os.path.dirname(p), "launch_epoch")
        if not os.path.exists(lf):
            continue
        try:
            s = int(open(lf).read().strip())
        except ValueError:
            continue
        dur = int(d.get("k6_run_s") or (d.get("duration_min") or 60) * 60)
        out.append((s, s + dur, run))
    return out


def harvest(cfg, windows, step=None):
    step = step or cfg.telemetry_step_s
    mimir = MimirClient(cfg)
    services = mimir.services()
    print(f"[harvest] {len(services)} services, {len(windows)} windows")

    frames = []
    for start, end, label in windows:
        if end - start < step * 2:
            continue
        try:
            fetched = mimir.fetch(services, start, end, step=step)
        except MimirError as exc:
            print(f"[harvest] window {label} failed: {exc}")
            continue
        for svc, feats in fetched.items():
            if not feats:
                continue
            series = {}
            for k, pts in feats.items():
                for ts, v in pts:
                    series.setdefault(ts, {})[k] = v
            if not series:
                continue
            for ts in sorted(series):
                row = {"service": svc, "ts": ts, "window": label}
                row.update(series[ts])
                frames.append(row)
        print(f"[harvest] {label}: {len(fetched)} services")
    if not frames:
        raise SystemExit("no telemetry harvested")
    df = pd.DataFrame.from_records(frames)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/corpus.parquet")
    ap.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--max-window-s", type=int, default=3600,
                    help="skip windows longer than this (keeps the corpus bounded)")
    a = ap.parse_args()

    cfg = Config()
    windows = [w for w in discover_windows(a.runs_glob)
               if w[1] - w[0] <= a.max_window_s]
    if a.max_windows:
        windows = windows[:a.max_windows]
    print(f"[harvest] {len(windows)} windows: " +
          ", ".join(f"{w[0]}-{w[1]}" for w in windows[:6]) + " ...")

    df = harvest(cfg, windows, a.step)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[harvest] wrote {len(df)} rows x {len(df.columns)} cols -> {a.out}")
    print(f"[harvest] services={df['service'].nunique()} "
          f"span={int(df['ts'].min())}..{int(df['ts'].max())}")


if __name__ == "__main__":
    main()
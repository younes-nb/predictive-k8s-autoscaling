#!/usr/bin/env python3
"""Generate a predictable per-minute http_mcr workload from Wikipedia pageviews.

Downloads the lightweight per-project aggregate files
(`projectviews-YYYYMMDD-HH0000`, ~22 KB each — NOT the ~50 MB per-article
pageviews dumps) from https://dumps.wikimedia.org/other/pageviews/ into
DATA_DIR (cached on disk), sums the English-Wikipedia all-access traffic
(domain codes `en` + `en.m`) into per-1-hour buckets, upsamples to per-1-minute
resolution, normalizes to [0,1] (peak = 1.0), and writes a CSV with columns
msname,timestamp,http_mcr — the exact format consumed by
load_testing/run_test.sh / locustfile.py.

The default window (Mon 2024-04-08 + 14 days = two full Mon-Sun weeks) shows a
genuine diurnal + weekend pattern: night trough vs. midday peak each day with
lower weekend traffic.

Predictability check: prints the autocorrelation of the emitted curve at lag 60
(1 hour) and lag 1440 (1 day), the peak |autocorr| over lags 1..2880, and the
mean day-over-day Pearson correlation of hourly profiles.

Notes on the source data:
  * One projectviews file per UTC hour; the hour in the filename is the END of
    the aggregation period (consistent with pagecounts-raw), so
    projectviews-20240101-010000 covers 00:00-01:00 UTC and its counts are
    attributed to minute timestamps starting at 00:00 UTC.
  * Line format is `<domain_code> - <count_views> 0`, e.g. `en - 2592893 0`
    (desktop) and `en.m - 9375445 0` (mobile). `en` + `en.m` cover >99% of
    English-Wikipedia traffic (the residual `en.d`/`en.m.d` zero-rated and
    sister-project `en.b`/`en.q`/… lines are excluded).
  * Raw granularity is hourly, so per-minute values are upsampled: `linear`
    (default) linearly interpolates between consecutive hourly totals for a
    smooth curve; `flat` repeats the hour's normalized total across its 60
    minutes (hourly steps).

Reference:
  Wikimedia Foundation Analytics, "Pageviews" dumps,
  https://dumps.wikimedia.org/other/pageviews/ (data since May 2015, CC0).
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

MIN_PER_HOUR = 60
MIN_PER_DAY = 1440

BASE_URL = "https://dumps.wikimedia.org/other/pageviews"
# Contact-style User-Agent per the Wikimedia Foundation User-Agent policy
# (https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).
USER_AGENT = "predictive-k8s-autoscaling-loadtest/1.0 (research workload generator)"

# English-Wikipedia all-access (desktop + mobile, >99% of en.wikipedia traffic).
DOMAIN_CODES = ("en", "en.m")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def hour_file_names(start_date: str, days: int) -> list[tuple[str, str]]:
    """Return [(utc_hour_start, file_name)] for `days` days from `start_date`.

    start_date is YYYYMMDD (a UTC calendar day); the first hour starts at 00:00
    UTC that day. The filename hour is the END of the period, i.e. one hour
    after the period start.
    """
    try:
        day0 = datetime.strptime(start_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        sys.exit(f"--start_date must be YYYYMMDD, got {start_date!r}")
    out = []
    for h in range(days * 24):
        period_start = day0 + timedelta(hours=h)
        period_end = period_start + timedelta(hours=1)
        fname = period_end.strftime("projectviews-%Y%m%d-%H0000")
        out.append((period_start, fname))
    return out


def download(hours: list[tuple[str, str]], data_dir: str) -> list[str]:
    """Download (cached) the projectviews file for each hour. Returns paths
    (None for hours that failed after retries)."""
    os.makedirs(data_dir, exist_ok=True)
    paths: list[str] = []
    missing = 0
    for period_start, fname in hours:
        ym = period_start.strftime("%Y/%Y-%m")
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            paths.append(path)
            continue
        url = f"{BASE_URL}/{ym}/{fname}"
        ok = False
        for attempt in range(1, 4):
            if os.system(f'curl -sSL --fail -A "{USER_AGENT}" '
                         f'-o "{path}" "{url}"') == 0 \
                    and os.path.getsize(path) > 0:
                ok = True
                break
            log(f"  retry {attempt}/3 for {fname}")
            time.sleep(2 * attempt)
        if not ok:
            log(f"WARNING: failed to download {url}")
            missing += 1
            paths.append(None)
        else:
            paths.append(path)
        # Be polite to the dumps host (rate limited, max 3 conns/IP).
        time.sleep(0.2)
    if missing:
        log(f"Downloaded {len(paths) - missing}/{len(paths)} hourly files "
            f"({missing} missing)")
    else:
        log(f"Downloaded all {len(paths)} hourly files (cached in {data_dir})")
    return paths


def count_hour(path: str) -> int:
    """Sum en.wikipedia all-access views in one projectviews file."""
    total = 0
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] in DOMAIN_CODES:
                try:
                    total += int(parts[2])
                except ValueError:
                    continue
    return total


def fill_gaps(hourly: list) -> list[int]:
    """Fill None (missing-file) hours by linear interpolation; exit if the
    window is unusable."""
    n = len(hourly)
    known = [i for i, v in enumerate(hourly) if v is not None]
    if not known:
        sys.exit("No usable hourly data (all downloads failed)")
    if len(known) < 0.9 * n:
        sys.exit(f"Too many missing hours: {n - len(known)}/{n}")
    filled = list(hourly)
    for i in range(n):
        if filled[i] is not None:
            continue
        prev = max([k for k in known if k < i], default=None)
        nxt = min([k for k in known if k > i], default=None)
        if prev is None:
            filled[i] = filled[nxt]
        elif nxt is None:
            filled[i] = filled[prev]
        else:
            frac = (i - prev) / (nxt - prev)
            filled[i] = int(round(filled[prev] + frac * (filled[nxt] - filled[prev])))
    if any(v is None for v in hourly):
        log(f"Filled {sum(1 for v in hourly if v is None)} missing hour(s) "
            f"by interpolation")
    if max(filled) <= 0:
        sys.exit("Empty workload window (peak hourly view count is 0)")
    return filled


def upsample(hourly: list[int], mode: str) -> list[float]:
    """Expand hourly totals to per-minute values (still in raw view counts)."""
    minutes: list[float] = []
    n = len(hourly)
    for h, value in enumerate(hourly):
        if mode == "flat" or h == n - 1:
            minutes.extend([float(value)] * MIN_PER_HOUR)
        else:  # linear: blend from this hour's total to the next
            nxt = hourly[h + 1]
            for k in range(MIN_PER_HOUR):
                minutes.append(value + (nxt - value) * (k / MIN_PER_HOUR))
    return minutes


def predictability_report(mcr) -> None:
    """Autocorrelation + day-over-day alignment of the emitted curve."""
    try:
        import numpy as np
    except ImportError:
        log("numpy unavailable; skipping autocorrelation stats")
        return
    s = np.asarray(mcr, dtype=float)
    n = len(s)
    if n < 2:
        log("Curve too short for autocorrelation stats")
        return
    centered = s - s.mean()
    var = float(np.dot(centered, centered)) / n
    if var < 1e-12:
        log("Curve is (near-)constant; no autocorrelation to report")
        return
    max_lag = min(2880, n - 1)
    full = np.correlate(centered, centered, mode="full")
    acf = full[n - 1:] / ((n - np.arange(n)) * var)
    acf = acf[:max_lag + 1]
    peak_idx = int(np.argmax(np.abs(acf[1:]))) + 1
    log(f"Predictability check (n={n} min, peak normalized to 1.0):")
    log(f"  autocorr lag 60   (1 h)  : {acf[60]:.3f}" if 60 <= max_lag else "  autocorr lag 60: n/a")
    log(f"  autocorr lag 1440 (1 d)  : {acf[1440]:.3f}" if 1440 <= max_lag else "  autocorr lag 1440: n/a")
    log(f"  peak |autocorr| over lags 1..{max_lag}: {abs(acf[peak_idx]):.3f} at lag {peak_idx}")
    if n >= 2 * MIN_PER_DAY:
        days = n // MIN_PER_DAY
        hour = s[: days * MIN_PER_DAY].reshape(days, MIN_PER_DAY)
        aligned = hour[: days - 1]
        nxt = hour[1:]
        cors = []
        for a, b in zip(aligned, nxt):
            a = a - a.mean()
            b = b - b.mean()
            denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
            if denom > 1e-12:
                cors.append(float(np.dot(a, b)) / denom)
        if cors:
            log(f"  day-over-day Pearson (hourly profiles): {np.mean(cors):.3f} +/- {np.std(cors):.3f} over {len(cors)} pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="/proj/k8sautoscaledl-PG0/wikipedia",
                    help="where the raw projectviews files live / are downloaded")
    ap.add_argument("--out_dir", default=None,
                    help="where to write the CSV (default: data_dir)")
    ap.add_argument("--start_date", default="20240408",
                    help="first UTC day to include, YYYYMMDD "
                         "(default: 20240408, a Monday — two full Mon-Sun weeks)")
    ap.add_argument("--days", type=int, default=7,
                    help="number of UTC days to include (default: 7)")
    ap.add_argument("--upsample", choices=("linear", "flat"), default="linear",
                    help="hourly -> per-minute method (default: linear)")
    args = ap.parse_args()

    if args.days < 1:
        sys.exit("--days must be >= 1")

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    hours = hour_file_names(args.start_date, args.days)
    log(f"Window: {args.days} day(s) from {args.start_date} UTC "
        f"({len(hours)} hourly files)")
    paths = download(hours, data_dir)

    hourly = [count_hour(p) if p is not None else None for p in paths]
    n_ok = sum(1 for v in hourly if v is not None)
    log(f"Parsed {n_ok}/{len(hourly)} hourly files "
        f"(en.wikipedia desktop+mobile views/hour)")
    hourly = fill_gaps(hourly)
    peak_hour = max(hourly)
    log(f"Hourly views: peak {peak_hour}/h, trough {min(hourly)}/h "
        f"(trough/peak {min(hourly) / peak_hour:.3f})")

    minutes = upsample(hourly, args.upsample)
    peak = max(minutes)
    mcr = [v / peak for v in minutes]

    # Minute timestamps in ms epoch UTC (period starts are UTC-aware).
    t0_ms = int(hours[0][0].timestamp() * 1000)
    timestamps = [t0_ms + i * 60000 for i in range(len(mcr))]

    out_name = f"http_mcr_WIKI_{args.start_date}_{args.days}d_{args.upsample}.csv"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", newline="") as f:
        f.write("msname,timestamp,http_mcr\n")
        for t, m in zip(timestamps, mcr):
            f.write(f"WIKI,{t},{m:.6f}\n")
    log(f"Wrote {len(mcr)} minutes -> {out_path}")

    predictability_report(mcr)


if __name__ == "__main__":
    main()

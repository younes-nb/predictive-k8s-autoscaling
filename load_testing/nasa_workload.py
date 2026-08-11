#!/usr/bin/env python3
"""Generate a predictable per-minute http_mcr workload from the NASA-HTTP trace.

Downloads NASA_access_log_{Jul95,Aug95}.gz into DATA_DIR (cached on disk),
parses the Apache common-log lines (one per request), aggregates requests into
per-1-minute buckets, fills empty minutes with 0, normalizes counts to [0,1]
(peak = 1.0), and writes a CSV with columns msname,timestamp,http_mcr — the
exact format consumed by load_testing/run_test.sh / locustfile.py.

The July trace is the cleaner choice: ~28 contiguous days of diurnal traffic.
August contains a multi-hour zero-traffic gap (Hurricane Erin, 01-03 Aug 1995).

Predictability check: prints the autocorrelation of the emitted curve at lag 60
(1 hour) and lag 1440 (1 day), the peak |autocorr| over lags 1..2880, and the
mean day-over-day Pearson correlation of hourly profiles.

Reference:
  M. Arlitt and C. Williamson, "Web Server Workload Characterization: The
  Search for Invariants", ACM SIGMETRICS 1996. Distributed via the Internet
  Traffic Archive (https://ita.ee.lbl.gov/traces/NASA-HTTP.html).
"""

import argparse
import gzip
import os
import re
import sys
from datetime import datetime

LINE_RE = re.compile(r"\[(\d{2})/([A-Za-z]{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})")

MONTHS = {
    "jul": ("Jul", 7, 31),
    "aug": ("Aug", 8, 31),
}

URLS = {
    "jul": [
        "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
        "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
        "https://raw.githubusercontent.com/PanCat26/time-decayed-caching/main/data/NASA_access_log_Jul95.gz",
    ],
    "aug": [
        "https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
        "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
        "https://raw.githubusercontent.com/jmcnabb1/NASA_ACCESS_LOG_AUG95/main/NASA_access_log_Aug95.gz",
    ],
}

MIN_PER_DAY = 1440


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def download(month: str, data_dir: str) -> str:
    """Download the trace for `month` into data_dir (cached). Returns path."""
    path = os.path.join(data_dir, f"NASA_access_log_{month.title()}95.gz")
    if os.path.exists(path):
        try:
            with gzip.open(path, "rb") as f:
                f.read(1)
            log(f"Using cached trace: {path}")
            return path
        except OSError:
            log(f"Cached trace invalid, re-downloading: {path}")
    for url in URLS[month]:
        log(f"Downloading {url}")
        if os.system(f'curl -sSL --fail -o "{path}" "{url}"') == 0:
            try:
                with gzip.open(path, "rb") as f:
                    f.read(1)
                log(f"Saved trace: {path}")
                return path
            except OSError:
                log(f"Bad download from {url}, trying next source")
    sys.exit(f"Failed to download NASA {month}95 trace into {data_dir}")


def count_requests(path: str) -> dict:
    """Return {minute_since_month_start: request_count} for one trace file."""
    counts = {}
    n_parsed = 0
    n_skipped = 0
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m is None:
                n_skipped += 1
                continue
            day, mon, _year, hh, mm, _ss = m.groups()
            minute = (int(day) - 1) * MIN_PER_DAY + int(hh) * 60 + int(mm)
            counts[minute] = counts.get(minute, 0) + 1
            n_parsed += 1
    log(f"Parsed {n_parsed} requests ({n_skipped} skipped malformed lines)")
    return counts


def last_active_day(counts: dict) -> int:
    """Last day-of-month (1-indexed) that has any recorded request."""
    return max(counts) // MIN_PER_DAY + 1


def build_curve(counts: dict, start_day: int, end_day: int, days_in_month: int):
    """Return (timestamps_ms, mcr_list) over the [start_day, end_day] window.

    Missing minutes (no requests) are filled with 0 so the curve is contiguous.
    Normalization is min-max over the selected window (peak = 1.0).
    """
    start_min = (start_day - 1) * MIN_PER_DAY
    end_min = end_day * MIN_PER_DAY - 1
    n_minutes = end_min - start_min + 1
    series = [counts.get(start_min + i, 0) for i in range(n_minutes)]
    peak = max(series)
    if peak <= 0:
        sys.exit("Empty workload window (peak request count is 0)")
    mcr = [c / peak for c in series]
    timestamps = [start_min * 60000 + i * 60000 for i in range(n_minutes)]
    return timestamps, mcr, peak


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="/proj/k8sautoscaledl-PG0/nasa",
                    help="where the raw .gz traces live / are downloaded")
    ap.add_argument("--out_dir", default=None,
                    help="where to write the CSV (default: data_dir)")
    ap.add_argument("--month", choices=sorted(MONTHS), default="jul",
                    help="trace month to use (default: jul — contiguous diurnal "
                         "traffic; aug has the Hurricane Erin gap)")
    ap.add_argument("--start_day", type=int, default=1,
                    help="first day-of-month to include (1-indexed)")
    ap.add_argument("--end_day", type=int, default=None,
                    help="last day-of-month to include (default: last day with "
                         "any request — the July trace ends Jul 28)")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    months = [args.month]
    ts_offset = 0
    all_timestamps = []
    all_mcr = []
    all_peaks = []
    for month in months:
        path = download(month, data_dir)
        abbr, month_num, days_in_month = MONTHS[month]
        counts = count_requests(path)
        end_day = args.end_day or last_active_day(counts)
        if args.start_day < 1 or end_day > days_in_month:
            sys.exit(f"--start_day/--end_day out of range for {month} "
                     f"(1..{days_in_month})")
        ts, mcr, peak = build_curve(counts, args.start_day, end_day, days_in_month)
        ts = [t + ts_offset for t in ts]
        ts_offset += days_in_month * MIN_PER_DAY * 60000
        all_timestamps.extend(ts)
        all_mcr.extend(mcr)
        all_peaks.append(peak)
        log(f"{month}95: {len(mcr)} minutes over days {args.start_day}-{end_day}, "
            f"peak {peak} req/min")

    out_name = f"http_mcr_NASA_{args.month}95.csv"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", newline="") as f:
        f.write("msname,timestamp,http_mcr\n")
        for t, mcr in zip(all_timestamps, all_mcr):
            f.write(f"NASA,{t},{mcr:.6f}\n")
    log(f"Wrote {len(all_mcr)} minutes -> {out_path}")

    predictability_report(all_mcr)


if __name__ == "__main__":
    main()

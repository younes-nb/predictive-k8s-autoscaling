#!/usr/bin/env python3
"""Select the highest-standard-deviation slice of a NASA-HTTP day.

Given a day (e.g. 1995-07-13) and a duration in minutes, compute the
per-minute request counts for that day and slide a window of that duration
across it, scoring each window by the standard deviation of its per-minute
counts. The window with the highest std is the "spikiest" window of that
length — the one that exercises autoscaling hardest for that duration.

Output is a slice CSV in the schema the k6 generator consumes:
    msname,timestamp,http_mcr
where http_mcr = count / peak(count within the window), matching
load_testing/nasa_workload.py.

Usage:
  nasa_trace_slice.py --day 1995-07-13 --duration 240 --out slice.csv
  nasa_trace_slice.py --day 1995-07-13 --duration 240 --top 5      # alternatives
"""
import argparse
import calendar
import collections
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
LINE_RE = re.compile(
    r'\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2}) ([+-]\d{4})\]')
DEFAULT_CACHE = "/proj/k8sautoscaledl-PG0/reports/stage7_28/nasa_cache"
SOURCES = {
    "jul": ("NASA_access_log_Jul95.gz",
            "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"),
    "aug": ("NASA_access_log_Aug95.gz",
            "https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz"),
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_source(source, cache):
    """Return the path to the (downloaded and cached) raw log."""
    name, url = SOURCES[source]
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, name)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path, False
    log("downloading %s -> %s" % (url, path))
    tmp = path + ".part"
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", tmp, url], check=True)
    os.replace(tmp, path)
    return path, True


def parse_day(path, day):
    """Per-minute request counts for one UTC-normalised day.

    The trace is local time -0400; the day boundary is taken in that same local
    time (so 'the 13th' means the 13th as recorded), then converted to UTC for
    the emitted timestamps.
    """
    y, m, d = (int(x) for x in day.split("-"))
    mon = [k for k, v in MONTHS.items() if v == m][0]
    needle = "%s/%s/%s" % (d, mon, y)
    counts = collections.Counter()
    matched = 0
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if needle not in line:
                continue
            mm = LINE_RE.search(line)
            if not mm:
                continue
            hh, mi = int(mm.group(4)), int(mm.group(5))
            counts[hh * 60 + mi] += 1
            matched += 1
    if not counts:
        raise SystemExit("no requests found for %s in %s" % (day, path))
    return counts, matched


def local_minutes_to_epoch(y, m, d, minute_of_day, tz_offset_h):
    """Local minute-of-day -> UTC epoch seconds."""
    base = calendar.timegm((y, m, d, 0, 0, 0, 0, 0, 0))
    return base + minute_of_day * 60 - tz_offset_h * 3600


def rolling_std(series, window):
    """Yield (start_index, std) for every window of `window` samples."""
    n = len(series)
    if n < window:
        return
    s = sum(series[:window])
    sq = sum(v * v for v in series[:window])
    for i in range(window, n + 1):
        mean = s / window
        var = sq / window - mean * mean
        yield i - window, math.sqrt(var) if var > 0 else 0.0
        if i < n:
            s += series[i] - series[i - window]
            sq += series[i] ** 2 - series[i - window] ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="YYYY-MM-DD, e.g. 1995-07-13")
    ap.add_argument("--duration", type=int, required=True,
                    help="window length in minutes")
    ap.add_argument("--out", required=True, help="output slice CSV")
    ap.add_argument("--source", default="jul", choices=sorted(SOURCES))
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--top", type=int, default=0,
                    help="also print the N best-scoring windows")
    a = ap.parse_args()

    if a.duration < 1:
        raise SystemExit("--duration must be >= 1 minute")
    if a.duration > 1440:
        raise SystemExit("--duration must be <= 1440 minutes (one day)")

    path, downloaded = ensure_source(a.source, a.cache)
    y, m, d = (int(x) for x in a.day.split("-"))
    mon = [k for k, v in MONTHS.items() if v == m][0]
    tz = -4  # the NASA-HTTP traces are logged at -0400

    t0 = time.time()
    counts, matched = parse_day(path, a.day)
    log("parsed %d requests for %s in %.1fs" % (matched, a.day, time.time() - t0))

    series = [counts.get(i, 0) for i in range(1440)]
    best_std, best_i = -1.0, None
    scored = []
    for i, sd in rolling_std(series, a.duration):
        scored.append((sd, i))
        if sd > best_std:
            best_std, best_i = sd, i
    if best_i is None:
        raise SystemExit("day %s has %d minutes < duration %d"
                         % (a.day, len(series), a.duration))

    window = series[best_i:best_i + a.duration]
    peak = max(window) or 1
    start_epoch = local_minutes_to_epoch(y, m, d, best_i, tz)
    end_epoch = local_minutes_to_epoch(y, m, d, best_i + a.duration, tz)

    with open(a.out, "w") as fh:
        fh.write("msname,timestamp,http_mcr\n")
        for k, c in enumerate(window):
            fh.write("NASA,%d,%.6f\n" % (start_epoch + k * 60, c / peak))

    summary = {
        "day": a.day,
        "source": os.path.basename(path),
        "source_sha256": sha256(path),
        "source_bytes": os.path.getsize(path),
        "requests_on_day": matched,
        "duration_min": a.duration,
        "window_start_minute_of_day": best_i,
        "window_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(start_epoch)),
        "window_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(end_epoch)),
        "window_std": round(best_std, 4),
        "window_peak_per_min": peak,
        "window_mean_per_min": round(sum(window) / a.duration, 3),
        "window_min_per_min": min(window),
        "day_peak_per_min": max(series),
        "output": os.path.abspath(a.out),
        "output_sha256": sha256(a.out),
        "output_rows": a.duration,
    }
    if a.top > 0:
        scored.sort(key=lambda x: -x[0])
        summary["top_windows"] = [
            {"start_minute_of_day": i, "std": round(sd, 3),
             "start_utc": time.strftime(
                 "%Y-%m-%dT%H:%M:%SZ",
                 time.gmtime(local_minutes_to_epoch(y, m, d, i, tz)))}
            for sd, i in scored[:a.top]]

    print(json.dumps(summary, indent=2))
    log("wrote %d rows -> %s (window %s..%s, std=%.1f, peak=%d/min)"
        % (a.duration, a.out, summary["window_start_utc"],
           summary["window_end_utc"], best_std, peak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
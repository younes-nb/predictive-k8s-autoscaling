#!/usr/bin/env python3
"""Analyse the JWT session-renewal behaviour of a k6 `trainticket.js` run.

Reads the k6 console log (the file the orchestrator captures as
`k6_console.log`, or the raw `> /tmp/k6_jwt_long.out` redirect) and reports:

  * renewals per VU and per session kind (user vs admin),
  * the per-minute renewal distribution (the herd test),
  * the per-VU renewal cycle lengths (jitter spread),
  * renewal scheduling vs. the token TTL,
  * every auth failure / token error line, so a silent regression is visible.

Renewal lines look like:
  RENEW-OK vu=2 kind=user exp_in=3600 renew_at=1794847927

`renew_at` is the epoch second at which the NEXT renewal becomes due
(`exp - safety - U(0, jitter)`), so consecutive renewals of one VU are one
~50-59 min cycle apart.

Usage:
    analyze_jwt_renewals.py <k6_console.log> [--json out.json] [--start ISO8601]

`--start` overrides the run start used for the T+ offsets; by default the
CONFIG line's own timestamp is used.
"""
import argparse
import collections
import json
import re
import sys
from datetime import datetime, timezone

# k6 log lines: time="2026-11-16T18:41:22+03:30" level=info msg="..." source=console
LINE_RE = re.compile(r'^time="([^"]+)"\s+level=(\w+)\s+msg="(.*)"\s+source=console')
RENEW_RE = re.compile(r"^RENEW-OK vu=(\d+) kind=(\w+) exp_in=(-?\d+) renew_at=(\d+)$")
AUTH_FAIL_RE = re.compile(r"^AUTH-FAIL vu=(\d+) kind=(\w+) user=(\S+)")
TOKEN_ERR_RE = re.compile(r"^TOKEN-ERR vu=(\d+) kind=(\w+) reason=(\S+)")


def parse_ts(s):
    """k6 writes RFC3339 with a numeric offset; normalise to epoch seconds."""
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", default=None, help="write the summary here")
    ap.add_argument("--start", default=None,
                    help="run start as ISO8601; defaults to the CONFIG line")
    a = ap.parse_args()

    renewals = []          # dicts: vu, kind, ts, exp_in, renew_at
    auth_fails = []
    token_errs = []
    run_start = parse_ts(a.start) if a.start else None
    first_ts = None

    with open(a.log, errors="replace") as fh:
        for raw in fh:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            ts, level, msg = parse_ts(m.group(1)), m.group(2), m.group(3)
            if ts is None:
                continue
            if first_ts is None:
                first_ts = ts
            if run_start is None and msg.startswith("CONFIG "):
                run_start = ts
            r = RENEW_RE.match(msg)
            if r:
                renewals.append({
                    "ts": ts,
                    "vu": int(r.group(1)),
                    "kind": r.group(2),
                    "exp_in": int(r.group(3)),
                    "renew_at": int(r.group(4)),
                    "t_plus_min": (round((ts - run_start) / 60, 2)
                                   if run_start else None),
                })
                continue
            r = AUTH_FAIL_RE.match(msg)
            if r:
                auth_fails.append({"ts": ts, "vu": int(r.group(1)),
                                   "kind": r.group(2), "user": r.group(3)})
                continue
            r = TOKEN_ERR_RE.match(msg)
            if r:
                token_errs.append({"ts": ts, "vu": int(r.group(1)),
                                   "kind": r.group(2), "reason": r.group(3)})

    # A VU owns TWO independent sessions: a ROLE_USER login and the shared
    # admin login. They renew on their own schedules, so every grouping and
    # every cycle-length computation has to key on (vu, kind), not on vu —
    # keying on vu alone makes the two sessions look like one 4-minute cycle.
    by_vu = collections.defaultdict(list)
    by_session = collections.defaultdict(list)
    for r in renewals:
        by_vu[r["vu"]].append(r)
        by_session[(r["vu"], r["kind"])].append(r)

    per_min = collections.Counter()
    for r in renewals:
        per_min[datetime.fromtimestamp(r["ts"], timezone.utc)
                .strftime("%Y-%m-%dT%H:%M")] += 1

    kinds = collections.Counter(r["kind"] for r in renewals)
    exp_ins = collections.Counter(r["exp_in"] for r in renewals)

    # Cycle length per VU: renew_at[n] - renew_at[n-1] is the scheduled gap
    # between consecutive cycles (the jittered TTL window).
    cycles = []
    for (vu, kind), rows in sorted(by_session.items()):
        rows.sort(key=lambda r: r["ts"])
        for prev, cur in zip(rows, rows[1:]):
            cycles.append({"vu": vu, "kind": kind,
                           "cycle_s": cur["renew_at"] - prev["renew_at"]})

    summary = {
        "log": a.log,
        "run_start_utc": (datetime.fromtimestamp(run_start, timezone.utc)
                          .isoformat() if run_start else None),
        "first_log_line_utc": (datetime.fromtimestamp(first_ts, timezone.utc)
                               .isoformat() if first_ts else None),
        "renewals_total": len(renewals),
        "renewals_by_kind": dict(kinds),
        "renewals_by_vu": {str(vu): len(rows) for vu, rows in sorted(by_vu.items())},
        "renewals_by_session": {"v%d/%s" % (vu, kind): len(rows)
                                for (vu, kind), rows in sorted(by_session.items())},
        "renewals_per_minute": dict(sorted(per_min.items())),
        "peak_renewals_in_one_minute": max(per_min.values()) if per_min else 0,
        "exp_in_seen": dict(exp_ins),
        "cycles_observed": len(cycles),
        "cycle_s_min": min((c["cycle_s"] for c in cycles), default=None),
        "cycle_s_max": max((c["cycle_s"] for c in cycles), default=None),
        "auth_failures": len(auth_fails),
        "auth_failure_lines": auth_fails[:20],
        "token_errors": len(token_errs),
        "token_error_lines": token_errs[:20],
        "renewals_detail": renewals,
        "cycles_detail": cycles,
    }

    print("run start (UTC)          : %s" % summary["run_start_utc"])
    print("renewals                 : %d  %s" % (summary["renewals_total"],
                                                 summary["renewals_by_kind"]))
    print("renewals per VU          : %s" % summary["renewals_by_vu"])
    print("renewals per session     : %s" % summary["renewals_by_session"])
    print("peak renewals in 1 min   : %d" % summary["peak_renewals_in_one_minute"])
    if cycles:
        print("scheduled cycle length   : %d-%d s (%.1f-%.1f min)"
              % (summary["cycle_s_min"], summary["cycle_s_max"],
                 summary["cycle_s_min"] / 60, summary["cycle_s_max"] / 60))
    print("exp_in seen              : %s" % summary["exp_in_seen"])
    print("auth failures            : %d" % summary["auth_failures"])
    print("token errors             : %d" % summary["token_errors"])
    if summary["renewals_per_minute"]:
        print("per-minute renewals      :")
        for minute, n in summary["renewals_per_minute"].items():
            print("   %s  %s" % (minute, "#" * n))
    if auth_fails:
        print("AUTH-FAIL lines:")
        for f in auth_fails[:10]:
            print("   %s" % f)
    if token_errs:
        print("TOKEN-ERR lines:")
        for e in token_errs[:10]:
            print("   %s" % e)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
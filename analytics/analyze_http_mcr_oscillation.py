import argparse
import json
import os
import sys
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths, PREPROCESSING

DEFAULT_SUBSET_SEED = 42
DEFAULT_WINDOW_HOURS = 6
DEFAULT_MIN_POINTS_RATIO = 0.5
DEFAULT_MAX_LAG = 180
DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                15, 20, 30, 45, 60, 90, 120, 150, 180)
SERVICE_INDEX_PATH = os.path.join(Paths.WINDOWS_DIR, "_service_index.json")
SERVICE_ARRAYS_PATH = os.path.join(Paths.WINDOWS_DIR, "_service_arrays.npy")
MS_PER_HOUR = 3_600_000
# 0.8 * 13 days: build_windows.py slices every service timeline at
# idx_val = int(n * (TRAIN_FRAC + VAL_FRAC)) and treats [idx_val, n) as the
# test split. Timestamps are 1-minute buckets starting at day 0, so the test
# split starts at TEST_START_MS (ms since day 0) for full-coverage services.
TEST_START_MS = int(0.8 * 13 * 24 * MS_PER_HOUR)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_service_names():
    """Load the list of microservice names used to build the windows cache."""
    idx_path = SERVICE_INDEX_PATH
    arrays_path = SERVICE_ARRAYS_PATH
    if not (os.path.exists(idx_path) and os.path.exists(arrays_path)):
        raise FileNotFoundError(
            f"Missing service cache: {idx_path} or {arrays_path}"
        )
    with open(idx_path, "r") as f:
        data = json.load(f)
    return list(data["index"].keys())


def load_service_split_sizes():
    """Return {msname: n_rows} where n_rows is the msresource feature-array
    length used by build_windows.py to derive the train/val/test split.

    build_windows.py (mirrored here):
        n = len(feat_raw)            # per-service msresource timeline (1/min)
        idx_tr = int(n * TRAIN_FRAC)
        idx_val = int(n * (TRAIN_FRAC + VAL_FRAC))
        test split = [idx_val, n)    # timestamps idx_val..n-1 minutes
    """
    with open(SERVICE_INDEX_PATH, "r") as f:
        data = json.load(f)
    return {name: entry[1] for name, entry in data["index"].items()}


def query_mcrtmcr_oscillations(con, mcr_dir, window_ms, in_clause,
                               split_frac, test_start_ms, expected_pts,
                               svc_n_df, restrict_test=True):
    """Scan every {window_ms}-long sliding segment of each service's http_mcr
    timeline (not just the last window) and return one row per
    (service, window end) with window stats.

    A segment is a frame of `expected_pts` consecutive 1-minute rows
    (ROWS BETWEEN expected_pts-1 PRECEDING AND CURRENT ROW), matching the
    existing window definition. oscillation is the std of the min-max
    normalized http_mcr within the segment, i.e. std_mcr / (g_max - g_min);
    win_start/win_end are the actual first/last timestamps of the segment.

    When restrict_test is True, only segments lying fully inside the model's
    TEST split (as defined by build_windows.py) are candidates: candidate rows
    are limited to timestamp >= test_start_ms, so every segment built from them
    lies inside the test split. Otherwise every segment of the full timeline is
    a candidate.

    Assumes all services listed in in_clause are present in mcr_dir."""
    con.register("svc_n", svc_n_df)
    cand_where = f" WHERE a.timestamp >= {test_start_ms}" if restrict_test else ""
    n_preceding = max(0, expected_pts - 1)
    sql = f"""
        WITH agg AS (
            SELECT msname, timestamp, SUM(http_mcr) AS http_mcr_sum
            FROM read_parquet('{mcr_dir}/*.parquet')
            WHERE msname IN ({in_clause}) AND http_mcr IS NOT NULL
            GROUP BY msname, timestamp
        ),
        maxes AS (
            SELECT msname, MAX(timestamp) AS max_ts
            FROM agg
            GROUP BY msname
            HAVING MAX(http_mcr_sum) > 0
        ),
        cand AS (
            SELECT a.msname, a.timestamp, a.http_mcr_sum
            FROM agg a
            JOIN maxes m ON a.msname = m.msname
            {cand_where}
        ),
        wstats AS (
            SELECT msname, timestamp, http_mcr_sum,
                   COUNT(http_mcr_sum) OVER w AS n_points,
                   SUM(CASE WHEN http_mcr_sum > 0 THEN 1 ELSE 0 END) OVER w AS n_nonzero,
                   AVG(http_mcr_sum) OVER w AS avg_mcr,
                   STDDEV(http_mcr_sum) OVER w AS std_mcr,
                   MIN(http_mcr_sum) OVER w AS g_min,
                   MAX(http_mcr_sum) OVER w AS g_max,
                   MIN(timestamp) OVER w AS win_start,
                   MAX(timestamp) OVER w AS win_end
            FROM cand
            WINDOW w AS (
                PARTITION BY msname ORDER BY timestamp
                ROWS BETWEEN {n_preceding} PRECEDING AND CURRENT ROW
            )
        ),
        final AS (
            SELECT w.msname,
                   w.std_mcr / NULLIF(w.g_max - w.g_min, 0) AS oscillation,
                   w.avg_mcr, w.std_mcr, w.win_start, w.win_end,
                   w.n_points, w.n_nonzero, w.g_min, w.g_max, m.max_ts,
                   n.n AS n_rows,
                   n.n - CAST(FLOOR({split_frac} * n.n) AS BIGINT) AS test_len
            FROM wstats w
            LEFT JOIN maxes m ON w.msname = m.msname
            LEFT JOIN svc_n n ON w.msname = n.msname
        )
        SELECT msname, oscillation, avg_mcr, std_mcr, win_start, win_end,
               n_points, n_nonzero, g_min, g_max, max_ts, test_len, n_rows
        FROM final
        ORDER BY oscillation DESC
    """
    df = con.execute(sql).df()
    return df


def query_mcr_series(con, mcr_dir, in_clause, test_start_ms, restrict_test=True):
    """Fetch the per-minute http_mcr series for exactly the same candidate rows
    used by query_mcrtmcr_oscillations (services with some non-zero load,
    optionally restricted to the TEST split), ordered by (msname, timestamp)."""
    cand_where = f" WHERE a.timestamp >= {test_start_ms}" if restrict_test else ""
    sql = f"""
        WITH agg AS (
            SELECT msname, timestamp, SUM(http_mcr) AS http_mcr_sum
            FROM read_parquet('{mcr_dir}/*.parquet')
            WHERE msname IN ({in_clause}) AND http_mcr IS NOT NULL
            GROUP BY msname, timestamp
        ),
        maxes AS (
            SELECT msname, MAX(timestamp) AS max_ts
            FROM agg
            GROUP BY msname
            HAVING MAX(http_mcr_sum) > 0
        )
        SELECT a.msname, a.timestamp, a.http_mcr_sum
        FROM agg a
        JOIN maxes m ON a.msname = m.msname
        {cand_where}
        ORDER BY a.msname, a.timestamp
    """
    return con.execute(sql).df()


def build_lags(max_lag, expected_pts):
    """Autocorrelation lags to test, capped by --max_lag and by the window size
    so a window of `expected_pts` minutes always leaves at least
    expected_pts // 2 aligned pairs (below that the correlation is unreliable)."""
    cap = min(max_lag, expected_pts // 2)
    return [L for L in DEFAULT_LAGS if L <= cap]


def compute_pattern_scores(series_df, expected_pts, lags):
    """Periodic-pattern score per (msname, window-end).

    pattern = max over the tested lags of |corr(x_t, x_{t-lag})| within the
    trailing `expected_pts`-point window ending at each minute. For lag L the
    correlation is taken over the W-L aligned pairs x_i vs x_{i-L} that lie
    inside the window. Samples are centered by their own rolling window mean
    before multiplying (numerically stable, keeps the score bounded by 1) and
    everything vectorizes over the whole timeline via rolling operations.
    Windows too short to carry a lag (or with zero variance) yield NaN."""
    df = series_df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    key = df["msname"]
    x = df["http_mcr_sum"]

    def roll_mean(s, win):
        return (s.groupby(key).rolling(win, min_periods=win)
                 .mean().reset_index(level=0, drop=True))

    best = np.full(len(df), np.nan)
    for L in lags:
        n_pairs = expected_pts - L
        if n_pairs < 2:
            continue
        # Center each sample by its own window mean before multiplying. This is
        # numerically stable for near-constant series (avoids catastrophic
        # cancellation when subtracting ~mean^2 from ~mean^2) at the cost of a
        # slightly approximate window centering -- fine for a ranking heuristic.
        cent = x - roll_mean(x, n_pairs)
        centL = cent.groupby(key).shift(L)
        num = roll_mean(cent * centL, n_pairs)
        denom = np.sqrt(np.maximum(
            (roll_mean(cent * cent, n_pairs)
             * roll_mean(centL * centL, n_pairs)).to_numpy(), 0.0))
        acf = num.to_numpy() / np.where(denom == 0.0, np.nan, denom)
        best = np.fmax(best, np.abs(acf))

    out = df[["msname", "timestamp"]].copy()
    out["pattern"] = best
    return out


def query_winner_mcr(con, mcr_dir, msname, win_start, win_end):
    """Query per-minute http_mcr for the winner window, min-max normalized to
    [0, 1] using the winner window's own min/max so the saved CSV always spans
    the full [0, 1] range (min value -> 0.0, max value -> 1.0)."""
    sql = f"""
        SELECT msname, timestamp, SUM(http_mcr) AS http_mcr_raw
        FROM read_parquet('{mcr_dir}/*.parquet')
        WHERE msname = '{msname}' AND timestamp >= {win_start}
          AND timestamp <= {win_end}
        GROUP BY msname, timestamp
        ORDER BY timestamp
    """
    df = con.execute(sql).df()
    win_min = df["http_mcr_raw"].min()
    win_max = df["http_mcr_raw"].max()
    span = win_max - win_min
    if span > 0:
        df["http_mcr"] = ((df["http_mcr_raw"] - win_min) / span).clip(0.0, 1.0)
    else:
        df["http_mcr"] = 0.5
    return df[["msname", "timestamp", "http_mcr"]]


def query_msresource_window(con, msresource_dir, msname, win_start, win_end):
    """Query CPU/memory utilization for a specific window of a service."""
    sql = f"""
        SELECT msname, timestamp, AVG(cpu_utilization) AS cpu,
               AVG(memory_utilization) AS mem
        FROM read_parquet('{msresource_dir}/*.parquet')
        WHERE msname = '{msname}' AND timestamp >= {win_start}
          AND timestamp <= {win_end}
        GROUP BY msname, timestamp
        ORDER BY timestamp
    """
    df = con.execute(sql).df()
    return df


def plot_timeseries(df_mcr: pd.DataFrame, df_res: pd.DataFrame,
                    service: str, out_dir: str, ts_str: str) -> None:
    """Plot http_mcr and CPU/memory for the winner window."""
    mcr_path = os.path.join(out_dir, f"http_mcr_{service}_{ts_str}.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_mcr["timestamp"], df_mcr["http_mcr"], color="#d62728",
            linewidth=0.5, label="http_mcr")
    ax.set_xlabel("timestamp (ms)")
    ax.set_ylabel("http_mcr (normalized)", color="#d62728")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"http_mcr oscillation window - {service}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(mcr_path, dpi=100)
    plt.close(fig)

    cpu_path = os.path.join(out_dir, f"cpu_{service}_{ts_str}.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_res["timestamp"], df_res["cpu"], color="#1f77b4",
            linewidth=0.5, label="cpu")
    ax.set_xlabel("timestamp (ms)")
    ax.set_ylabel("cpu utilization", color="#1f77b4")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"CPU utilization - {service}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(cpu_path, dpi=100)
    plt.close(fig)

    mem_path = os.path.join(out_dir, f"memory_{service}_{ts_str}.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_res["timestamp"], df_res["mem"], color="#2ca02c",
            linewidth=0.5, label="memory")
    ax.set_xlabel("timestamp (ms)")
    ax.set_ylabel("memory utilization", color="#2ca02c")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Memory utilization - {service}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(mem_path, dpi=100)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan every {--window_hours}-long sliding segment of each "
        "service's http_mcr timeline (not just the last window) -- optionally "
        "restricted to the model's TEST split as defined by build_windows.py "
        "(--split) -- ranked by --score (oscillation, avg mcr, periodic pattern, "
        "or combinations), and plot the winner's http_mcr plus CPU/memory "
        "utilization for that window."
    )
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Root containing msrtmcre/ and msresource/ subdirs. "
                             "Defaults to Paths.PARQUET_ROOT.")
    parser.add_argument("--max_services", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--out_dir", type=str, default=Paths.ANALYTICS_OUT_DIR)
    parser.add_argument("--window_hours", type=float, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--min_points_ratio", type=float, default=DEFAULT_MIN_POINTS_RATIO)
    parser.add_argument("--score", type=str, default="both",
                        choices=["oscillation", "mcr", "both",
                                 "pattern", "both_pattern"],
                        help="Ranking score for the winner: 'oscillation' = highest "
                             "normalized std (old behavior, may pick a low-load "
                             "window), 'mcr' = highest average http_mcr, 'both' = "
                             "product of oscillation and normalized avg_mcr, "
                             "'pattern' = highest periodic-pattern score (peak "
                             "|autocorrelation|), 'both_pattern' = oscillation * "
                             "pattern (must both oscillate and be predictable).")
    parser.add_argument("--max_lag", type=int, default=DEFAULT_MAX_LAG,
                        help="Largest autocorrelation lag (minutes) considered "
                             "for the periodic-pattern score (capped by window "
                             "size).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "all"],
                        help="Which part of the timeline to scan for candidate "
                             "segments: 'test' = only segments fully inside the "
                             "model's TEST split (default, mirrors "
                             "build_windows.py), 'all' = scan the full timeline "
                             "of every service.")
    parser.add_argument("--train_frac", type=float, default=PREPROCESSING.TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=PREPROCESSING.VAL_FRAC)
    parser.add_argument("--test_start_ms", type=int, default=TEST_START_MS)
    return parser.parse_args()


def print_eval_results(df: pd.DataFrame) -> None:
    cols = ["msname", "oscillation", "avg_mcr", "std_mcr", "win_start", "win_end",
            "n_points", "n_nonzero", "g_min", "g_max", "max_ts", "test_len", "n_rows"]
    show = df.head(20).copy()
    for _, r in show.iterrows():
        pat = f"pat={r['pattern']:.3f} " if pd.notna(r.get("pattern", np.nan)) else "pat=  n/a "
        print(f"{r['msname']:<12} {r['oscillation']:>7.3f} "
              f"avg={r['avg_mcr']:>9.3g} std={r['std_mcr']:>9.3g} "
              f"start={datetime.fromtimestamp(r['win_start']/1000):%Y-%m-%d %H:%M} "
              f"end={datetime.fromtimestamp(r['win_end']/1000):%Y-%m-%d %H:%M} "
              f"{pat}pts={int(r['n_points']):>3d} nz={int(r['n_nonzero']):>3d} "
              f"g=[{r['g_min']:.3g},{r['g_max']:.3g}] "
              f"test_len={int(r['test_len']):>4d} n_rows={int(r['n_rows']):>6d}")
    print()


def compute_score(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rank candidates by a score. `oscillation` is the std of the window's
    min-max-normalized http_mcr (shape only, unitless); `avg_mcr` is the raw
    average load over the window; `pattern` is the peak |autocorrelation| of
    http_mcr over the tested lags (how periodic/predictable the curve is). Rank
    by:
      * oscillation:  current behavior (highest normalized std).
      * mcr:          highest average load.
      * both:         product of oscillation and min-max-normalized avg_mcr, so
                      a window must be both oscillating AND carry meaningful load.
      * pattern:      highest periodic-pattern score (windows with a repeating,
                      predictable shape).
      * both_pattern: oscillation * pattern, so a window must both oscillate AND
                      be predictable.
    """
    df = df.copy()
    if mode == "oscillation":
        df["score"] = df["oscillation"]
    elif mode == "mcr":
        df["score"] = df["avg_mcr"]
    elif mode == "pattern":
        df = df.dropna(subset=["pattern"])
        df["score"] = df["pattern"]
    elif mode == "both_pattern":
        df = df.dropna(subset=["pattern"])
        df["score"] = df["oscillation"] * df["pattern"]
    else:
        span = df["avg_mcr"].max() - df["avg_mcr"].min()
        mcr_norm = ((df["avg_mcr"] - df["avg_mcr"].min()) / span) if span > 0 else 0.0
        df["score"] = df["oscillation"] * mcr_norm
    return df.sort_values("score", ascending=False)


def main():
    args = parse_args()

    parquet_root = args.parquet_dir or Paths.PARQUET_ROOT
    mcr_dir = os.path.join(parquet_root, "msrtmcre")
    msresource_dir = os.path.join(parquet_root, "msresource")
    window_ms = int(args.window_hours * MS_PER_HOUR)
    expected = window_ms // 60_000 + 1
    split_frac = args.train_frac + args.val_frac

    names = load_service_names()

    # Mirror build_windows.py subset selection.
    if args.max_services and len(names) > args.max_services:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(names), size=args.max_services, replace=False)
        names = sorted(np.array(names)[idx].tolist())

    sizes = load_service_split_sizes()
    svc_n_df = pd.DataFrame(
        {"msname": [n for n in names if n in sizes],
         "n": [sizes[n] for n in names if n in sizes]}
    )

    in_clause = ",".join(f"'{n}'" for n in names)

    con = duckdb.connect()
    con.execute("SET threads TO 16")
    con.execute("SET memory_limit = \"16GB\"")

    log("Scanning http_mcr (msrtmcre) across all sliding segments...")
    df = query_mcrtmcr_oscillations(
        con, mcr_dir, window_ms, in_clause, split_frac,
        args.test_start_ms, expected, svc_n_df,
        restrict_test=(args.split == "test"),
    )

    log("Filtering candidates...")
    valid = df.dropna(subset=["oscillation"])
    valid = valid[valid["n_points"] >= max(1, int(expected * args.min_points_ratio))]

    if args.score in ("pattern", "both_pattern"):
        log("Computing periodic-pattern (autocorrelation) scores...")
        lags = build_lags(args.max_lag, expected)
        series = query_mcr_series(con, mcr_dir, in_clause, args.test_start_ms,
                                  restrict_test=(args.split == "test"))
        pat = compute_pattern_scores(series, expected, lags)
        valid = valid.merge(pat.rename(columns={"timestamp": "win_end"}),
                            on=["msname", "win_end"], how="left")
    else:
        valid["pattern"] = np.nan

    valid = compute_score(valid, args.score)
    print_eval_results(valid)

    if valid.empty:
        log("No valid windows found.")
        return

    winner = valid.iloc[0]
    pat_str = (f", pattern={winner['pattern']:.3f}"
               if pd.notna(winner.get("pattern", np.nan)) else "")
    log(f"Winner: {winner['msname']} "
        f"(score={winner['score']:.3f}, oscillation={winner['oscillation']:.3f}, "
        f"avg_mcr={winner['avg_mcr']:.3g}{pat_str}, window "
        f"{datetime.fromtimestamp(winner['win_start']/1000):%Y-%m-%d %H:%M} -> "
        f"{datetime.fromtimestamp(winner['win_end']/1000):%Y-%m-%d %H:%M}, "
        f"test_len={int(winner['test_len'])} min, max_ts="
        f"{datetime.fromtimestamp(winner['max_ts']/1000):%Y-%m-%d %H:%M})")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # http_mcr table for the winner window
    mcr_full = query_winner_mcr(con, mcr_dir, winner["msname"],
                                winner["win_start"], winner["win_end"])
    mcr_full.to_csv(f"{out_dir}/http_mcr_{winner['msname']}_{ts_str}.csv",
                    index=False)

    # msresource CPU/memory for the winner window
    res_df = query_msresource_window(con, msresource_dir, winner["msname"],
                                     winner["win_start"], winner["win_end"])
    res_df.to_csv(f"{out_dir}/resource_{winner['msname']}_{ts_str}.csv",
                  index=False)

    plot_timeseries(mcr_full, res_df, winner["msname"], out_dir, ts_str)

    print(f"\nSaved outputs to {out_dir}/")


if __name__ == "__main__":
    main()

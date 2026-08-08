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
                               svc_n_df):
    """Compute, per service, the oscillation of the last {window_ms} window of
    http_mcr, restricted to services whose window lies fully inside the TEST
    split (as defined by build_windows.py), along with window stats.

    Test-split constraints (cheap, no msresource scan):
      * max_ts - window_ms >= test_start_ms  -- the last-6h window ends no
        earlier than the global test boundary (day 10.4), so it never
        straddles it for full-coverage services.
      * test_len >= expected_pts  -- the service's own test split
        (n - floor(split_frac * n)) is long enough to hold a full window.

    Assumes all services listed in in_clause are present in mcr_dir."""
    con.register("svc_n", svc_n_df)
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
        tested AS (
            SELECT a.msname, a.timestamp, a.http_mcr_sum, m.max_ts,
                   n.n AS n_rows,
                   n.n - CAST(FLOOR({split_frac} * n.n) AS BIGINT) AS test_len
            FROM agg a
            JOIN maxes m ON a.msname = m.msname
            JOIN svc_n n ON a.msname = n.msname
            WHERE m.max_ts - {window_ms} >= {test_start_ms}
              AND n.n - CAST(FLOOR({split_frac} * n.n) AS BIGINT) >= {expected_pts}
        ),
        last6 AS (
            SELECT msname, timestamp, http_mcr_sum, max_ts, test_len, n_rows
            FROM tested
            WHERE timestamp >= max_ts - {window_ms}
        ),
        stats AS (
            SELECT MIN(http_mcr_sum) AS g_min, MAX(http_mcr_sum) AS g_max
            FROM last6
        ),
        final AS (
            SELECT l.msname,
                   STDDEV((l.http_mcr_sum - s.g_min) / NULLIF(s.g_max - s.g_min, 0)) AS oscillation,
                   MIN(l.timestamp) AS win_start,
                   MAX(l.timestamp) AS win_end,
                   COUNT(*) AS n_points,
                   SUM(CASE WHEN l.http_mcr_sum > 0 THEN 1 ELSE 0 END) AS n_nonzero,
                   MAX(l.max_ts) AS max_ts,
                   MAX(l.test_len) AS test_len,
                   MAX(l.n_rows) AS n_rows,
                   MIN(s.g_min) AS g_min,
                   MAX(s.g_max) AS g_max
            FROM last6 l
            CROSS JOIN stats s
            GROUP BY l.msname
        )
        SELECT msname, oscillation, win_start, win_end, n_points, n_nonzero,
               g_min, g_max, max_ts, test_len, n_rows
        FROM final
        ORDER BY oscillation DESC
    """
    df = con.execute(sql).df()
    return df


def query_winner_mcr(con, mcr_dir, msname, win_start, win_end, g_min, g_max):
    """Query per-minute http_mcr for the winner window, min-max normalized to
    [0, 1] using the global g_min/g_max over all candidates' windows."""
    sql = f"""
        SELECT msname, timestamp, SUM(http_mcr) AS http_mcr_raw
        FROM read_parquet('{mcr_dir}/*.parquet')
        WHERE msname = '{msname}' AND timestamp >= {win_start}
          AND timestamp <= {win_end}
        GROUP BY msname, timestamp
        ORDER BY timestamp
    """
    df = con.execute(sql).df()
    span = g_max - g_min
    if span > 0:
        df["http_mcr"] = ((df["http_mcr_raw"] - g_min) / span).clip(0.0, 1.0)
    else:
        df["http_mcr"] = 0.0
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
        description="Find the microservice with the highest http_mcr oscillation over "
        "the last N hours of its data -- restricted to the model's TEST split as "
        "defined by build_windows.py -- and plot its http_mcr plus CPU/memory "
        "utilization for that window."
    )
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Root containing msrtmcre/ and msresource/ subdirs. "
                             "Defaults to Paths.PARQUET_ROOT.")
    parser.add_argument("--max_services", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--window_hours", type=float, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--min_points_ratio", type=float, default=DEFAULT_MIN_POINTS_RATIO)
    parser.add_argument("--train_frac", type=float, default=PREPROCESSING.TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=PREPROCESSING.VAL_FRAC)
    parser.add_argument("--test_start_ms", type=int, default=TEST_START_MS)
    return parser.parse_args()


def print_eval_results(df: pd.DataFrame) -> None:
    cols = ["msname", "oscillation", "win_start", "win_end", "n_points", "n_nonzero",
            "g_min", "g_max", "max_ts", "test_len", "n_rows"]
    show = df.head(20).copy()
    for _, r in show.iterrows():
        print(f"{r['msname']:<12} {r['oscillation']:>7.3f} "
              f"start={datetime.fromtimestamp(r['win_start']/1000):%Y-%m-%d %H:%M} "
              f"end={datetime.fromtimestamp(r['win_end']/1000):%Y-%m-%d %H:%M} "
              f"pts={int(r['n_points']):>3d} nz={int(r['n_nonzero']):>3d} "
              f"g=[{r['g_min']:.3g},{r['g_max']:.3g}] "
              f"test_len={int(r['test_len']):>4d} n_rows={int(r['n_rows']):>6d}")
    print()


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

    log("Scanning http_mcr (msrtmcre) for candidate services...")
    df = query_mcrtmcr_oscillations(
        con, mcr_dir, window_ms, in_clause, split_frac,
        args.test_start_ms, expected, svc_n_df
    )

    log("Filtering candidates...")
    valid = df.dropna(subset=["oscillation"])
    valid = valid[valid["n_points"] >= max(1, int(expected * args.min_points_ratio))]
    print_eval_results(valid)

    if valid.empty:
        log("No valid services found in the test split.")
        return

    winner = valid.iloc[0]
    log(f"Winner: {winner['msname']} "
        f"(oscillation={winner['oscillation']:.3f}, window "
        f"{datetime.fromtimestamp(winner['win_start']/1000):%Y-%m-%d %H:%M} -> "
        f"{datetime.fromtimestamp(winner['win_end']/1000):%Y-%m-%d %H:%M}, "
        f"test_len={int(winner['test_len'])} min, max_ts="
        f"{datetime.fromtimestamp(winner['max_ts']/1000):%Y-%m-%d %H:%M})")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # http_mcr table for the winner window
    mcr_full = query_winner_mcr(con, mcr_dir, winner["msname"],
                                winner["win_start"], winner["win_end"],
                                winner["g_min"], winner["g_max"])
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

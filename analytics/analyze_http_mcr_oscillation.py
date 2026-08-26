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
SERVICE_INDEX_PATH = os.path.join(Paths.WINDOWS_DIR, "_service_index.json")
SERVICE_ARRAYS_PATH = os.path.join(Paths.WINDOWS_DIR, "_service_arrays.npy")
MS_PER_HOUR = 3_600_000


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_service_cache(con, mcr_dir):
    """Build service cache files: _service_index.json and _service_arrays.npy.

    The _service_index.json contains:
      - "index": dict mapping msname to [length_of_timeline, timestamp_range]
      - "train_frac": used by build_windows.py to derive splits
      - "val_frac": used by build_windows.py to derive splits
      - "stats": dict with per-service statistics (min, max, std, n_nonzero) for full sequence
    
    The _service_arrays.npy contains the actual http_mcr arrays for each service.
    """
    log("Building service cache (single query)...")

    # Single query to get all data at once
    sql = f"""
        SELECT msname, timestamp, SUM(http_mcr) AS http_mcr_sum
        FROM read_parquet('{mcr_dir}/*.parquet')
        WHERE http_mcr IS NOT NULL
        GROUP BY msname, timestamp
        ORDER BY msname, timestamp
    """
    df = con.execute(sql).df()

    # Group by msname and build arrays
    grouped = df.groupby("msname")
    names = list(grouped.groups.keys())
    sizes = {name: len(group) for name, name, group in grouped}

    index_data = {
        "index": {},
        "train_frac": PREPROCESSING.TRAIN_FRAC,
        "val_frac": PREPROCESSING.VAL_FRAC,
        "stats": {},  # Will store min, max, std, n_nonzero for each service
    }

    arrays_list = []
    for msname in names:
        group = grouped.get_group(msname)
        length = len(group)
        index_data["index"][msname] = [length, length]
        
        # Compute statistics for full sequence
        http_mcr_series = group["http_mcr_sum"].values
        g_min = float(http_mcr_series.min()) if len(http_mcr_series) > 0 else 0.0
        g_max = float(http_mcr_series.max()) if len(http_mcr_series) > 0 else 0.0
        std_mcr = float(http_mcr_series.std()) if len(http_mcr_series) > 0 else 0.0
        n_nonzero = int((http_mcr_series > 0).sum()) if len(http_mcr_series) > 0 else 0
        
        index_data["stats"][msname] = {
            "g_min": g_min,
            "g_max": g_max,
            "std_mcr": std_mcr,
            "n_nonzero": n_nonzero,
            "length": length,
        }
        
        arrays_list.append(http_mcr_series)

    # Stack all arrays into a single numpy array (variable lengths - use object array)
    if arrays_list:
        arrays_array = np.array(arrays_list, dtype=object)
        if not os.path.exists(os.path.dirname(SERVICE_ARRAYS_PATH)):
            os.makedirs(os.path.dirname(SERVICE_ARRAYS_PATH), exist_ok=True)
        np.save(SERVICE_ARRAYS_PATH, arrays_array, allow_pickle=True)

    # Save the index data to JSON
    os.makedirs(os.path.dirname(SERVICE_INDEX_PATH), exist_ok=True)
    with open(SERVICE_INDEX_PATH, "w") as f:
        json.dump(index_data, f, indent=2)

    log(f"Service cache built: {SERVICE_INDEX_PATH}, {SERVICE_ARRAYS_PATH}")


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
                                expected_pts, svc_n_df):
    """Scan every {window_ms}-long sliding segment of each service's http_mcr
    timeline (not just the last window) and return one row per
    (service, window end) with window stats.

    A segment is a frame of `expected_pts` consecutive 1-minute rows
    (ROWS BETWEEN expected_pts-1 PRECEDING AND CURRENT ROW), matching the
    existing window definition. oscillation is the std of the min-max
    normalized http_mcr within the segment, i.e. std_mcr / (g_max - g_min);
    win_start/win_end are the actual first/last timestamps of the segment.

    Every segment of the full timeline is a candidate.

    Assumes all services listed in in_clause are present in mcr_dir."""
    con.register("svc_n", svc_n_df)
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
                   n.n AS n_rows
            FROM wstats w
            LEFT JOIN maxes m ON w.msname = m.msname
            LEFT JOIN svc_n n ON w.msname = n.msname
        )
        SELECT msname, oscillation, avg_mcr, std_mcr, win_start, win_end,
               n_points, n_nonzero, g_min, g_max, max_ts, n_rows, n_rows AS test_len
        FROM final
        ORDER BY oscillation DESC
    """
    df = con.execute(sql).df()
    return df


def load_cached_stats():
    """Load per-service statistics from cached _service_index.json."""
    with open(SERVICE_INDEX_PATH, "r") as f:
        data = json.load(f)
    return data.get("stats", {})


def compute_full_sequence_oscillation_from_cache(names, sizes, min_points):
    """Compute oscillation stats for each service from cached statistics.
    
    Much faster than querying parquet since we use precomputed stats.
    """
    import numpy as np
    
    stats = load_cached_stats()  # Dict with per-service stats
    
    results = []
    for msname in names:
        if msname not in sizes:
            continue
        n_rows = sizes[msname]
        if n_rows < min_points:
            continue
         
        # Get cached stats for this service
        if msname not in stats:
            continue
        stat = stats[msname]
        
        g_min = stat["g_min"]
        g_max = stat["g_max"]
        std_mcr = stat["std_mcr"]
        n_points = stat["length"]
        n_nonzero = stat["n_nonzero"]
        
        if g_max - g_min == 0:
            oscillation = 0.0
        else:
            oscillation = std_mcr / (g_max - g_min)
        
        # win_start and win_end: first and last timestamps of the service
        # Since we don't store exact timestamps in stats, approximate:
        # win_start = 0 (first minute), win_end = (n_rows-1) * 60_000 ms
        win_start = 0
        win_end = int((n_rows - 1) * 60_000)
        max_ts = win_end
        
        results.append({
            "msname": msname,
            "oscillation": oscillation,
            "avg_mcr": 0.0,
            "std_mcr": std_mcr,
            "win_start": win_start,
            "win_end": win_end,
            "n_points": n_points,
            "n_nonzero": n_nonzero,
            "g_min": g_min,
            "g_max": g_max,
            "max_ts": max_ts,
            "n_rows": n_rows,
            "test_len": n_rows,
        })
    
    return pd.DataFrame(results)


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
        description="Analyze each service's full http_mcr timeline -- ranked by oscillation "
        "(std of normalized http_mcr over sliding window of --window_hours, or over full sequence if not specified) "
        "-- and plot the winner's http_mcr plus CPU/memory utilization for that window."
    )
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Root containing msrtmcre/ and msresource/ subdirs. "
                             "Defaults to Paths.PARQUET_ROOT.")
    parser.add_argument("--max_services", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--out_dir", type=str, default=Paths.ANALYTICS_OUT_DIR)
    parser.add_argument("--window_hours", type=float, default=None,
                        help="Window size in hours for oscillation calculation. "
                             "If not specified, analyzes the full sequence for each service.")
    parser.add_argument("--min_hours", type=float, default=1.0,
                        help="Minimum hours of data required per service/window. "
                             "In sliding window mode: minimum hours within window. "
                             "In full sequence mode: minimum total hours of data per service.")
    return parser.parse_args()


def format_relative(ms: int) -> str:
    """Format milliseconds as relative days/hours/minutes from trace start (ms=0)."""
    minutes = ms // 60_000
    days = minutes // 1440
    hours = (minutes % 1440) // 60
    mins = minutes % 60
    if days > 0:
        return f"day {days} {hours:02d}:{mins:02d}"
    elif hours > 0:
        return f"{hours}h {mins:02d}m"
    else:
        return f"{mins}m"


def print_eval_results(df: pd.DataFrame) -> None:
    cols = ["msname", "oscillation", "std_mcr", "win_start", "win_end",
            "n_points", "n_nonzero", "g_min", "g_max", "max_ts", "test_len", "n_rows"]
    show = df.head(20).copy()
    for _, r in show.iterrows():
        print(f"{r['msname']:<12} {r['oscillation']:>7.3f} "
              f"std={r['std_mcr']:>9.3g} "
              f"start={format_relative(r['win_start'])} "
              f"end={format_relative(r['win_end'])} "
              f"pts={int(r['n_points']):>3d} nz={int(r['n_nonzero']):>3d} "
              f"g=[{r['g_min']:.3g},{r['g_max']:.3g}] "
              f"test_len={int(r['test_len']):>4d} n_rows={int(r['n_rows']):>6d}")
    print()


def main():
    args = parse_args()

    parquet_root = args.parquet_dir or Paths.PARQUET_ROOT
    mcr_dir = os.path.join(parquet_root, "msrtmcre")
    msresource_dir = os.path.join(parquet_root, "msresource")

    con = duckdb.connect()
    con.execute("SET threads TO 16")
    con.execute("SET memory_limit = \"16GB\"")

    # Use cached service info if available, otherwise build cache from parquet
    idx_path = SERVICE_INDEX_PATH
    arrays_path = SERVICE_ARRAYS_PATH
    if not (os.path.exists(idx_path) and os.path.exists(arrays_path)):
        log("Building service cache from parquet files...")
        build_service_cache(con, mcr_dir)
    else:
        log("Loading service info from cache...")

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

    if args.window_hours is None:
        log("Analyzing full http_mcr sequence from cached stats...")
        min_points = int(args.min_hours * 60) + 1
        df = compute_full_sequence_oscillation_from_cache(names, sizes, min_points)
        window_size_for_filtering = None
    else:
        window_ms = int(args.window_hours * MS_PER_HOUR)
        expected = window_ms // 60_000 + 1
        log("Scanning http_mcr (msrtmcre) across all sliding segments...")
        df = query_mcrtmcr_oscillations(
            con, mcr_dir, window_ms, in_clause,
            expected, svc_n_df,
        )
        window_size_for_filtering = expected

    log("Filtering candidates...")
    valid = df.dropna(subset=["oscillation"])
    min_points = int(args.min_hours * 60) + 1
    if args.window_hours is not None:
        # Sliding window mode: filter by minimum points in window
        valid = valid[valid["n_points"] >= min_points]
    # For full sequence mode, filtering already done in compute_full_sequence_oscillation_from_cache
    valid = valid.sort_values("oscillation", ascending=False)
    print_eval_results(valid)

    if valid.empty:
        log("No valid windows found.")
        return

    winner = valid.iloc[0]
    if args.window_hours is None:
        log(f"Winner: {winner['msname']} "
            f"(oscillation={winner['oscillation']:.3f}, "
            f"std_mcr={winner['std_mcr']:.3g}, full sequence "
            f"{format_relative(winner['win_start'])} -> "
            f"{format_relative(winner['win_end'])}, "
            f"length={int(winner['n_rows'])} min, max_ts="
            f"{format_relative(winner['max_ts'])})")
    else:
        log(f"Winner: {winner['msname']} "
            f"(oscillation={winner['oscillation']:.3f}, "
            f"std_mcr={winner['std_mcr']:.3g}, window "
            f"{format_relative(winner['win_start'])} -> "
            f"{format_relative(winner['win_end'])}, "
            f"test_len={int(winner['test_len'])} min, max_ts="
            f"{format_relative(winner['max_ts'])})")

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

    mcr_csv = f"{out_dir}/http_mcr_{winner['msname']}_{ts_str}.csv"
    res_csv = f"{out_dir}/resource_{winner['msname']}_{ts_str}.csv"
    mcr_png = f"{out_dir}/http_mcr_{winner['msname']}_{ts_str}.png"
    cpu_png = f"{out_dir}/cpu_{winner['msname']}_{ts_str}.png"
    mem_png = f"{out_dir}/memory_{winner['msname']}_{ts_str}.png"

    print(f"\nSaved outputs:")
    print(f"  {mcr_csv}")
    print(f"  {res_csv}")
    print(f"  {mcr_png}")
    print(f"  {cpu_png}")
    print(f"  {mem_png}")


if __name__ == "__main__":
    main()

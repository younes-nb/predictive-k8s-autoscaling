import csv
import gc
import glob
import logging
import os
import pickle
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_preprocessing_defaults import PREPROCESSING

STEP_MS = 60_000
MIN_ALIGNED = 10
MIN_UMS = 1
LOG_EVERY_N = 20


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(os.path.join(out_dir, "analyze_upstream.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def _extract_pairs_from_df(df: pl.DataFrame, valid: Set[str]) -> List[Tuple[str, str]]:
    if "um" not in df.columns or "dm" not in df.columns:
        return []

    valid_list = list(valid)
    filtered = (
        df.lazy()
        .filter(
            pl.col("um").is_in(valid_list)
            & pl.col("dm").is_in(valid_list)
            & (pl.col("um") != pl.col("dm"))
        )
        .select(["dm", "um"])
        .unique()
        .collect()
    )
    return list(zip(filtered["dm"].to_list(), filtered["um"].to_list()))


def _scan_parquet_for_pairs(parquet_dir: str, valid: Set[str]) -> Dict[str, Set[str]]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
    if not files:
        return {}

    adj: Dict[str, Set[str]] = {}
    logging.info("Scanning %d callgraph parquet files…", len(files))

    for i, fp in enumerate(files, 1):
        try:
            df = (
                pl.scan_parquet(fp)
                .select(["um", "dm"])
                .drop_nulls()
                .collect(engine="streaming")
            )
            for dm, um in _extract_pairs_from_df(df, valid):
                adj.setdefault(dm, set()).add(um)
            del df
        except Exception as exc:
            logging.warning("  [SKIP parquet] %s: %s", Path(fp).name, exc)

        if i % LOG_EVERY_N == 0:
            gc.collect()
            logging.info("  %d/%d parquet | DMs so far: %d", i, len(files), len(adj))

    gc.collect()
    return adj


def _scan_csv_for_pairs(raw_dir: str, valid: Set[str]) -> Dict[str, Set[str]]:
    files = sorted(
        glob.glob(os.path.join(raw_dir, "*.csv"))
        + glob.glob(os.path.join(raw_dir, "*.csv.gz"))
    )
    if not files:
        return {}

    adj: Dict[str, Set[str]] = {}
    logging.info("Scanning %d raw callgraph CSV files…", len(files))

    for i, fp in enumerate(files, 1):
        try:
            df = pl.read_csv(
                fp,
                columns=["um", "dm"],
                infer_schema_length=500,
                low_memory=True,
                try_parse_dates=False,
            ).drop_nulls()
            for dm, um in _extract_pairs_from_df(df, valid):
                adj.setdefault(dm, set()).add(um)
            del df
        except Exception as exc:
            logging.warning("  [SKIP CSV] %s: %s", Path(fp).name, exc)

        if i % LOG_EVERY_N == 0:
            gc.collect()
            logging.info("  %d/%d CSV | DMs so far: %d", i, len(files), len(adj))

    gc.collect()
    return adj


def build_adjacency(
    valid_msnames: Set[str],
    cache_path: str,
    rebuild: bool,
) -> Dict[str, Set[str]]:
    if not rebuild and os.path.exists(cache_path):
        logging.info("Loading cached adjacency map from %s", cache_path)
        with open(cache_path, "rb") as f:
            adj = pickle.load(f)
        logging.info("Cached adjacency: %d DMs loaded", len(adj))
        return adj

    cg = DATASET_TABLES["mscallgraph"]
    parquet_dir = cg.get("parquet_dir", "")
    raw_dir = cg.get("raw_dir", "")

    parquet_files = glob.glob(os.path.join(parquet_dir, "part-*.parquet"))
    raw_files = glob.glob(os.path.join(raw_dir, "*.csv")) + glob.glob(
        os.path.join(raw_dir, "*.csv.gz")
    )

    if not parquet_files and not raw_files:
        raise FileNotFoundError(
            "No MSCallGraph data found.\n"
            f"  Expected parquet in: {parquet_dir}\n"
            f"  Expected raw CSV in: {raw_dir}\n"
            "Fetch the callgraph traces first:\n"
            "  python preprocessing/fetch_traces.py "
            "--start_date 0d0 --end_date 7d0 --tables mscallgraph"
        )

    if parquet_files:
        adj = _scan_parquet_for_pairs(parquet_dir, valid_msnames)
        if not adj:
            logging.warning("Parquet scan yielded no pairs; falling back to raw CSVs.")
            adj = _scan_csv_for_pairs(raw_dir, valid_msnames)
    else:
        logging.info("No callgraph parquet found; using raw CSV files.")
        adj = _scan_csv_for_pairs(raw_dir, valid_msnames)

    adj = {dm: ums for dm, ums in adj.items() if len(ums) >= MIN_UMS}

    logging.info(
        "Adjacency complete: %d DMs with ≥%d upstream callers", len(adj), MIN_UMS
    )

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(adj, f)
    logging.info("Adjacency cached to %s", cache_path)
    return adj


def get_valid_msnames(parquet_dir: str) -> Set[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No MSResource parquet files found in {parquet_dir}.\n"
            "Run preprocessing first to ingest MSResource."
        )
    names = (
        pl.scan_parquet(files)
        .select("msname")
        .unique()
        .collect(engine="streaming")["msname"]
        .to_list()
    )
    result = {n for n in names if isinstance(n, str)}
    logging.info("MSResource: %d unique stateless MSes", len(result))
    return result


def load_timeseries_batch(
    parquet_dir: str,
    msnames: List[str],
) -> pl.DataFrame:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
    if not files or not msnames:
        return pl.DataFrame()

    try:
        df = (
            pl.scan_parquet(files)
            .filter(pl.col("msname").is_in(msnames))
            .select(["timestamp", "msname", "cpu_utilization", "memory_utilization"])
            .drop_nulls()
            .group_by(["timestamp", "msname"])
            .agg(
                pl.col("cpu_utilization").mean(),
                pl.col("memory_utilization").mean(),
            )
            .sort(["msname", "timestamp"])
            .collect(engine="streaming")
        )
        return df
    except Exception as exc:
        logging.warning("load_timeseries_batch failed: %s", exc)
        return pl.DataFrame()


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X[:, np.newaxis]
    X_aug = np.column_stack([X, np.ones(len(X))])
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-14:
        return float("nan")
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ coeffs
        return float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)
    except Exception:
        return float("nan")


def analyze_one_ms(
    dm: str,
    um_set: Set[str],
    ts_df: pl.DataFrame,
    pred_horizon: int,
    keep_arrays: bool = False,
) -> Optional[dict]:
    horizon_offset = pred_horizon * STEP_MS

    dm_df = (
        ts_df.filter(pl.col("msname") == dm)
        .select(["timestamp", "cpu_utilization"])
        .rename({"cpu_utilization": "dm_cpu"})
        .sort("timestamp")
    )
    um_df = ts_df.filter(pl.col("msname").is_in(list(um_set)))

    if dm_df.is_empty() or um_df.is_empty():
        return None

    up_agg = (
        um_df.group_by("timestamp")
        .agg(
            pl.col("cpu_utilization").mean().alias("up_cpu"),
            pl.col("memory_utilization").mean().alias("up_mem"),
            pl.col("msname").n_unique().alias("n_um_active"),
        )
        .sort("timestamp")
    )

    dm_future = dm_df.rename({"timestamp": "future_ts", "dm_cpu": "dm_cpu_tH"})

    ds_A = up_agg.with_columns(
        (pl.col("timestamp").cast(pl.Int64) + horizon_offset).alias("future_ts")
    ).join(dm_future, on="future_ts", how="inner")

    ds_B = (
        dm_df.with_columns(
            (pl.col("timestamp").cast(pl.Int64) + horizon_offset).alias("future_ts")
        )
        .rename({"dm_cpu": "dm_cpu_t"})
        .join(dm_future, on="future_ts", how="inner")
    )

    ds_C = ds_A.select(["timestamp", "up_cpu", "up_mem", "dm_cpu_tH"]).join(
        ds_B.select(["timestamp", "dm_cpu_t"]),
        on="timestamp",
        how="inner",
    )

    n = len(ds_C)
    if n < MIN_ALIGNED:
        return None

    up_cpu = ds_C["up_cpu"].to_numpy().astype(np.float64)
    up_mem = ds_C["up_mem"].to_numpy().astype(np.float64)
    dm_t = ds_C["dm_cpu_t"].to_numpy().astype(np.float64)
    dm_tH = ds_C["dm_cpu_tH"].to_numpy().astype(np.float64)

    mask = (
        np.isfinite(up_cpu)
        & np.isfinite(up_mem)
        & np.isfinite(dm_t)
        & np.isfinite(dm_tH)
    )
    up_cpu, up_mem, dm_t, dm_tH = up_cpu[mask], up_mem[mask], dm_t[mask], dm_tH[mask]
    n = int(mask.sum())
    if n < MIN_ALIGNED:
        return None

    r_up_cpu, p_up_cpu = stats.pearsonr(up_cpu, dm_tH)
    r_up_mem, p_up_mem = stats.pearsonr(up_mem, dm_tH)
    r_autocorr, _ = stats.pearsonr(dm_t, dm_tH)

    rs_up_cpu, _ = stats.spearmanr(up_cpu, dm_tH)
    rs_up_mem, _ = stats.spearmanr(up_mem, dm_tH)
    rs_autocorr, _ = stats.spearmanr(dm_t, dm_tH)

    r2_up_cpu_only = _ols_r2(up_cpu, dm_tH)
    r2_up_mem_only = _ols_r2(up_mem, dm_tH)
    r2_upstream = _ols_r2(np.column_stack([up_cpu, up_mem]), dm_tH)
    r2_autocorr = _ols_r2(dm_t, dm_tH)
    r2_combined = _ols_r2(np.column_stack([up_cpu, up_mem, dm_t]), dm_tH)

    r2_gain = (
        float(r2_combined) - float(r2_autocorr)
        if np.isfinite(r2_combined) and np.isfinite(r2_autocorr)
        else float("nan")
    )

    result = {
        "dm": dm,
        "n_um": len(um_set),
        "n_pairs": n,
        "r_up_cpu": float(r_up_cpu),
        "p_up_cpu": float(p_up_cpu),
        "r_up_mem": float(r_up_mem),
        "p_up_mem": float(p_up_mem),
        "r_autocorr": float(r_autocorr),
        "rs_up_cpu": float(rs_up_cpu),
        "rs_up_mem": float(rs_up_mem),
        "rs_autocorr": float(rs_autocorr),
        "r2_up_cpu_only": float(r2_up_cpu_only),
        "r2_up_mem_only": float(r2_up_mem_only),
        "r2_upstream": float(r2_upstream),
        "r2_autocorr": float(r2_autocorr),
        "r2_combined": float(r2_combined),
        "r2_gain": float(r2_gain),
    }

    if keep_arrays:
        result["_up_cpu"] = up_cpu
        result["_up_mem"] = up_mem
        result["_dm_t"] = dm_t
        result["_dm_tH"] = dm_tH

    return result


def _finite(vals: List[float]) -> np.ndarray:
    return np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)


def plot_correlation_distribution(
    results: List[dict],
    pred_horizon: int,
    out_path: str,
) -> None:
    r_up_cpu = _finite([r["r_up_cpu"] for r in results])
    r_up_mem = _finite([r["r_up_mem"] for r in results])
    r_autocorr = _finite([r["r_autocorr"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    triples = [
        (axes[0], r_up_cpu, "Upstream CPU at t  →  Target CPU at t+H", "steelblue"),
        (
            axes[1],
            r_up_mem,
            "Upstream MEM at t  →  Target CPU at t+H",
            "mediumseagreen",
        ),
        (
            axes[2],
            r_autocorr,
            f"Target CPU at t  →  Target CPU at t+H\n" "(autocorr baseline)",
            "coral",
        ),
    ]
    for ax, vals, title, color in triples:
        ax.hist(vals, bins=30, color=color, edgecolor="white", alpha=0.85)
        mean_v = float(np.mean(vals))
        ax.axvline(
            mean_v, color="black", linestyle="--", lw=1.5, label=f"mean = {mean_v:+.3f}"
        )
        ax.axvline(0, color="gray", linestyle=":", lw=1)
        ax.set_xlabel("Pearson r", fontsize=10)
        ax.set_ylabel("MS count", fontsize=10)
        ax.set_title(title, fontsize=9, pad=6)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Upstream predictive correlation  |  "
        f"H = {pred_horizon} steps  ({pred_horizon} min)  |  "
        f"n = {len(results)} target MSes",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


def plot_r2_comparison(
    results: List[dict],
    pred_horizon: int,
    out_path: str,
    max_bars: int = 50,
) -> None:
    valid = [
        r
        for r in results
        if np.isfinite(r["r2_upstream"]) and np.isfinite(r["r2_autocorr"])
    ]
    sorted_r = sorted(valid, key=lambda r: r["r2_upstream"], reverse=True)[:max_bars]
    if not sorted_r:
        return

    labels = [r["dm"][:14] for r in sorted_r]
    r2_up = [r["r2_upstream"] for r in sorted_r]
    r2_auto = [r["r2_autocorr"] for r in sorted_r]
    r2_comb = [r["r2_combined"] for r in sorted_r]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
    ax.bar(x - w, r2_up, w, label="Upstream only", color="steelblue", alpha=0.85)
    ax.bar(x, r2_auto, w, label="Autocorr (naive)", color="coral", alpha=0.85)
    ax.bar(
        x + w,
        r2_comb,
        w,
        label="Upstream + self (combined)",
        color="mediumseagreen",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("OLS R²", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Upstream vs Autocorr vs Combined  R²  (H={pred_horizon} min, "
        f"top {len(sorted_r)} by upstream R²)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


def plot_top_scatter(
    results: List[dict],
    pred_horizon: int,
    out_path: str,
    top_n: int = 6,
    max_pts: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> None:
    candidates = [r for r in results if "_up_cpu" in r and np.isfinite(r["r_up_cpu"])]
    if not candidates:
        logging.warning("No scatter data available (top_n results missing arrays).")
        return

    top = sorted(candidates, key=lambda r: r["r_up_cpu"], reverse=True)[:top_n]
    n_cols = 2
    n_rows = len(top)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    _rng = rng if rng is not None else np.random.default_rng(42)

    for row_idx, res in enumerate(top):
        dm = res["dm"]
        for col_idx, (feat, feat_label, color) in enumerate(
            [
                (res["_up_cpu"], "Mean upstream CPU at t", "steelblue"),
                (res["_up_mem"], "Mean upstream MEM at t", "mediumseagreen"),
            ]
        ):
            ax = axes[row_idx][col_idx]
            target = res["_dm_tH"]

            n = len(feat)
            idx = (
                _rng.choice(n, size=min(max_pts, n), replace=False)
                if n > max_pts
                else np.arange(n)
            )
            ax.scatter(
                feat[idx], target[idx], alpha=0.30, s=6, color=color, linewidths=0
            )

            m, b = np.polyfit(feat, target, 1)
            xr = np.linspace(feat.min(), feat.max(), 200)
            ax.plot(xr, m * xr + b, "k--", lw=1.5, label=f"slope={m:.3f}")

            pearson_r = res["r_up_cpu"] if col_idx == 0 else res["r_up_mem"]
            r2_val = res["r2_up_cpu_only"] if col_idx == 0 else res["r2_up_mem_only"]
            ax.set_xlabel(feat_label, fontsize=9)
            ax.set_ylabel(f"Target CPU at t+{pred_horizon}", fontsize=9)
            ax.set_title(
                f"{dm[:24]}  |  r={pearson_r:+.3f}  R²={r2_val:.3f}",
                fontsize=9,
                pad=4,
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

    fig.suptitle(
        f"Scatter: upstream features vs target CPU at t+{pred_horizon} min"
        f"  (top {len(top)} MSes by upstream CPU correlation)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


def _interpret(results: List[dict], pred_horizon: int) -> None:
    if not results:
        return

    n = len(results)
    r2_up = _finite([r["r2_upstream"] for r in results])
    r2_auto = _finite([r["r2_autocorr"] for r in results])
    r2_comb = _finite([r["r2_combined"] for r in results])
    r_up_c = _finite([r["r_up_cpu"] for r in results])
    r_up_m = _finite([r["r_up_mem"] for r in results])
    r_auto = _finite([r["r_autocorr"] for r in results])

    frac_up_beats_auto = np.mean(
        [
            r["r2_upstream"] > r["r2_autocorr"]
            for r in results
            if np.isfinite(r["r2_upstream"]) and np.isfinite(r["r2_autocorr"])
        ]
    )
    avg_gain = float(np.mean(_finite([r["r2_gain"] for r in results])))

    sig_cpu = sum(
        1 for r in results if np.isfinite(r["p_up_cpu"]) and r["p_up_cpu"] < 0.05
    )
    sig_mem = sum(
        1 for r in results if np.isfinite(r["p_up_mem"]) and r["p_up_mem"] < 0.05
    )

    def _desc(r):
        return (
            "strong"
            if r > 0.4
            else "moderate" if r > 0.2 else "weak" if r > 0.05 else "negligible"
        )

    logging.info("")
    logging.info("━" * 62)
    logging.info("INTERPRETATION  (H = %d min, n = %d MSes)", pred_horizon, n)
    logging.info("━" * 62)
    logging.info(
        "Mean Pearson r  [upstream CPU → target CPU]  : %+.3f  (%s)",
        float(np.mean(r_up_c)),
        _desc(abs(float(np.mean(r_up_c)))),
    )
    logging.info(
        "Mean Pearson r  [upstream MEM → target CPU]  : %+.3f  (%s)",
        float(np.mean(r_up_m)),
        _desc(abs(float(np.mean(r_up_m)))),
    )
    logging.info(
        "Mean Pearson r  [autocorr baseline]           : %+.3f  (%s)",
        float(np.mean(r_auto)),
        _desc(abs(float(np.mean(r_auto)))),
    )
    logging.info("─" * 62)
    logging.info(
        "Mean R²  upstream [cpu+mem]                  : %.4f",
        float(np.mean(r2_up)),
    )
    logging.info(
        "Mean R²  autocorr                            : %.4f",
        float(np.mean(r2_auto)),
    )
    logging.info(
        "Mean R²  combined [upstream + self]          : %.4f",
        float(np.mean(r2_comb)),
    )
    logging.info("─" * 62)
    logging.info(
        "MSes where R²_upstream > R²_autocorr        : %.1f%%",
        frac_up_beats_auto * 100,
    )
    logging.info(
        "Avg R² gain (combined − autocorr)            : %+.4f",
        avg_gain,
    )
    logging.info(
        "Sig. upstream CPU (p<0.05)                   : %d / %d",
        sig_cpu,
        n,
    )
    logging.info(
        "Sig. upstream MEM (p<0.05)                   : %d / %d",
        sig_mem,
        n,
    )
    logging.info("─" * 62)

    # Plain-language verdict
    if float(np.mean(r_up_c)) > 0.3 and frac_up_beats_auto > 0.5:
        verdict = (
            "✅  STRONG upstream signal. Upstream CPU is moderately-to-strongly\n"
            "   correlated with future target CPU, and it beats the autocorr\n"
            "   baseline for the majority of MSes. Graph features would add\n"
            "   real predictive value to the RNNForecaster."
        )
    elif float(np.mean(r_up_c)) > 0.1 or avg_gain > 0.01:
        verdict = (
            "🟡  WEAK-TO-MODERATE upstream signal. Upstream features show some\n"
            "   correlation with future CPU, but the autocorr baseline is\n"
            "   competitive. Adding graph features may help for high-traffic MSes\n"
            "   but may not generalise across the full MS population."
        )
    else:
        verdict = (
            "❌  NEGLIGIBLE upstream signal. Upstream CPU/memory shows little\n"
            "   correlation with future target CPU beyond the autocorr baseline.\n"
            "   Consider longer horizons, different aggregation strategies, or\n"
            "   focusing on high-fan-in MSes only."
        )
    logging.info("")
    for line in verdict.splitlines():
        logging.info(line)
    logging.info("━" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pred_horizon",
        type=int,
        default=PREPROCESSING.PRED_HORIZON,
        help="Prediction horizon in 60-s steps (default: %(default)s → %(default)s min)",
    )
    ap.add_argument(
        "--max_dms",
        type=int,
        default=0,
        help="Max number of target DMs to analyze. 0 = all (default: %(default)s)",
    )
    ap.add_argument(
        "--dm_batch",
        type=int,
        default=25,
        help="DMs per time-series scan batch — trade speed vs peak RAM (default: %(default)s)",
    )
    ap.add_argument(
        "--max_ums_per_dm",
        type=int,
        default=0,
        help="Cap upstream callers per DM to avoid very high-degree nodes "
        "dominating (0 = no cap, default: %(default)s)",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=6,
        help="Number of MSes to include in the scatter plot (default: %(default)s)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--msresource_dir",
        type=str,
        default=PATHS.PARQUET_MSRESOURCE,
        help="MSResource parquet directory (default: %(default)s)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            PATHS.LOGS_DIR,
            f"upstream_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        help="Directory for all output files",
    )
    ap.add_argument(
        "--adj_cache",
        type=str,
        default=os.path.join(PATHS.LOGS_DIR, "upstream_adjacency.pkl"),
        help="Path to cache the adjacency map (default: %(default)s)",
    )
    ap.add_argument(
        "--rebuild_adj",
        action="store_true",
        help="Force a fresh adjacency scan even if the cache exists",
    )
    args = ap.parse_args()

    setup_logging(args.out_dir)
    rng = np.random.default_rng(args.seed)

    logging.info("=" * 62)
    logging.info("UPSTREAM PREDICTIVE POWER ANALYSIS")
    logging.info(
        "pred_horizon : %d steps  (%d min ahead)", args.pred_horizon, args.pred_horizon
    )
    logging.info("max_dms      : %s", args.max_dms or "all")
    logging.info("dm_batch     : %d", args.dm_batch)
    logging.info("msresource   : %s", args.msresource_dir)
    logging.info("out_dir      : %s", args.out_dir)
    logging.info("=" * 62)

    valid_msnames = get_valid_msnames(args.msresource_dir)

    adj = build_adjacency(valid_msnames, args.adj_cache, args.rebuild_adj)
    if not adj:
        logging.error(
            "No DMs found with upstream neighbours that exist in MSResource. "
            "Make sure the callgraph data has been fetched. Exiting."
        )
        sys.exit(1)

    if args.max_ums_per_dm > 0:
        trimmed = 0
        for dm in list(adj.keys()):
            if len(adj[dm]) > args.max_ums_per_dm:
                adj[dm] = set(
                    rng.choice(
                        sorted(adj[dm]),
                        size=args.max_ums_per_dm,
                        replace=False,
                    ).tolist()
                )
                trimmed += 1
        if trimmed:
            logging.info(
                "Capped UMs to %d for %d high-degree DMs", args.max_ums_per_dm, trimmed
            )

    all_dms = sorted(adj.keys())
    if args.max_dms and len(all_dms) > args.max_dms:
        chosen_idx = rng.choice(len(all_dms), size=args.max_dms, replace=False)
        target_dms = sorted([all_dms[i] for i in chosen_idx])
        logging.info("Sampled %d / %d DMs for analysis", len(target_dms), len(all_dms))
    else:
        target_dms = all_dms
        logging.info("Analyzing all %d DMs", len(target_dms))

    top_n_dms_placeholder: Set[str] = set()

    all_results: List[dict] = []
    total_batches = int(np.ceil(len(target_dms) / args.dm_batch))

    for batch_idx in range(total_batches):
        b_start = batch_idx * args.dm_batch
        b_end = b_start + args.dm_batch
        batch_dms = target_dms[b_start:b_end]

        logging.info(
            "Batch %d/%d  (%d DMs: %s … %s)",
            batch_idx + 1,
            total_batches,
            len(batch_dms),
            batch_dms[0],
            batch_dms[-1],
        )

        batch_all = list(
            {dm for dm in batch_dms} | {um for dm in batch_dms for um in adj[dm]}
        )

        ts_df = load_timeseries_batch(args.msresource_dir, batch_all)

        if ts_df.is_empty():
            logging.warning("  No time-series data found for this batch; skipping.")
            del ts_df
            gc.collect()
            continue

        batch_results = []
        for dm in batch_dms:
            res = analyze_one_ms(
                dm=dm,
                um_set=adj[dm],
                ts_df=ts_df,
                pred_horizon=args.pred_horizon,
                keep_arrays=False,
            )
            if res is not None:
                batch_results.append(res)
            else:
                logging.debug("  [skip] %s: insufficient aligned data points", dm)

        all_results.extend(batch_results)
        del ts_df, batch_results
        gc.collect()

        logging.info(
            "  Batch %d done | results: %d/%d DMs so far",
            batch_idx + 1,
            len(all_results),
            b_end,
        )

    if not all_results:
        logging.error(
            "No results produced. "
            "Possible causes: no time-series overlap between UMs and DMs, "
            "or too few timestamps per MS. Exiting."
        )
        sys.exit(1)

    sorted_by_r = sorted(
        [r for r in all_results if np.isfinite(r["r_up_cpu"])],
        key=lambda r: r["r_up_cpu"],
        reverse=True,
    )
    top_scatter_dms = {r["dm"] for r in sorted_by_r[: args.top_n]}

    if top_scatter_dms:
        logging.info(
            "Second pass: loading arrays for top %d MSes (scatter plots)…",
            len(top_scatter_dms),
        )
        top_all_msnames = list(
            top_scatter_dms | {um for dm in top_scatter_dms for um in adj[dm]}
        )
        ts_top = load_timeseries_batch(args.msresource_dir, top_all_msnames)

        if not ts_top.is_empty():
            top_with_arrays: List[dict] = []
            for dm in top_scatter_dms:
                res = analyze_one_ms(
                    dm=dm,
                    um_set=adj[dm],
                    ts_df=ts_top,
                    pred_horizon=args.pred_horizon,
                    keep_arrays=True,
                )
                if res is not None:
                    top_with_arrays.append(res)

            top_map = {r["dm"]: r for r in top_with_arrays}
            all_results = [top_map.get(r["dm"], r) for r in all_results]
            del ts_top, top_with_arrays
            gc.collect()

    n = len(all_results)
    logging.info("")
    logging.info("=" * 62)
    logging.info("AGGREGATE RESULTS  (%d target MSes)", n)
    logging.info("=" * 62)

    def _stat_row(label, vals):
        arr = _finite(vals)
        if len(arr) == 0:
            logging.info("%-42s  (no finite values)", label)
            return
        logging.info(
            "%-42s  mean=%+.4f  p25=%+.4f  p50=%+.4f  p75=%+.4f",
            label,
            float(np.mean(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 75)),
        )

    _stat_row("r  upstream CPU → target CPU(t+H)", [r["r_up_cpu"] for r in all_results])
    _stat_row("r  upstream MEM → target CPU(t+H)", [r["r_up_mem"] for r in all_results])
    _stat_row("rs upstream CPU (Spearman)", [r["rs_up_cpu"] for r in all_results])
    _stat_row(
        "r  autocorr target_t → target_tH", [r["r_autocorr"] for r in all_results]
    )
    logging.info("─" * 62)
    _stat_row("R² upstream [cpu+mem]", [r["r2_upstream"] for r in all_results])
    _stat_row("R² autocorr (naive baseline)", [r["r2_autocorr"] for r in all_results])
    _stat_row("R² combined [upstream + self]", [r["r2_combined"] for r in all_results])
    _stat_row("R² gain  (combined − autocorr)", [r["r2_gain"] for r in all_results])

    _interpret(all_results, args.pred_horizon)

    csv_path = os.path.join(args.out_dir, "summary.csv")
    fieldnames = [
        "dm",
        "n_um",
        "n_pairs",
        "r_up_cpu",
        "p_up_cpu",
        "r_up_mem",
        "p_up_mem",
        "r_autocorr",
        "rs_up_cpu",
        "rs_up_mem",
        "rs_autocorr",
        "r2_up_cpu_only",
        "r2_up_mem_only",
        "r2_upstream",
        "r2_autocorr",
        "r2_combined",
        "r2_gain",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    logging.info("Summary CSV saved → %s", csv_path)

    plot_correlation_distribution(
        all_results,
        args.pred_horizon,
        os.path.join(args.out_dir, "correlation_dist.png"),
    )
    plot_r2_comparison(
        all_results,
        args.pred_horizon,
        os.path.join(args.out_dir, "r2_comparison.png"),
    )
    plot_top_scatter(
        all_results,
        args.pred_horizon,
        os.path.join(args.out_dir, "top_scatter.png"),
        top_n=args.top_n,
        rng=rng,
    )

    logging.info("")
    logging.info("All outputs written to → %s", args.out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()

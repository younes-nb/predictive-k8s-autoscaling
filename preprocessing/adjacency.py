import os
import glob
import pickle
import logging
from typing import Dict, Set, List, Tuple

import polars as pl

from shared.config_paths import DATASET_TABLES

logger = logging.getLogger(__name__)


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
    logger.info("Scanning %d callgraph parquet files", len(files))

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
            logger.warning("  [SKIP parquet] %s: %s", os.path.basename(fp), exc)

        if i % 10 == 0:
            logger.info("  %d/%d parquet | DMs so far: %d", i, len(files), len(adj))

    return adj


def _scan_csv_for_pairs(raw_dir: str, valid: Set[str]) -> Dict[str, Set[str]]:
    files = sorted(
        glob.glob(os.path.join(raw_dir, "*.csv"))
        + glob.glob(os.path.join(raw_dir, "*.csv.gz"))
    )
    if not files:
        return {}

    adj: Dict[str, Set[str]] = {}
    logger.info("Scanning %d raw callgraph CSV files", len(files))

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
            logger.warning("  [SKIP CSV] %s: %s", os.path.basename(fp), exc)

        if i % 10 == 0:
            logger.info("  %d/%d CSV | DMs so far: %d", i, len(files), len(adj))

    return adj


def build_adjacency_map(
    valid_msnames: Set[str],
    cache_path: str,
    rebuild: bool = False,
) -> Dict[str, Set[str]]:
    if not rebuild and os.path.exists(cache_path):
        logger.info("Loading cached adjacency map from %s", cache_path)
        with open(cache_path, "rb") as f:
            adj = pickle.load(f)
        logger.info("Cached adjacency: %d DMs loaded", len(adj))
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
            logger.warning("Parquet scan yielded no pairs; falling back to raw CSVs.")
            adj = _scan_csv_for_pairs(raw_dir, valid_msnames)
    else:
        logger.info("No callgraph parquet found; using raw CSV files.")
        adj = _scan_csv_for_pairs(raw_dir, valid_msnames)

    logger.info("Adjacency complete: %d DMs with upstream callers", len(adj))

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(adj, f)
    logger.info("Adjacency cached to %s", cache_path)
    return adj

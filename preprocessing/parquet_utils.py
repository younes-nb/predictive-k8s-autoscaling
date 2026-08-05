import os
import glob
import json
import tempfile
from typing import Optional, List

import polars as pl

from shared.config_paths import PATHS

_CACHE_VERSION = 1


def list_parquet_parts(parquet_dir: str) -> list:
    return sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))


def build_table_agg(
    df_or_lazy,
    time_col: str,
    id_cols: list,
    freq: str,
    feature_exprs: list,
    agg_exprs: Optional[List[pl.Expr]] = None,
):
    if isinstance(df_or_lazy, pl.DataFrame):
        df_or_lazy = df_or_lazy.lazy()

    df_or_lazy = df_or_lazy.with_columns(pl.col(time_col).cast(pl.Datetime))

    if agg_exprs is None:
        agg_exprs = [
            pl.col(raw_col).last().alias(feat_name)
            for feat_name, raw_col in feature_exprs
        ]

    out = (
        df_or_lazy.with_columns(pl.col(time_col).dt.truncate(freq).alias("_t"))
        .group_by(["_t"] + id_cols)
        .agg(agg_exprs)
        .sort(["_t"] + id_cols)
    )
    return out


def _parquet_fingerprint(parquet_dir: str) -> list:
    fp = []
    for p in list_parquet_parts(parquet_dir):
        st = os.stat(p)
        fp.append({"name": os.path.basename(p), "size": st.st_size, "mtime": st.st_mtime})
    return fp


def _scan_services_from_parquet(parquet_dir: str, service_col: str) -> list:
    parts = list_parquet_parts(parquet_dir)
    df = (
        pl.scan_parquet(parts, low_memory=True)
        .select(service_col)
        .unique()
        .collect(engine="streaming")
    )
    return sorted(df[service_col].to_list())


def _load_cache(cache_file: str):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            return {}
        return data.get("entries", {})
    except (OSError, ValueError):
        return {}


def _save_cache(cache_file: str, entries: dict) -> None:
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    data = {"version": _CACHE_VERSION, "entries": entries}
    fd, tmp_path = tempfile.mkstemp(prefix=".service_cache.", dir=cache_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, cache_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def discover_unique_services(
    parquet_dir: str,
    service_col: str,
    cache_file: Optional[str] = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> list:
    if cache_file is None:
        cache_file = PATHS.SERVICE_CACHE_FILE

    fingerprint = _parquet_fingerprint(parquet_dir)
    cache_key = f"{parquet_dir}|{service_col}"

    if use_cache and not refresh:
        entries = _load_cache(cache_file)
        entry = entries.get(cache_key)
        if entry and entry.get("parts") == fingerprint:
            return list(entry.get("services", []))

    services = _scan_services_from_parquet(parquet_dir, service_col)

    if use_cache:
        entries = _load_cache(cache_file)
        entries[cache_key] = {
            "parquet_dir": parquet_dir,
            "service_col": service_col,
            "parts": fingerprint,
            "services": services,
        }
        _save_cache(cache_file, entries)

    return services

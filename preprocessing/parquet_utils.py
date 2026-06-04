import os
import glob
from typing import Optional, List

import polars as pl


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

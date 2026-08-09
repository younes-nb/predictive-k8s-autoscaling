import importlib

__all__ = [
    "main",
    "save_chunk",
    "list_parquet_parts",
    "build_table_agg",
]

_LAZY = {
    "main": "build_windows",
    "save_chunk": "build_windows",
    "list_parquet_parts": "parquet_utils",
    "build_table_agg": "parquet_utils",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY.keys()))

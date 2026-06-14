"""Inspect and print the contents of the training resume state file."""

import argparse
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CONFIG_PATHS_FILE = os.path.join(REPO_ROOT, "shared", "config_paths.py")
CONFIG_PATHS_SPEC = importlib.util.spec_from_file_location(
    "shared_config_paths",
    CONFIG_PATHS_FILE,
)
if CONFIG_PATHS_SPEC is None or CONFIG_PATHS_SPEC.loader is None:
    raise ImportError(f"Unable to load config paths from {CONFIG_PATHS_FILE}")

CONFIG_PATHS_MODULE = importlib.util.module_from_spec(CONFIG_PATHS_SPEC)
CONFIG_PATHS_SPEC.loader.exec_module(CONFIG_PATHS_MODULE)
PATHS = CONFIG_PATHS_MODULE.PATHS


SUMMARY_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "candidate_model_state_dict",
    "candidate_optimizer_state_dict",
}
EXPAND_KEYS = {"args", "hyperparams", "sfoa_hyperparams", "sfoa_state"}
DEFAULT_MAX_PREVIEW_ITEMS = 6


def _load_resume_state(path: str):
    if not os.path.exists(path):
        return None

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read the resume state file."
        ) from exc

    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load resume state from {path}: {exc}") from exc


def _normalize_mapping(value):
    if isinstance(value, argparse.Namespace):
        return vars(value)
    return value


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_scalar(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _format_tensor_like(value) -> str:
    module_name = value.__class__.__module__.split(".", 1)[0]
    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)

    if module_name == "torch":
        device = getattr(value, "device", None)
        return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

    if module_name == "numpy":
        return f"ndarray(shape={shape}, dtype={dtype})"

    return f"{value.__class__.__name__}(shape={shape}, dtype={dtype})"


def _format_sequence(value, max_items=DEFAULT_MAX_PREVIEW_ITEMS) -> str:
    items = list(value)
    if not items:
        return f"{type(value).__name__}([])"

    if all(_is_scalar(item) for item in items):
        preview = ", ".join(_format_scalar(item) for item in items[:max_items])
        if len(items) > max_items:
            preview += ", ..."

        if isinstance(value, tuple):
            if len(items) == 1:
                preview += ","
            return f"({preview})"
        return f"[{preview}]"

    preview = ", ".join(_format_value(item, max_items=max_items) for item in items[:max_items])
    if len(items) > max_items:
        preview += ", ..."
    return f"{type(value).__name__}(len={len(items)}, preview=[{preview}])"


def _format_mapping(value, max_items=DEFAULT_MAX_PREVIEW_ITEMS) -> str:
    keys = list(value.keys())
    preview = ", ".join(str(key) for key in keys[:max_items])
    if len(keys) > max_items:
        preview += ", ..."
    return f"dict(len={len(keys)}, keys=[{preview}])"


def _format_value(value, max_items=DEFAULT_MAX_PREVIEW_ITEMS) -> str:
    value = _normalize_mapping(value)

    if isinstance(value, Mapping):
        return _format_mapping(value, max_items=max_items)

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _format_tensor_like(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _format_sequence(value, max_items=max_items)

    return _format_scalar(value)


def _print_mapping(mapping, indent=0, expand_state_dicts=False):
    pad = " " * indent
    for key in sorted(mapping.keys(), key=str):
        value = _normalize_mapping(mapping[key])

        if isinstance(value, Mapping):
            if key in SUMMARY_KEYS and not expand_state_dicts:
                print(f"{pad}{key}: {_format_mapping(value)}")
            else:
                print(f"{pad}{key}:")
                _print_mapping(value, indent=indent + 2, expand_state_dicts=expand_state_dicts)
        else:
            print(f"{pad}{key}: {_format_value(value)}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the contents of the resume state file used by training and SFOA."
    )
    parser.add_argument(
        "--path",
        default=PATHS.RESUME_STATE_FILE,
        help="Path to the resume state .pt file.",
    )
    parser.add_argument(
        "--expand-state-dicts",
        action="store_true",
        help="Print the contents of model_state_dict and optimizer_state_dict too.",
    )
    args = parser.parse_args(argv)

    try:
        state = _load_resume_state(args.path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if state is None:
        print(f"[ERROR] Resume state file not found: {args.path}", file=sys.stderr)
        return 1

    state = _normalize_mapping(state)
    if not isinstance(state, Mapping):
        print(
            f"[ERROR] Expected a mapping in {args.path}, got {type(state).__name__}",
            file=sys.stderr,
        )
        return 1

    print(f"Resume state file: {args.path}")
    print(f"Top-level keys: {len(state)}")
    print("-" * 72)

    for key in sorted(state.keys(), key=str):
        value = _normalize_mapping(state[key])

        if key in EXPAND_KEYS and isinstance(value, Mapping):
            print(f"{key}:")
            _print_mapping(value, indent=2, expand_state_dicts=args.expand_state_dicts)
            continue

        if key in SUMMARY_KEYS and isinstance(value, Mapping) and not args.expand_state_dicts:
            print(f"{key}: {_format_mapping(value)}")
            continue

        if isinstance(value, Mapping):
            print(f"{key}:")
            _print_mapping(value, indent=2, expand_state_dicts=args.expand_state_dicts)
            continue

        print(f"{key}: {_format_value(value)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

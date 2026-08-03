import os
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, DATASET_TABLES
from shared.features import FEATURE_SETS, tables_for_feature_set
from shared.subprocess_utils import run


def main():
    ap = argparse.ArgumentParser(
        description="Preprocessing: fetch -> ingest(all needed tables) -> build_windows(join)"
    )
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")
    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN,
                     help="Sliding window input length passed to build_windows (default: %(default)s)")
    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_raw_windows", action="store_true",
                     help="Skip building raw sliding windows (build_windows.py)")
    ap.add_argument("--skip_preprocessing_approach", action="store_true",
                     help="Skip the preprocessing approach step (swt/cskv/smoothing)")
    ap.add_argument("--recompute_windows", action="store_true",
                     help="Recompute raw sliding windows, ignoring cached shards (build_windows.py)")
    ap.add_argument(
        "--delete_raw",
        action="store_false",
        dest="keep_raw",
    )
    ap.add_argument(
        "--max_services",
        type=int,
        default=PREPROCESSING.MAX_SERVICES,
    )
    ap.add_argument(
        "--preprocess_approach",
        default="swt",
        choices=["none", "smoothing", "swt", "cskv"],
        help="Post-processing approach applied after windows are built.",
    )
    ap.add_argument("--smooth_window", type=int, default=5, help="Smoothing window size (for 'smoothing' approach)")
    ap.add_argument("--dataset_workers", type=int, default=0, help="Workers for swt/cskv decomposition")
    ap.add_argument("--swt_level", type=int, default=None, help="SWT level for CPU (swt only, default: config)")
    ap.add_argument("--mem_swt_level", type=int, default=None, help="SWT level for memory (swt only, default: config)")
    ap.add_argument("--subset_seed", type=int, default=42, help="Seed for service subsampling in build_windows")
    ap.add_argument("--recompute_preprocessing", action="store_true",
                     help="Recompute the preprocessing approach output, ignoring cached shards")

    args = ap.parse_args()

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    print(f"feature_set={args.feature_set} => tables={needed_tables}")

    fetch_script = os.path.join(REPO_ROOT, "preprocessing", "fetch_traces.py")
    ingest_script = os.path.join(REPO_ROOT, "preprocessing", "ingest_traces_parquet.py")
    windows_script = os.path.join(REPO_ROOT, "preprocessing", "build_windows.py")
    smooth_script = os.path.join(REPO_ROOT, "preprocessing", "smooth_windows.py")
    swt_script = os.path.join(REPO_ROOT, "preprocessing", "swt", "preprocess.py")
    cskv_script = os.path.join(REPO_ROOT, "preprocessing", "cskv", "preprocess.py")

    if not args.skip_fetch:
        cmd = [
            sys.executable,
            fetch_script,
            "--start_date",
            args.start_date,
            "--end_date",
            args.end_date,
            "--feature_set",
            args.feature_set,
        ]
        run(cmd, "Step 1: Fetch")
    else:
        print("\n=== Skipping fetch ===")

    if not args.skip_ingest:
        for t in needed_tables:
            cfg = DATASET_TABLES[t]
            cmd = [
                sys.executable,
                ingest_script,
                "--table",
                t,
                "--feature_set",
                args.feature_set,
                "--raw_dir",
                cfg["raw_dir"],
                "--out_dir",
                cfg["parquet_dir"],
            ]
            if not args.keep_raw:
                cmd.append("--delete_raw")
            run(cmd, f"Step 2: Ingest table={t}")
    else:
        print("\n=== Skipping ingest ===")

    if not args.skip_raw_windows:
        cmd = [
            sys.executable,
            windows_script,
            "--out_dir",
            args.windows_dir,
            "--feature_set",
            args.feature_set,
        ]
        if args.max_services is not None:
            cmd.extend(["--max_services", str(args.max_services)])
        cmd.extend(["--subset_seed", str(args.subset_seed)])
        cmd.extend(["--input_len", str(args.input_len)])
        if args.recompute_windows:
            cmd.append("--recompute")
        run(cmd, "Step 3: Build windows (join tables)")
    else:
        print("\n=== Skipping raw windows ===")

    if not args.skip_preprocessing_approach:
        if args.preprocess_approach == "smoothing":
            cmd_smooth = [
                sys.executable, smooth_script,
                "--windows_dir", args.windows_dir,
                "--smooth_window", str(args.smooth_window),
            ]
            run(cmd_smooth, "Step 3b: Smoothing")
        elif args.preprocess_approach == "swt":
            swt_out = os.path.join(args.windows_dir, "swt")
            cmd_swt = [sys.executable, swt_script,
                       "--windows_dir", args.windows_dir,
                       "--out_dir", swt_out,
                       "--feature_set", args.feature_set]
            if args.swt_level is not None:
                cmd_swt.extend(["--swt_level", str(args.swt_level)])
            if args.mem_swt_level is not None:
                cmd_swt.extend(["--mem_swt_level", str(args.mem_swt_level)])
            if args.recompute_preprocessing:
                cmd_swt.append("--recompute_preprocessing")
            run(cmd_swt, "Step 3b: SWT Decomposition")
        elif args.preprocess_approach == "cskv":
            cskv_out = os.path.join(args.windows_dir, "cskv")
            cmd_cskv = [sys.executable, cskv_script,
                        "--windows_dir", args.windows_dir,
                        "--out_dir", cskv_out]
            if args.recompute_preprocessing:
                cmd_cskv.append("--recompute_preprocessing")
            run(cmd_cskv, "Step 3b: CSKV Decomposition")
    else:
        print("\n=== Skipping preprocessing approach ===")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

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
    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
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
        default="none",
        choices=["none", "smoothing", "sv", "cskv"],
        help="Post-processing approach applied after windows are built.",
    )
    ap.add_argument("--smooth_window", type=int, default=5, help="Smoothing window size (for 'smoothing' approach)")
    ap.add_argument("--dataset_workers", type=int, default=0, help="Workers for sv/cskv decomposition")
    ap.add_argument("--subset_seed", type=int, default=42, help="Seed for service subsampling in build_windows")

    args = ap.parse_args()

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    print(f"feature_set={args.feature_set} => tables={needed_tables}")

    fetch_script = os.path.join(REPO_ROOT, "preprocessing", "fetch_traces.py")
    ingest_script = os.path.join(REPO_ROOT, "preprocessing", "ingest_traces_parquet.py")
    windows_script = os.path.join(REPO_ROOT, "preprocessing", "build_windows.py")
    smooth_script = os.path.join(REPO_ROOT, "preprocessing", "smooth_windows.py")
    sv_script = os.path.join(REPO_ROOT, "preprocessing", "sv", "preprocess.py")
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

    if not args.skip_windows:
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
        run(cmd, "Step 3: Build windows (join tables)")

        if args.preprocess_approach == "smoothing":
            cmd_smooth = [
                sys.executable, smooth_script,
                "--windows_dir", args.windows_dir,
                "--smooth_window", str(args.smooth_window),
            ]
            run(cmd_smooth, "Step 3b: Smoothing")
        elif args.preprocess_approach == "sv":
            cmd_sv = [sys.executable, sv_script]
            run(cmd_sv, "Step 3b: SV Decomposition")
        elif args.preprocess_approach == "cskv":
            cmd_cskv = [sys.executable, cskv_script]
            run(cmd_cskv, "Step 3b: CSKV Decomposition")
    else:
        print("\n=== Skipping windows ===")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

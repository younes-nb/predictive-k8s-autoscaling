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
    ap.add_argument("--skip_raw_windows", action="store_true",
                     help="Skip building raw sliding windows (build_windows.py)")
    ap.add_argument("--skip_preprocessing_approach", action="store_true",
                     help="Skip the preprocessing approach step (sv/cskv/smoothing)")
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
    ap.add_argument("--swt_level", type=int, default=None, help="SWT level for CPU (sv only, default: config)")
    ap.add_argument("--mem_swt_level", type=int, default=None, help="SWT level for memory (sv only, default: config)")
    ap.add_argument("--no_vmd", action="store_true",
                     help="Skip VMD decomposition; use only SWT coefficients (sv only)")
    ap.add_argument("--vmd_k", type=int, default=None,
                     help="VMD K (modes) for CPU (sv only, default: config)")
    ap.add_argument("--mem_vmd_k", type=int, default=None,
                     help="VMD K for memory (sv only, default: config)")
    ap.add_argument("--vmd_swt_level", type=int, default=None,
                     help="SWT detail level fed into VMD for CPU (sv only, default: config)")
    ap.add_argument("--mem_vmd_swt_level", type=int, default=None,
                     help="SWT detail level fed into VMD for memory (sv only, default: config)")
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
        elif args.preprocess_approach == "sv":
            sv_out = os.path.join(args.windows_dir, "sv")
            cmd_sv = [sys.executable, sv_script,
                      "--windows_dir", args.windows_dir,
                      "--out_dir", sv_out,
                      "--feature_set", args.feature_set]
            if args.swt_level is not None:
                cmd_sv.extend(["--swt_level", str(args.swt_level)])
            if args.mem_swt_level is not None:
                cmd_sv.extend(["--mem_swt_level", str(args.mem_swt_level)])
            if args.no_vmd:
                cmd_sv.append("--no_vmd")
            if args.vmd_k is not None:
                cmd_sv.extend(["--vmd_k", str(args.vmd_k)])
            if args.mem_vmd_k is not None:
                cmd_sv.extend(["--mem_vmd_k", str(args.mem_vmd_k)])
            if args.vmd_swt_level is not None:
                cmd_sv.extend(["--vmd_swt_level", str(args.vmd_swt_level)])
            if args.mem_vmd_swt_level is not None:
                cmd_sv.extend(["--mem_vmd_swt_level", str(args.mem_vmd_swt_level)])
            if args.recompute_preprocessing:
                cmd_sv.append("--recompute_preprocessing")
            run(cmd_sv, "Step 3b: SV Decomposition")
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

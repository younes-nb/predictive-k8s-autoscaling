import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Run full preprocessing pipeline"
    )
    ap.add_argument("--start_date", required=True, help="e.g. 0d0")
    ap.add_argument("--end_date", required=True, help="e.g. 1d0")

    ap.add_argument(
        "--raw_dir",
        default="/dataset1/alibaba_v2022/raw/msresource",
    )
    ap.add_argument(
        "--parquet_dir",
        default="/dataset1/alibaba_v2022/parquet/msresource"
    )
    ap.add_argument(
        "--windows_dir",
        default="/dataset2/alibaba_v2022/windows"
    )

    ap.add_argument("--input_len", type=int, default=60)
    ap.add_argument("--pred_horizon", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--smoothing_window", type=int, default=5)

    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    fetch_script = os.path.join(repo_root, "scripts", "fetch_traces.py")
    ingest_script = os.path.join(repo_root, "scripts", "ingest_traces_parquet.py")
    windows_script = os.path.join(repo_root, "scripts", "build_windows.py")

    cmd_fetch = [
        sys.executable, fetch_script,
        "--start_date", args.start_date,
        "--end_date", args.end_date,
        "--table", "msresource",
        "--raw_dir", args.raw_dir,
    ]
    print("=== Step 1: Fetch Data ===")
    print("Running:", " ".join(cmd_fetch))
    subprocess.run(cmd_fetch, check=True)

    os.makedirs(args.parquet_dir, exist_ok=True)
    cmd_ingest = [
        sys.executable, ingest_script,
        "--raw_dir", args.raw_dir,
        "--out_dir", args.parquet_dir,
    ]
    print("=== Step 2: Ingest CSV -> Parquet ===")
    print("Running:", " ".join(cmd_ingest))
    subprocess.run(cmd_ingest, check=True)

    os.makedirs(args.windows_dir, exist_ok=True)
    cmd_windows = [
        sys.executable, windows_script,
        "--parquet_dir", args.parquet_dir,
        "--out_dir", args.windows_dir,
        "--input_len", str(args.input_len),
        "--pred_horizon", str(args.pred_horizon),
        "--stride", str(args.stride),
        "--train_frac", str(args.train_frac),
        "--val_frac", str(args.val_frac),
        "--smoothing_window", str(args.smoothing_window),
    ]
    print("=== Step 3: Build windows ===")
    print("Running:", " ".join(cmd_windows))
    subprocess.run(cmd_windows, check=True)

    print("=== Preprocessing pipeline completed successfully ===")


if __name__ == "__main__":
    main()

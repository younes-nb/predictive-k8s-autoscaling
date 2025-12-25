import argparse
import os
import sys
import urllib.request
import tarfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    DATASET_TABLES,
    PREPROCESSING,
    FEATURE_SETS,
    tables_for_feature_set,
)

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"


def parse_dh(s: str):
    if "d" not in s:
        raise ValueError("Bad time spec: %r (expected like '0d0' or '1d12')" % s)
    d_str, h_str = s.split("d", 1)
    return int(d_str), int(h_str)


def compute_indices(start_date: str, end_date: str, ratio_min: int):
    sd, sh = parse_dh(start_date)
    ed, eh = parse_dh(end_date)

    start_min = sd * 24 * 60 + sh * 60
    end_min = ed * 24 * 60 + eh * 60
    if end_min <= start_min:
        raise ValueError("end_date must be after start_date")

    start_idx = start_min // ratio_min
    end_idx = end_min // ratio_min - 1
    if end_idx < start_idx:
        end_idx = start_idx
    return start_idx, end_idx


def download_file(url: str, dst_path: str) -> bool:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"Downloading {url} -> {dst_path}")
    try:
        with urllib.request.urlopen(url) as resp, open(dst_path, "wb") as out:
            chunk_size = 1 << 20
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        print(f"[WARN] Failed to download {url}: {e}", file=sys.stderr)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False
    return True


def extract_and_remove_tar(tar_path: str, out_dir: str) -> bool:
    print(f"Extracting {tar_path} into {out_dir}")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            try:
                tar.extractall(path=out_dir, filter="data")
            except TypeError:
                tar.extractall(path=out_dir)
        os.remove(tar_path)
        return True
    except Exception as e:
        print(f"[WARN] Failed to extract {tar_path}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Download and extract Alibaba v2022 trace chunks."
    )
    ap.add_argument("--start_date", default="0d0", help="e.g. 0d0")
    ap.add_argument("--end_date", default="7d0", help="e.g. 7d0")

    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="Which feature set to prepare (controls which tables are fetched).",
    )

    ap.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Optional explicit table list (overrides --feature_set).",
    )

    args = ap.parse_args()

    if args.tables is None:
        needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    else:
        needed_tables = list(args.tables)

    print(f"Feature set: {args.feature_set}")
    print(f"Tables to fetch: {needed_tables}")
    print(f"Range: {args.start_date} -> {args.end_date}")

    for table in needed_tables:
        if table not in DATASET_TABLES:
            raise SystemExit(
                f"Unknown table '{table}'. Add it to DATASET_TABLES in config/defaults.py"
            )

        cfg = DATASET_TABLES[table]
        raw_dir = cfg["raw_dir"]
        os.makedirs(raw_dir, exist_ok=True)

        try:
            start_idx, end_idx = compute_indices(
                args.start_date, args.end_date, int(cfg["ratio_min"])
            )
        except ValueError as e:
            print("ERROR:", e, file=sys.stderr)
            sys.exit(2)

        print(f"\n=== TABLE {table} ===")
        print(f"prefix={cfg['prefix']}")
        print(f"indices: {start_idx} .. {end_idx}")
        print(f"raw_dir: {raw_dir}")

        for idx in range(start_idx, end_idx + 1):
            url = f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"
            tar_path = os.path.join(raw_dir, f"{table}_{idx}.tar.gz")

            if not os.path.exists(tar_path):
                ok = download_file(url, tar_path)
                if not ok:
                    continue
            else:
                print(f"Tar already exists, reusing: {tar_path}")

            ok = extract_and_remove_tar(tar_path, raw_dir)
            if not ok:
                continue

    print("\nDone downloading and extracting.")


if __name__ == "__main__":
    main()



import argparse
import os
import sys
import urllib.request
import tarfile

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"

TABLE_CONFIG = {
    "msresource": {
        "prefix": "MSMetricsUpdate/MSMetricsUpdate",
        "ratio_min": 30,
        "default_raw_dir": "/dataset1/alibaba_v2022/raw/msresource",
    },
}


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


def download_file(url: str, dst_path: str):
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


def extract_and_remove_tar(tar_path: str, out_dir: str):
    print(f"Extracting {tar_path} into {out_dir}")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
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
    ap.add_argument("--start_date", required=True, help="e.g. 0d0")
    ap.add_argument("--end_date", required=True, help="e.g. 1d0")
    ap.add_argument(
        "--table",
        default="msresource",
    )
    ap.add_argument(
        "--raw_dir",
        default=None,
    )
    args = ap.parse_args()

    if args.table not in TABLE_CONFIG:
        print(f"ERROR: table '{args.table}' not in TABLE_CONFIG.", file=sys.stderr)
        sys.exit(2)

    cfg = TABLE_CONFIG[args.table]
    raw_dir = args.raw_dir or cfg["default_raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    try:
        start_idx, end_idx = compute_indices(
            args.start_date, args.end_date, cfg["ratio_min"]
        )
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

    print(f"Table: {args.table}")
    print(f"Indices: {start_idx} .. {end_idx}")
    print(f"Raw dir: {raw_dir}")

    for idx in range(start_idx, end_idx + 1):
        url = f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"
        tar_path = os.path.join(raw_dir, f"{args.table}_{idx}.tar.gz")

        if not os.path.exists(tar_path):
            ok = download_file(url, tar_path)
            if not ok:
                continue
        else:
            print(f"Tar already exists, reusing: {tar_path}")

        ok = extract_and_remove_tar(tar_path, raw_dir)
        if not ok:
            continue

    print("Done downloading and extracting.")


if __name__ == "__main__":
    main()

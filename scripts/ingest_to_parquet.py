import polars as pl, glob, os, argparse

p = argparse.ArgumentParser()
p.add_argument("--raw_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--repartition", type=int, default=4)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

files = sorted(
    glob.glob(os.path.join(args.raw_dir, "*.csv")) +
    glob.glob(os.path.join(args.raw_dir, "*.csv.gz"))
)

if not files:
    raise SystemExit(f"No CSV/CSV.GZ files found under {args.raw_dir}")

batches = []
scan_kwargs = dict(low_memory=True, try_parse_dates=False, infer_schema_length=50000)

for f in files:
    df = pl.read_csv(f, **scan_kwargs)

    needed = {"timestamp", "msname", "msinstanceid", "cpu_utilization"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[WARN] {f} missing columns {missing}; skipping.")
        continue

    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp") / 1000, time_unit="s").alias("timestamp_dt")
    )

    df = df.select(
        "timestamp", "timestamp_dt", "msname", "msinstanceid", "cpu_utilization"
    )

    batches.append(df)

if not batches:
    raise SystemExit("No usable CSV batches after validation")

big = pl.concat(batches, rechunk=True)

big = big.sort(["timestamp_dt", "msname", "msinstanceid"])

big = big.rechunk()
npart = max(1, args.repartition)
rows = big.height
rows_per = (rows + npart - 1) // npart

for i in range(npart):
    start = i * rows_per
    if start >= rows:
        break
    end = min(rows, start + rows_per)
    out = os.path.join(args.out_dir, f"part-{i:05d}.parquet")
    big.slice(start, end - start).write_parquet(out, compression="zstd")

print("Wrote parquet parts to", args.out_dir)

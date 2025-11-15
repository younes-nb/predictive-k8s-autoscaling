import os, glob, argparse
import polars as pl
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--parquet_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--time_col", default="timestamp_dt")
p.add_argument("--target_col", default="cpu_utilization")
p.add_argument("--id_cols", nargs="+", default=["msname", "msinstanceid"])
p.add_argument("--freq", default="1m")
p.add_argument("--input_len", type=int, default=60)
p.add_argument("--pred_horizon", type=int, default=5)
p.add_argument("--stride", type=int, default=1)
p.add_argument("--train_frac", type=float, default=0.7)
p.add_argument("--val_frac", type=float, default=0.15)
args = p.parse_args()

if args.train_frac <= 0 or args.val_frac < 0:
    raise SystemExit("train_frac must be > 0 and val_frac must be >= 0")

if args.train_frac + args.val_frac >= 1.0:
    raise SystemExit("train_frac + val_frac must be < 1.0")

os.makedirs(args.out_dir, exist_ok=True)

parts = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
if not parts:
    raise SystemExit("No parquet found")

def windowize(arr, in_len, horizon, stride):
    X, y = [], []
    N = len(arr)
    for i in range(0, N - in_len - horizon + 1, stride):
        X.append(arr[i:i+in_len])
        y.append(arr[i+in_len:i+in_len+horizon])
    return np.asarray(X), np.asarray(y)

all_train_X, all_train_y = [], []
all_val_X, all_val_y = [], []
all_test_X, all_test_y = [], []

for pq in parts:
    df = pl.read_parquet(pq)

    if df[args.time_col].dtype != pl.Datetime:
        df = df.with_columns(pl.col(args.time_col).cast(pl.Datetime))

    df = df.with_columns(
        pl.col(args.time_col).dt.truncate(args.freq).alias("_t")
    ).group_by("_t", *args.id_cols).agg(
        pl.col(args.target_col).last().alias(args.target_col)
    ).sort(["_t", *args.id_cols])

    t0 = df["_t"].min()
    df = df.with_columns(
        ((pl.col("_t") - t0).dt.total_minutes().round(0).cast(pl.Int32)).alias("_minute")
    )

    min_minute = int(df["_minute"].min())
    max_minute = int(df["_minute"].max())
    span = max_minute - min_minute + 1

    train_end = min_minute + int(span * args.train_frac)
    val_end   = min_minute + int(span * (args.train_frac + args.val_frac))

    train_end = max(train_end, min_minute + 1)
    val_end   = max(val_end, train_end + 1)
    val_end   = min(val_end, max_minute + 1)

    for _, g in df.group_by(args.id_cols, maintain_order=True):
        g = g.sort("_minute")
        arr = g[args.target_col].to_numpy()

        tr = g.filter(pl.col("_minute") < train_end)
        vl = g.filter((pl.col("_minute") >= train_end) & (pl.col("_minute") < val_end))
        te = g.filter(pl.col("_minute") >= val_end)

        def windows_from_frame(fr):
            a = fr[args.target_col].to_numpy()
            if len(a) < args.input_len + args.pred_horizon + 5:
                return None, None
            return windowize(a, args.input_len, args.pred_horizon, args.stride)

        trX, trY = windows_from_frame(tr)
        vlX, vlY = windows_from_frame(vl)
        teX, teY = windows_from_frame(te)

        if trX is not None: all_train_X.append(trX); all_train_y.append(trY)
        if vlX is not None: all_val_X.append(vlX); all_val_y.append(vlY)
        if teX is not None: all_test_X.append(teX); all_test_y.append(teY)

def cat_or_empty(chunks, shape):
    if not chunks: return np.empty(shape)
    return np.concatenate(chunks, axis=0)

Xtr = cat_or_empty(all_train_X, (0, args.input_len))
ytr = cat_or_empty(all_train_y, (0, args.pred_horizon))
Xv  = cat_or_empty(all_val_X,   (0, args.input_len))
yv  = cat_or_empty(all_val_y,   (0, args.pred_horizon))
Xte = cat_or_empty(all_test_X,  (0, args.input_len))
yte = cat_or_empty(all_test_y,  (0, args.pred_horizon))

np.save(os.path.join(args.out_dir, "X_train.npy"), Xtr)
np.save(os.path.join(args.out_dir, "y_train.npy"), ytr)
np.save(os.path.join(args.out_dir, "X_val.npy"),   Xv)
np.save(os.path.join(args.out_dir, "y_val.npy"),   yv)
np.save(os.path.join(args.out_dir, "X_test.npy"),  Xte)
np.save(os.path.join(args.out_dir, "y_test.npy"),  yte)

print("Saved shapes:")
print("  train:", Xtr.shape, ytr.shape)
print("  val:  ", Xv.shape,  yv.shape)
print("  test: ", Xte.shape, yte.shape)

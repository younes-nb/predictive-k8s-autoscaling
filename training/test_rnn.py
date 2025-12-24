import os
import sys
import glob
import time
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING, DEFAULT_CHECKPOINT_PATH


class ShardedWindowsDataset(Dataset):
    def __init__(self, windows_dir: str, split: str, input_len: int, horizon: int):
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.split = split

        self.shards = []
        self.lengths = []
        self.cum_lengths = []

        pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(pattern))
        if not x_files:
            raise RuntimeError(f"No X_{split}.npy files found under {windows_dir}")

        total = 0
        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"
            if not os.path.exists(y_path):
                raise RuntimeError(f"Missing matching y file for {x_path}: expected {y_path}")

            X = np.load(x_path, mmap_mode="r")
            Y = np.load(y_path, mmap_mode="r")

            if X.shape[0] != Y.shape[0]:
                raise RuntimeError(f"Shape mismatch between {x_path} and {y_path}")

            if X.ndim == 2:
                if X.shape[1] != self.input_len:
                    print(f"[WARN] {x_path} has shape {X.shape}, expected (*, {self.input_len})")
            elif X.ndim == 3:
                if X.shape[1] != self.input_len:
                    print(f"[WARN] {x_path} has shape {X.shape}, expected (*, {self.input_len}, F)")
            else:
                raise RuntimeError(f"[ERROR] {x_path} has unsupported ndim={X.ndim} (shape={X.shape})")

            if Y.ndim != 2 or Y.shape[1] != self.horizon:
                print(f"[WARN] {y_path} has shape {Y.shape}, expected (*, {self.horizon})")

            self.shards.append((X, Y))
            self.lengths.append(X.shape[0])
            total += X.shape[0]
            self.cum_lengths.append(total)

        self.total_len = total
        print(f"[{split}] Loaded {len(self.shards)} shard(s), total windows: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        for shard_idx, cum in enumerate(self.cum_lengths):
            if idx < cum:
                prev_cum = 0 if shard_idx == 0 else self.cum_lengths[shard_idx - 1]
                local_idx = idx - prev_cum
                X, Y = self.shards[shard_idx]

                x_arr = np.array(X[local_idx], copy=True)
                y_arr = np.array(Y[local_idx], copy=True)

                if x_arr.ndim == 1:
                    x_arr = x_arr[:, None] 
                elif x_arr.ndim == 2:
                    pass 
                else:
                    raise RuntimeError(f"Unexpected x_arr ndim={x_arr.ndim}, shape={x_arr.shape}")

                x_tensor = torch.from_numpy(x_arr).float()
                y_tensor = torch.from_numpy(y_arr).float()
                return x_tensor, y_tensor

        raise IndexError(idx)


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)

        if x.dim() != 3:
            raise ValueError(f"RNN: Expected input 3D (B,T,F), got {x.dim()}D instead (shape={tuple(x.shape)})")

        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def evaluate_test(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Using device:", device)

    torch.manual_seed(args.seed)

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(PATHS.LOGS_DIR, f"test_{run_ts}.log")
    print(f"Test summary will be logged to: {log_path}")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", checkpoint.get("hparams", {}))

    hidden_size = int(ckpt_args.get("hidden_size", TRAINING.HIDDEN_SIZE))
    num_layers = int(ckpt_args.get("num_layers", TRAINING.NUM_LAYERS))
    dropout = float(ckpt_args.get("dropout", TRAINING.DROPOUT))
    pred_horizon = int(ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON))
    input_len = int(ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN))
    ckpt_rnn_type = str(ckpt_args.get("rnn_type", "lstm")).lower()

    if args.rnn_type is not None:
        if args.rnn_type.lower() != ckpt_rnn_type:
            raise SystemExit(
                f"--rnn_type={args.rnn_type} does not match checkpoint rnn_type={ckpt_rnn_type}. "
                f"Use the same rnn_type used during training."
            )
    rnn_type = ckpt_rnn_type

    with open(log_path, "a") as f:
        f.write(f"Run timestamp: {run_ts}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Windows dir: {args.windows_dir}\n")
        f.write(f"Checkpoint path: {args.checkpoint_path}\n")
        f.write(f"Loaded hyperparams from checkpoint: {ckpt_args}\n")
        f.write(f"Effective input_len={input_len}, pred_horizon={pred_horizon}, rnn_type={rnn_type}\n")
        f.write(f"class_threshold={args.class_threshold}\n")
        f.write("-" * 60 + "\n")

    print("Loading test dataset from:", args.windows_dir)
    print(f"  input_len={input_len}, horizon={pred_horizon}, rnn_type={rnn_type}, class_threshold={args.class_threshold}")

    test_dataset = ShardedWindowsDataset(
        windows_dir=args.windows_dir,
        split="test",
        input_len=input_len,
        horizon=pred_horizon,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    input_size = ckpt_args.get("input_size", None)
    if input_size is None:
        first_X, _ = test_dataset[0]
        input_size = int(first_X.shape[1]) 
    input_size = int(input_size)

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=pred_horizon,
        rnn_type=rnn_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model for testing:")
    print(model)
    print("-" * 80)

    mse_sum = 0.0
    mae_sum = 0.0
    total_elems = 0

    tp = fp = tn = fn = 0

    start_test = time.time()
    thr = float(args.class_threshold)

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader, start=1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch) 

            diff = preds - y_batch
            mse_sum += torch.sum(diff * diff).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            total_elems += y_batch.numel()

            y_true = (y_batch >= thr)
            y_pred = (preds >= thr)

            tp += torch.sum(y_pred & y_true).item()
            fp += torch.sum(y_pred & (~y_true)).item()
            fn += torch.sum((~y_pred) & y_true).item()
            tn += torch.sum((~y_pred) & (~y_true)).item()

            if batch_idx % args.log_interval == 0 or batch_idx == len(test_loader):
                running_mse = mse_sum / max(1, total_elems)
                elapsed = time.time() - start_test
                print(
                    f"[Test] Batch {batch_idx:06d}/{len(test_loader):06d} "
                    f"running_mse={running_mse:.6f} elapsed={elapsed:.2f}s"
                )

    test_time = time.time() - start_test
    if total_elems == 0:
        raise RuntimeError("Test set produced 0 elements; please check your windows.")

    test_mse = mse_sum / total_elems
    test_mae = mae_sum / total_elems

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    test_summary = (
        f"TEST SUMMARY: "
        f"mse={test_mse:.6f}, "
        f"mae={test_mae:.6f}, "
        f"test_time={test_time:.2f}s, "
        f"num_windows={len(test_dataset)}, "
        f"batch_size={args.batch_size}"
    )

    cls_summary = (
        f"CLASSIFICATION (thr={thr}): "
        f"acc={accuracy:.6f}, "
        f"prec={precision:.6f}, "
        f"recall={recall:.6f}, "
        f"f1={f1:.6f}, "
        f"tp={int(tp)}, fp={int(fp)}, tn={int(tn)}, fn={int(fn)}"
    )

    print("\n" + test_summary)
    print(cls_summary)

    with open(log_path, "a") as f:
        f.write(test_summary + "\n")
        f.write(cls_summary + "\n")

    infer_msg = "Inference benchmark could not be run (empty test set)."
    if len(test_dataset) > 0:
        sample_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        sample_X, _ = next(iter(sample_loader))
        sample_X = sample_X.to(device)

        with torch.no_grad():
            for _ in range(5):
                _ = model(sample_X)

        repeats = int(args.inference_repeats)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(sample_X)
        t1 = time.perf_counter()

        total_inf_time = t1 - t0
        avg_batch_time = total_inf_time / repeats
        avg_sample_time = avg_batch_time / sample_X.size(0)

        infer_msg = (
            "INFERENCE BENCHMARK: "
            f"batch_size={sample_X.size(0)}, "
            f"repeats={repeats}, "
            f"avg_batch_time={avg_batch_time:.6f}s, "
            f"avg_sample_time={avg_sample_time*1000:.6f}ms"
        )

        print(infer_msg)
        with open(log_path, "a") as f:
            f.write(infer_msg + "\n")
    else:
        print(infer_msg)
        with open(log_path, "a") as f:
            f.write(infer_msg + "\n")

    final_msg = "Test run complete."
    print(final_msg)
    with open(log_path, "a") as f:
        f.write(final_msg + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RNN (LSTM/GRU) model on test set (CPU).")
    p.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)

    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)

    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--log_interval", type=int, default=TRAINING.LOG_INTERVAL)
    p.add_argument("--inference_repeats", type=int, default=TRAINING.INFERENCE_REPEATS)

    p.add_argument(
        "--rnn_type",
        choices=["lstm", "gru"],
        default=None,
        help="Must match checkpoint rnn_type; used as a safety check.",
    )
    p.add_argument(
        "--class_threshold",
        type=float,
        default=0.8,
        help="Threshold on (normalized) cpu utilization to compute Accuracy/Precision/Recall/F1.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_test(args)

import os
import sys
import time
import math
import argparse
import subprocess
import threading

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from shared.config_training_defaults import TRAINING
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import target_features_for_feature_set
from core.dataset import ShardedWindowsDataset
from core.models import RNNForecaster


GPU_UTIL_SAMPLES = []
_util_stop = threading.Event()


def _poll_gpu_util(interval: float = 0.1):
    while not _util_stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            vals = [int(v.strip()) for v in out.stdout.strip().split("\n") if v.strip()]
            if vals:
                GPU_UTIL_SAMPLES.append(max(vals))
        except Exception:
            pass
        _util_stop.wait(interval)


def measure_throughput(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    warmup_steps: int = 3,
    measure_steps: int = 10,
    pin_memory: bool = True,
    mode: str = "train",
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if mode == "eval":
        model.eval()
    else:
        model.train()

    optimizer = None
    if mode == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    global GPU_UTIL_SAMPLES
    GPU_UTIL_SAMPLES = []

    poller = threading.Thread(target=_poll_gpu_util, daemon=True)
    poller.start()

    step = 0
    throughputs = []

    def loss_fn(preds, y):
        return ((preds - y) ** 2).mean()

    for batch in loader:
        x = batch[0].to(device, non_blocking=pin_memory)
        y = batch[1].to(device, non_blocking=pin_memory)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            if mode == "eval":
                with torch.no_grad():
                    start.record()
                    _ = model(x)
                    end.record()
            else:
                optimizer.zero_grad()
                start.record()
                preds = model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                end.record()
                optimizer.step()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            if mode == "eval":
                with torch.no_grad():
                    _ = model(x)
            else:
                optimizer.zero_grad()
                preds = model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
            elapsed = time.perf_counter() - t0

        step += 1
        if step > warmup_steps:
            throughputs.append(x.size(0) / elapsed)

        if step >= warmup_steps + measure_steps:
            break

    _util_stop.set()
    poller.join(timeout=2)

    mem_allocated = torch.cuda.max_memory_allocated(device) / 1e6
    torch.cuda.empty_cache()

    if not throughputs:
        return {"samples_per_sec": 0, "gpu_mem_mb": mem_allocated, "gpu_util_pct": 0}

    avg_util = int(np.mean(GPU_UTIL_SAMPLES)) if GPU_UTIL_SAMPLES else 0

    return {
        "samples_per_sec": round(np.mean(throughputs), 1),
        "gpu_mem_mb": round(mem_allocated, 1),
        "gpu_util_pct": avg_util,
    }


def run_smoke_test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB total)")
    print(f"Dataset: {args.windows_dir}")
    print()

    dataset = ShardedWindowsDataset(
        args.windows_dir, "train",
        args.input_len, args.pred_horizon,
        use_weights=False,
    )
    total = len(dataset)
    n_samples = min(args.num_samples, total)
    if n_samples < total:
        dataset = Subset(dataset, range(n_samples))
    print(f"Using {n_samples} / {total} training samples")

    if len(dataset) > 0:
        first_x, *_ = dataset[0]
        input_size = first_x.shape[-1]
    else:
        raise RuntimeError("Dataset is empty")

    num_targets = len(target_features_for_feature_set(args.feature_set))

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        quantiles=None,
        num_targets=num_targets,
    ).to(device)

    print(f"Model: {args.rnn_type} hidden={args.hidden_size} layers={args.num_layers}")
    print()

    batch_sizes = args.batch_sizes or [512, 1024, 2048, 4096, 8192, 16384, 32768]
    if args.all_workers:
        worker_counts = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    else:
        worker_counts = args.worker_counts or [0, 2, 4, 8, 16, 32]

    header = f"{'batch_size':>10} {'workers':>8} {'samples/s':>10} {'GPU_mem(MB)':>12} {'GPU_util%':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for bs in batch_sizes:
        for nw in worker_counts:
            if nw > 0 and nw > os.cpu_count():
                continue
            if bs * args.measure > n_samples:
                continue
            try:
                m = measure_throughput(
                    model, dataset, bs, nw, device,
                    warmup_steps=args.warmup,
                    measure_steps=args.measure,
                    pin_memory=(device.type != "cpu"),
                    mode=args.mode,
                )
                print(f"{bs:>10} {nw:>8} {m['samples_per_sec']:>10.1f} {m['gpu_mem_mb']:>12.1f} {m['gpu_util_pct']:>8}%")
                results.append((bs, nw, m))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{bs:>10} {nw:>8} {'OOM':>10} {'-':>12} {'-':>8}")
                else:
                    raise e
            except Exception as e:
                print(f"{bs:>10} {nw:>8} {'ERR':>10}")

    print()
    if results:
        best = max(results, key=lambda r: r[2]["samples_per_sec"])
        print(f"Best throughput: batch_size={best[0]}, workers={best[1]}, "
              f"{best[2]['samples_per_sec']} samples/sec, "
              f"mem={best[2]['gpu_mem_mb']}MB, util={best[2]['gpu_util_pct']}%")

        mem_efficient = [r for r in results if r[2]["gpu_mem_mb"] < args.mem_limit_mb]
        if mem_efficient:
            best_eff = max(mem_efficient, key=lambda r: r[2]["samples_per_sec"])
            print(f"Best within {args.mem_limit_mb}MB budget: "
                  f"batch_size={best_eff[0]}, workers={best_eff[1]}, "
                  f"{best_eff[2]['samples_per_sec']} samples/sec")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", default="/dataset/windows")
    p.add_argument("--num_samples", type=int, default=100000,
                   help="Number of dataset samples to use for profiling")
    p.add_argument("--batch_sizes", type=int, nargs="+",
                   help="Batch sizes to test (default: 512 1024 2048 4096 8192 16384 32768)")
    p.add_argument("--worker_counts", type=int, nargs="+",
                   help="num_workers values to test (default: 0 2 4 8 16 32)")
    p.add_argument("--all_workers", action="store_true",
                   help="Test wide range: 0 2 4 8 12 16 24 32 48 64")
    p.add_argument("--mode", default="train", choices=["train", "eval"],
                   help="Train mode (fwd+bwd) or eval only (default: train)")
    p.add_argument("--warmup", type=int, default=3, help="Warmup steps per config")
    p.add_argument("--measure", type=int, default=10, help="Measurement steps per config")
    p.add_argument("--mem_limit_mb", type=int, default=30000,
                   help="GPU memory budget in MB (default: 30000 = ~75% of A100-40GB)")
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--hidden_size", type=int, default=TRAINING.HIDDEN_SIZE)
    p.add_argument("--num_layers", type=int, default=TRAINING.NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL)
    args = p.parse_args()
    run_smoke_test(args)


if __name__ == "__main__":
    main()

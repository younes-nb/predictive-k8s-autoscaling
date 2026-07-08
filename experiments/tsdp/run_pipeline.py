import argparse
import os
import subprocess
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_MAP = {
    "patchtst": "patchtst",
    "cnn_bilstm": "cnn_bilstm",
    "tcn_bigru": "tcn_bigru",
}


def run_step(cmd: list, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"=== {label} ===")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"=== {label} completed in {elapsed:.1f}s ===\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="TSDP end-to-end pipeline")
    ap.add_argument("--arch", choices=list(ARCH_MAP.keys()), default="patchtst",
                    help="Architecture: patchtst | cnn_bilstm | tcn_bigru")
    ap.add_argument("--preprocess_dir", default="/dataset/tsdp_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/tsdp")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/tsdp")
    ap.add_argument("--max_services", type=int, default=0,
                    help="Maximum services to process (default: 0 = all)")
    ap.add_argument("--epochs_override", type=int, default=None,
                    help="Override CFG.EPOCHS for quick testing")
    ap.add_argument("--skip_preprocess", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dataset_workers", type=int, default=max(1, int(os.cpu_count() * 0.7)),
                    help="Workers for parallel dataset loading (0 = sequential)")
    args = ap.parse_args()

    arch_dir = os.path.join(THIS_DIR, ARCH_MAP[args.arch])

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.cpu or gpu_count <= 1:
        launcher_cmd = [sys.executable]
        if gpu_count == 1 and not args.cpu:
            print("[INFO] 1 GPU detected. Running with standard python invocation.")
        elif gpu_count == 0 or args.cpu:
            print("[INFO] Running strictly on CPU.")
    else:
        launcher_cmd = ["accelerate", "launch"]
        print(f"[INFO] {gpu_count} GPUs detected! Using accelerate launch (DDP).")

    if not args.skip_preprocess:
        preprocess_cmd = [
            os.path.join(THIS_DIR, "preprocess_services.py"),
            "--out_dir", args.preprocess_dir,
            "--max_services", str(args.max_services),
        ]
        label = "Step 1 — SVMD-DE-Otsu-MODWT Decomposition"
        run_step(py + preprocess_cmd, label)

    if not args.skip_train:
        train_cmd = launcher_cmd + [
            os.path.join(arch_dir, "train.py"),
            "--preprocess_dir", args.preprocess_dir,
            "--out_dir", args.model_dir,
            "--log_dir", args.log_dir,
        ]
        if args.cpu:
            train_cmd.append("--cpu")
        if args.epochs_override is not None:
            train_cmd.extend(["--epochs", str(args.epochs_override)])
        if args.dataset_workers > 0:
            train_cmd.extend(["--dataset_workers", str(args.dataset_workers)])
        run_step(train_cmd, f"Step 2 — Train TSDP ({args.arch})")

    if not args.skip_eval:
        eval_cmd = launcher_cmd + [
            os.path.join(arch_dir, "evaluate.py"),
            "--preprocess_dir", args.preprocess_dir,
            "--model_dir", args.model_dir,
            "--log_dir", args.log_dir,
        ]
        if args.cpu:
            eval_cmd.append("--cpu")
        run_step(eval_cmd, f"Step 3 — Evaluate TSDP ({args.arch})")

    print("\n" + "=" * 60)
    print(f"=== TSDP Pipeline ({args.arch}) Complete ===")
    print("=" * 60)


if __name__ == "__main__":
    main()

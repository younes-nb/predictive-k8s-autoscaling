import argparse
import os
import subprocess
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    ap = argparse.ArgumentParser(description="SDT-Net end-to-end pipeline")
    ap.add_argument("--preprocess_dir", default="/dataset/sdtnet_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/sdtnet")
    ap.add_argument("--max_services", type=int, default=0,
                    help="Maximum services to process (default: 0 = all)")
    ap.add_argument("--epochs_override", type=int, default=None,
                    help="Override CFG.EPOCHS for quick testing")
    ap.add_argument("--skip_preprocess", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    py = [sys.executable]

    if not args.skip_preprocess:
        preprocess_cmd = [
            os.path.join(THIS_DIR, "preprocess_services.py"),
            "--out_dir", args.preprocess_dir,
            "--max_services", str(args.max_services),
        ]
        label = "Step 1 — SVMD-DE-Quantile-SVMD Decomposition"
        run_step(py + preprocess_cmd, label)

    if not args.skip_train:
        train_cmd = py + [
            os.path.join(THIS_DIR, "train.py"),
            "--preprocess_dir", args.preprocess_dir,
            "--out_dir", args.model_dir,
        ]
        if args.cpu:
            train_cmd.append("--cpu")
        if args.epochs_override is not None:
            train_cmd.extend(["--epochs", str(args.epochs_override)])
        run_step(train_cmd, "Step 2 — Train SDT-Net Model (TCN)")

    if not args.skip_eval:
        eval_cmd = py + [
            os.path.join(THIS_DIR, "evaluate.py"),
            "--preprocess_dir", args.preprocess_dir,
            "--model_dir", args.model_dir,
        ]
        if args.cpu:
            eval_cmd.append("--cpu")
        run_step(eval_cmd, "Step 3 — Evaluate SDT-Net")

    print("\n" + "=" * 60)
    print("=== SDT-Net Pipeline Complete ===")
    print("=" * 60)


if __name__ == "__main__":
    main()

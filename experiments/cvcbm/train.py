
import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG
from experiments.cvcbm.dataset import CoImfDataset
from experiments.cvcbm.model import CvcbmModel

def setup_logging(out_dir: str, co_imf_index: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"train_co_imf_{co_imf_index}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(logging.FileHandler(log_path, mode="a"))
    root.addHandler(logging.StreamHandler(sys.stdout))
    for h in root.handlers:
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return log_path

def train_one_co_imf(co_imf_index: int, args) -> None:
    log_path = setup_logging(args.out_dir, co_imf_index)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    epochs = args.epochs if args.epochs is not None else CFG.EPOCHS

    logging.info("=" * 60)
    logging.info("CVCBM — Training Co-IMF-%d on %s", co_imf_index, device)
    logging.info("Log file: %s", log_path)
    logging.info("=" * 60)

    train_ds = CoImfDataset(
        args.preprocess_dir, co_imf_index, "train",
        input_len=CFG.INPUT_LEN, stride=CFG.STRIDE,
        test_size=CFG.TEST_SIZE, val_frac=CFG.VAL_FRAC,
    )
    val_ds = CoImfDataset(
        args.preprocess_dir, co_imf_index, "val",
        input_len=CFG.INPUT_LEN, stride=CFG.STRIDE,
        test_size=CFG.TEST_SIZE, val_frac=CFG.VAL_FRAC,
    )

    if len(train_ds) == 0:
        logging.error("Empty training dataset for co_imf_%d. Aborting.", co_imf_index)
        return

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    model = CvcbmModel(
        input_len=CFG.INPUT_LEN,
        kernel_sizes=CFG.KERNEL_SIZES,
        conv1_out_ch=CFG.CONV1_OUT_CH,
        conv2_out_ch=CFG.CONV2_OUT_CH,
        bilstm_hidden=CFG.BILSTM_HIDDEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model parameters: %d", n_params)
    logging.info("Train windows: %d | Val windows: %d", len(train_ds), len(val_ds))
    logging.info("Epochs: %d", epochs)

    ckpt_path = os.path.join(args.out_dir, f"cvcbm_co_imf_{co_imf_index}.pt")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_accum, n_train = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_accum += loss.item() * x.size(0)
            n_train += x.size(0)
        avg_train = train_accum / max(n_train, 1)

        model.eval()
        val_accum, n_val = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_accum += loss.item() * x.size(0)
                n_val += x.size(0)
        avg_val = val_accum / max(n_val, 1)

        dt = time.time() - t0
        suffix = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "co_imf_index": co_imf_index,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "cfg": {
                        "input_len": CFG.INPUT_LEN,
                        "kernel_sizes": CFG.KERNEL_SIZES,
                        "conv1_out_ch": CFG.CONV1_OUT_CH,
                        "conv2_out_ch": CFG.CONV2_OUT_CH,
                        "bilstm_hidden": CFG.BILSTM_HIDDEN,
                    },
                },
                ckpt_path,
            )
            suffix = " [Checkpoint Saved]"

        logging.info(
            "Epoch %d/%d | train=%.8f | val=%.8f | %.1fs%s",
            epoch, epochs, avg_train, avg_val, dt, suffix,
        )

    logging.info(
        "=== Co-IMF-%d training complete. Best val loss: %.8f | Saved: %s ===",
        co_imf_index, best_val_loss, ckpt_path,
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train CVCBM models (one per Co-IMF component)."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/cvcbm_preprocess",
                    help="Directory with Co-IMF numpy files (default: /dataset/cvcbm_preprocess)")
    ap.add_argument("--out_dir", default="/proj/k8sautoscaledl-PG0/models/cvcbm",
                    help="Directory for model checkpoints (default: /proj/k8sautoscaledl-PG0/models/cvcbm)")
    ap.add_argument(
        "--co_imf_index", type=int, default=None,
        help="Train a single Co-IMF model (0, 1, or 2). Omit to train all 3.",
    )
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override paper EPOCHS for smoke tests without editing config.py")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    indices = (
        [args.co_imf_index]
        if args.co_imf_index is not None
        else list(range(CFG.N_CLUSTERS))
    )
    invalid = [idx for idx in indices if idx < 0 or idx >= CFG.N_CLUSTERS]
    if invalid:
        raise ValueError(f"Invalid Co-IMF index/indices: {invalid}")

    for idx in indices:
        train_one_co_imf(idx, args)

if __name__ == "__main__":
    main()

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tsdp.config import CFG as GLOBAL_CFG
from experiments.tsdp.cnn_bilstm.config import CFG as ARCH_CFG
from experiments.tsdp.dataset import TsdpDataset, N_CHANNELS
from experiments.tsdp.cnn_bilstm.model import CnnBiLSTM
from training.metrics import compute_metrics

ARCH_NAME = "CNN-BiLSTM"


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("cfg", {})
    model = CnnBiLSTM(
        in_channels=cfg.get("n_channels", N_CHANNELS),
        input_len=cfg.get("input_len", GLOBAL_CFG.INPUT_LEN),
        pred_horizon=cfg.get("pred_horizon", GLOBAL_CFG.PRED_HORIZON),
        kernel_sizes=cfg.get("kernel_sizes", ARCH_CFG.KERNEL_SIZES),
        conv1_out_ch=cfg.get("conv1_out_ch", ARCH_CFG.CONV1_OUT_CH),
        conv2_out_ch=cfg.get("conv2_out_ch", ARCH_CFG.CONV2_OUT_CH),
        bilstm_hidden=cfg.get("bilstm_hidden", ARCH_CFG.BILSTM_HIDDEN),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logging.info("Loaded %s model from %s", ARCH_NAME, ckpt_path)
    return model


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    try:
        tz = ZoneInfo("Asia/Tehran")
    except Exception:
        tz = None
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Evaluate TSDP {ARCH_NAME} model."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/tsdp_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/tsdp")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/tsdp")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    log_path = setup_logging(args.log_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info("Evaluating TSDP [%s] on %s", ARCH_NAME, device)

    ckpt = os.path.join(args.model_dir, "tsdp.pt")
    if not os.path.exists(ckpt):
        logging.error("Checkpoint not found: %s — train the model first.", ckpt)
        sys.exit(1)

    model = load_model(ckpt, device)
    logging.info("Loaded TSDP [%s] model from %s", ARCH_NAME, ckpt)

    test_ds = TsdpDataset(
        args.preprocess_dir, "test",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
    )
    if len(test_ds) == 0:
        logging.error("Empty test dataset. Cannot evaluate.")
        sys.exit(1)

    batch_size = args.batch_size if args.batch_size is not None else GLOBAL_CFG.BATCH_SIZE
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list, true_list, last_list = [], [], []

    with torch.no_grad():
        for x, y, last in loader:
            pred = model(x.to(device)).cpu().numpy()
            preds_list.append(pred)
            true_list.append(y.numpy())
            last_list.append(last.numpy().ravel())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(true_list, axis=0)
    lasts = np.concatenate(last_list)

    n = len(preds)
    logging.info("Test windows: %d", n)

    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))

    logging.info("")
    logging.info("=" * 60)
    logging.info("TSDP [%s] — Test Results  (n=%d samples)", ARCH_NAME, n)
    logging.info("=" * 60)
    logging.info("Paper Metrics:")
    logging.info("  MAE : %.8f", mae)
    logging.info("  MSE : %.8f", mse)
    logging.info("-" * 60)

    y_pred_2d = preds.reshape(-1, GLOBAL_CFG.PRED_HORIZON)
    y_true_2d = trues.reshape(-1, GLOBAL_CFG.PRED_HORIZON)

    logging.info("Shadowing Diagnostics (via training.metrics.compute_metrics):")
    compute_metrics(
        y_pred=y_pred_2d,
        y_true=y_true_2d,
        y_last=lasts,
        horizon=GLOBAL_CFG.PRED_HORIZON,
        total_samples=n,
        log_info=logging.info,
    )

    logging.info("")
    logging.info("=" * 60)
    logging.info("Full log saved to: %s", log_path)


if __name__ == "__main__":
    main()

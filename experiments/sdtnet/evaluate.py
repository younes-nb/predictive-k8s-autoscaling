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
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.sdtnet.config import CFG
from experiments.sdtnet.dataset import SdtnetDataset, N_CHANNELS
from experiments.sdtnet.model import SdtnetCNNBiLSTM
from training.metrics import compute_metrics


class _TehranFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            tz = ZoneInfo("Asia/Tehran")
        except Exception:
            tz = None
        ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.getMessage()}"


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    try:
        tz = ZoneInfo("Asia/Tehran")
    except Exception:
        tz = None
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"evaluate_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = _TehranFormatter()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def load_sdtnet_model(ckpt_path: str, device: torch.device) -> SdtnetCNNBiLSTM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    model = SdtnetCNNBiLSTM(
        in_channels=saved_cfg.get("total_channels", N_CHANNELS),
        input_len=saved_cfg.get("input_len", CFG.INPUT_LEN),
        pred_horizon=saved_cfg.get("pred_horizon", CFG.PRED_HORIZON),
        kernel_sizes=saved_cfg.get("kernel_sizes", CFG.KERNEL_SIZES),
        conv1_out_ch=saved_cfg.get("conv1_out_ch", CFG.CONV1_OUT_CH),
        conv2_out_ch=saved_cfg.get("conv2_out_ch", CFG.CONV2_OUT_CH),
        bilstm_hidden=saved_cfg.get("bilstm_hidden", CFG.BILSTM_HIDDEN),
        dropout=saved_cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(
    model: SdtnetCNNBiLSTM,
    dataset: SdtnetDataset,
    device: torch.device,
    batch_size: int = 512,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list, true_list, last_list = [], [], []

    with torch.no_grad():
        for x, y, last in loader:
            pred = model(x.to(device)).cpu().numpy()
            preds_list.append(pred)
            true_list.append(y.numpy())
            last_list.append(last.numpy().ravel())

    return (
        np.concatenate(preds_list, axis=0),
        np.concatenate(true_list, axis=0),
        np.concatenate(last_list),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate SDT-Net model (SVMD-DE-CNNBiLSTM)."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/sdtnet_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/sdtnet")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/sdtnet",
                    help="Directory for evaluation logs (default: /proj/k8sautoscaledl-PG0/logs/sdtnet)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    log_path = setup_logging(args.log_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info("Evaluating SDT-Net on %s", device)

    ckpt = os.path.join(args.model_dir, "sdtnet.pt")
    if not os.path.exists(ckpt):
        logging.error("Checkpoint not found: %s — train the model first.", ckpt)
        sys.exit(1)

    model = load_sdtnet_model(ckpt, device)
    logging.info("Loaded SDT-Net model from %s", ckpt)

    test_ds = SdtnetDataset(
        args.preprocess_dir, "test",
        input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
    )
    if len(test_ds) == 0:
        logging.error("Empty test dataset. Cannot evaluate.")
        sys.exit(1)

    preds, trues, lasts = predict(model, test_ds, device, args.batch_size)
    n = len(preds)
    logging.info("Test windows: %d", n)

    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))

    logging.info("")
    logging.info("=" * 60)
    logging.info("SDT-Net — Test Results  (n=%d samples)", n)
    logging.info("=" * 60)
    logging.info("Paper Metrics:")
    logging.info("  MAE : %.8f", mae)
    logging.info("  MSE : %.8f", mse)
    logging.info("-" * 60)

    y_pred_2d = preds.reshape(-1, CFG.PRED_HORIZON)
    y_true_2d = trues.reshape(-1, CFG.PRED_HORIZON)

    logging.info("Shadowing Diagnostics (via training.metrics.compute_metrics):")
    compute_metrics(
        y_pred=y_pred_2d,
        y_true=y_true_2d,
        y_last=lasts,
        horizon=CFG.PRED_HORIZON,
        total_samples=n,
        log_info=logging.info,
    )

    logging.info("")
    logging.info("=" * 60)
    logging.info("Full log saved to: %s", log_path)


if __name__ == "__main__":
    main()

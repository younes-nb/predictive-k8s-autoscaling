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
from experiments.tsdp.patchtst.config import CFG as ARCH_CFG
from experiments.tsdp.dataset import TsdpDataset, N_CHANNELS
from experiments.tsdp.patchtst.model import PatchTST
from training.metrics import compute_metrics

ARCH_NAME = "PatchTST"


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


def load_model(ckpt_path: str, device: torch.device) -> PatchTST:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    model = PatchTST(
        input_len=saved_cfg.get("input_len", GLOBAL_CFG.INPUT_LEN),
        pred_horizon=saved_cfg.get("pred_horizon", GLOBAL_CFG.PRED_HORIZON),
        n_channels=saved_cfg.get("n_channels", N_CHANNELS),
        patch_len=saved_cfg.get("patch_len", ARCH_CFG.PATCH_LEN),
        stride=saved_cfg.get("patch_stride", ARCH_CFG.PATCH_STRIDE),
        d_model=saved_cfg.get("d_model", ARCH_CFG.D_MODEL),
        n_heads=saved_cfg.get("n_heads", ARCH_CFG.N_HEADS),
        n_layers=saved_cfg.get("n_layers", ARCH_CFG.N_LAYERS),
        d_ff=saved_cfg.get("d_ff", ARCH_CFG.D_FF),
        dropout=saved_cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(
    model: PatchTST,
    dataset: TsdpDataset,
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
        description=f"Evaluate TSDP model ({ARCH_NAME})."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/tsdp_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/tsdp")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/tsdp")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)
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

    preds, trues, lasts = predict(model, test_ds, device, args.batch_size)
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

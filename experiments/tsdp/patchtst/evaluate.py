import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs

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
    log_path = os.path.join(log_dir, f"eval_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = _TehranFormatter()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def load_model(ckpt_path: str, device: torch.device, is_main: bool = False) -> PatchTST:
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
    if is_main:
        logging.info("Loaded %s model from %s", ARCH_NAME, ckpt_path)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Evaluate TSDP model ({ARCH_NAME})."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/tsdp_preprocess")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/tsdp")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/tsdp")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--dataset_workers", type=int, default=max(1, int(os.cpu_count() * 0.7)))
    args = ap.parse_args()

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    mixed_precision = "fp16" if not args.cpu and torch.cuda.is_available() else "no"
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=mixed_precision, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    log_info = lambda msg, *a: (
        logging.info(msg, *a) if accelerator.is_local_main_process else None
    )

    if accelerator.is_local_main_process:
        log_path = setup_logging(args.log_dir)
    else:
        log_path = None

    log_info("Evaluating TSDP [%s] on %s", ARCH_NAME, device)
    log_info("Distributed Processes: %d", accelerator.num_processes)

    ckpt = os.path.join(args.model_dir, "tsdp.pt")
    if not os.path.exists(ckpt):
        logging.error("Checkpoint not found: %s — train the model first.", ckpt)
        sys.exit(1)

    model = load_model(ckpt, device, is_main=accelerator.is_local_main_process)
    log_info("Loaded TSDP [%s] model from %s", ARCH_NAME, ckpt)

    test_ds = TsdpDataset(
        args.preprocess_dir, "test",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
        num_workers=args.dataset_workers,
    )
    if len(test_ds) == 0:
        logging.error("Empty test dataset. Cannot evaluate.")
        sys.exit(1)

    batch_size = args.batch_size
    pin_memory = device.type != "cpu"
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model, loader = accelerator.prepare(model, loader)

    preds_list, true_list, last_list = [], [], []

    with torch.no_grad():
        for x, y, last_val in loader:
            pred = model(x)
            gathered_pred, gathered_y, gathered_last = accelerator.gather_for_metrics((pred, y, last_val))

            if accelerator.is_local_main_process:
                preds_list.append(gathered_pred.cpu().numpy())
                true_list.append(gathered_y.cpu().numpy())
                last_list.append(gathered_last.cpu().numpy())

    if not accelerator.is_local_main_process:
        return

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(true_list, axis=0)
    lasts = np.concatenate(last_list)

    n = len(preds)
    log_info("Test windows: %d", n)

    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))

    log_info("")
    log_info("=" * 60)
    log_info("TSDP [%s] — Test Results  (n=%d samples)", ARCH_NAME, n)
    log_info("=" * 60)
    log_info("Paper Metrics:")
    log_info("  MAE : %.8f", mae)
    log_info("  MSE : %.8f", mse)
    log_info("-" * 60)

    y_pred_2d = preds.reshape(-1, GLOBAL_CFG.PRED_HORIZON)
    y_true_2d = trues.reshape(-1, GLOBAL_CFG.PRED_HORIZON)

    log_info("Shadowing Diagnostics (via training.metrics.compute_metrics):")
    compute_metrics(
        y_pred=y_pred_2d,
        y_true=y_true_2d,
        y_last=lasts,
        horizon=GLOBAL_CFG.PRED_HORIZON,
        total_samples=n,
        log_info=logging.info,
    )

    log_info("")
    log_info("=" * 60)
    if log_path:
        log_info("Full log saved to: %s", log_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()

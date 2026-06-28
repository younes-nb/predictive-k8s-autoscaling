
import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG
from experiments.cvcbm.dataset import CoImfDataset
from experiments.cvcbm.model import CvcbmModel
from training.metrics import compute_metrics

def setup_logging(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "evaluate_cvcbm.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path

def load_cvcbm_model(ckpt_path: str, device: torch.device) -> CvcbmModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    model = CvcbmModel(
        input_len=saved_cfg.get("input_len", CFG.INPUT_LEN),
        pred_horizon=saved_cfg.get("pred_horizon", CFG.PRED_HORIZON),
        kernel_sizes=saved_cfg.get("kernel_sizes", CFG.KERNEL_SIZES),
        conv1_out_ch=saved_cfg.get("conv1_out_ch", CFG.CONV1_OUT_CH),
        conv2_out_ch=saved_cfg.get("conv2_out_ch", CFG.CONV2_OUT_CH),
        bilstm_hidden=saved_cfg.get("bilstm_hidden", CFG.BILSTM_HIDDEN),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def predict_component(
    model: CvcbmModel,
    dataset: CoImfDataset,
    device: torch.device,
    batch_size: int = 512,
):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list, true_list, last_list = [], [], []
    pred_horizon = dataset.y.shape[1]

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
        description="Evaluate CVCBM models with shadowing diagnostics."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/cvcbm_preprocess",
                    help="Co-IMF directory (default: /dataset/cvcbm_preprocess)")
    ap.add_argument("--model_dir", default="/proj/k8sautoscaledl-PG0/models/cvcbm",
                    help="Checkpoint directory (default: /proj/k8sautoscaledl-PG0/models/cvcbm)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    log_path = setup_logging(args.model_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info("Evaluating CVCBM on %s", device)

    models = {}
    for k in range(CFG.N_CLUSTERS):
        ckpt = os.path.join(args.model_dir, f"cvcbm_co_imf_{k}.pt")
        if not os.path.exists(ckpt):
            logging.error("Checkpoint not found: %s — train this model first.", ckpt)
            sys.exit(1)
        models[k] = load_cvcbm_model(ckpt, device)
        logging.info("Loaded Co-IMF-%d model from %s", k, ckpt)

    component_preds = {}
    component_true = {}
    component_last = {}

    for k in range(CFG.N_CLUSTERS):
        test_ds = CoImfDataset(
            args.preprocess_dir, k, "test",
            input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
            stride=1,
            test_size=CFG.TEST_SIZE, val_frac=CFG.VAL_FRAC,
        )
        if len(test_ds) == 0:
            logging.error("Empty test dataset for co_imf_%d. Cannot evaluate.", k)
            sys.exit(1)

        preds, trues, lasts = predict_component(models[k], test_ds, device, args.batch_size)
        component_preds[k] = preds
        component_true[k] = trues
        component_last[k] = lasts
        logging.info("Co-IMF-%d: %d test windows", k, len(preds))

    n = min(len(v) for v in component_preds.values())
    if n == 0:
        logging.error("Zero aligned test samples. Exiting.")
        sys.exit(1)

    # With windowed decomposition, each Co-IMF model predicts the full original signal's next value.
    # Average predictions across components rather than summing them.
    # All Co-IMFs share the same target (original signal value), so component_true[0] suffices.
    summed_preds = sum(component_preds[k][:n] for k in range(CFG.N_CLUSTERS)) / CFG.N_CLUSTERS
    summed_true = component_true[0][:n]
    y_last = component_last[0][:n]

    mae = float(np.mean(np.abs(summed_preds - summed_true)))
    mse = float(np.mean((summed_preds - summed_true) ** 2))

    logging.info("")
    logging.info("=" * 60)
    logging.info("CVCBM — Test Results  (n=%d samples)", n)
    logging.info("=" * 60)
    logging.info("Paper Metrics (on reconstructed signal):")
    logging.info("  MAE : %.8f", mae)
    logging.info("  MSE : %.8f", mse)
    logging.info("-" * 60)

    y_pred_2d = summed_preds.reshape(-1, CFG.PRED_HORIZON)
    y_true_2d = summed_true.reshape(-1, CFG.PRED_HORIZON)

    logging.info("Shadowing Diagnostics (via training.metrics.compute_metrics):")
    compute_metrics(
        y_pred=y_pred_2d,
        y_true=y_true_2d,
        y_last=y_last,
        horizon=CFG.PRED_HORIZON,
        total_samples=n,
        log_info=logging.info,
    )

    logging.info("")
    logging.info("Per-Component Breakdown:")
    for k in range(CFG.N_CLUSTERS):
        cp = component_preds[k][:n]
        ct = component_true[k][:n]
        logging.info(
            "  Co-IMF-%d: MAE=%.8f  MSE=%.8f",
            k,
            float(np.mean(np.abs(cp - ct))),
            float(np.mean((cp - ct) ** 2)),
        )
    logging.info("=" * 60)
    logging.info("Full log saved to: %s", log_path)

if __name__ == "__main__":
    main()

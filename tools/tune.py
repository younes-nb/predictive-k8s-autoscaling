import os
import sys
import torch
import optuna
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.defaults import PATHS, PREPROCESSING, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster


# --- 1. LOSS FUNCTION ---
def weighted_mse(preds, target, w, under_penalty=5.0):
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    loss_matrix = sq_err * asym_weight
    per_sample = loss_matrix.mean(dim=1)

    w = w.clamp(min=0.1, max=20.0)
    return (w * per_sample).sum() / w.sum().clamp_min(1e-6)


def get_service_baselines():
    print("📊 Pre-computing service-level quantiles...")
    ds = ShardedWindowsDataset(
        PATHS.WINDOWS_DIR, "train", PREPROCESSING.INPUT_LEN, PREPROCESSING.PRED_HORIZON
    )

    service_vals = {}
    limit = min(len(ds), 50000)
    for i in range(0, limit, 10):
        x, _, sid = ds[i]
        s = int(sid)
        val = x[-1, -1].item()
        if s not in service_vals:
            service_vals[s] = []
        service_vals[s].append(val)

    baselines = {
        s: np.quantile(v, TRAINING.THETA_BASE) for s, v in service_vals.items()
    }
    return baselines


SERVICE_BASELINES = get_service_baselines()


@torch.no_grad()
def get_adaptive_weights(model, x_batch, y_batch, sid_batch, device, gamma, delta, k):
    model.train()
    preds = []
    for _ in range(10):
        preds.append(model(x_batch).unsqueeze(0))

    preds_stack = torch.cat(preds, dim=0)
    sigma = preds_stack.std(dim=0)[:, -1]

    sigma_global = sigma.mean()

    y_target = y_batch[:, -1]

    tb = torch.tensor(
        [SERVICE_BASELINES.get(int(s), 0.7) for s in sid_batch], device=device
    )

    theta = torch.clamp(
        tb - k * sigma_global, min=TRAINING.THETA_MIN, max=TRAINING.THETA_MAX
    )

    dist_sq = (y_target - theta) ** 2
    w = 1.0 + gamma * torch.exp(-dist_sq / (2.0 * (delta**2)))
    return w


def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    gamma = trial.suggest_float("gamma", 15.0, 25.0)
    delta = trial.suggest_float("delta", 0.05, 0.12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ShardedWindowsDataset(
        PATHS.WINDOWS_DIR, "train", PREPROCESSING.INPUT_LEN, PREPROCESSING.PRED_HORIZON
    )
    val_ds = ShardedWindowsDataset(
        PATHS.WINDOWS_DIR, "val", PREPROCESSING.INPUT_LEN, PREPROCESSING.PRED_HORIZON
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAINING.BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = RNNForecaster(
        input_size=train_ds[0][0].shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=PREPROCESSING.PRED_HORIZON,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=TRAINING.WEIGHT_DECAY
    )

    for epoch in range(2):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()

    for epoch in range(2, 10):
        model.train()
        train_loss = 0.0
        for x, y, sid in train_loader:
            x, y = x.to(device), y.to(device)

            w = get_adaptive_weights(
                model, x, y, sid, device, gamma, delta, TRAINING.K_UNCERTAINTY
            )

            model.train()
            optimizer.zero_grad()
            preds = model(x)
            loss = weighted_mse(preds, y, w, under_penalty=TRAINING.UNDER_PENALTY)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING.GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += nn.MSELoss()(model(x), y).item()

        avg_val = val_loss / len(val_loader)
        trial.report(avg_val, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=15)

    print("\n🏆 Best Trial Results:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

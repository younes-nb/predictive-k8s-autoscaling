import os
import sys
import glob
import torch
import optuna
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.defaults import PATHS, PREPROCESSING, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster


def get_ground_truth_weights(
    y_batch, sid_batch, device, gamma, delta, theta_min, theta_mode="adaptive"
):
    y_target = y_batch[:, -1]

    if theta_mode == "adaptive":
        tb = torch.tensor(
            [SERVICE_BASELINES.get(int(s), TRAINING.THETA_BASE) for s in sid_batch],
            device=device,
        )
        dist = torch.max(
            torch.zeros_like(y_target), torch.max(theta_min - y_target, y_target - tb)
        )
    else:
        dist = torch.abs(y_target - TRAINING.THETA_BASE)

    w = 1.0 + gamma * torch.exp(-(dist**2) / (2.0 * (delta**2)))
    return w


def weighted_mse(preds, target, w, under_penalty=5.0):
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    loss_matrix = sq_err * asym_weight
    per_sample = loss_matrix.mean(dim=1)

    return (w * per_sample).sum() / w.sum().clamp_min(1e-6)


def get_service_baselines():
    print("📊 Pre-computing service-level quantiles...")
    x_files = sorted(glob.glob(os.path.join(PATHS.WINDOWS_DIR, "part-*_X_train.npy")))
    service_vals = {}

    for x_path in x_files[:5]:
        sid_path = x_path.replace("_X_train.npy", "_sid_train.npy")
        if not os.path.exists(sid_path):
            continue

        X = np.load(x_path, mmap_mode="r")
        SIDs = np.load(sid_path, mmap_mode="r")
        u_now = X[:, -1, -1] if X.ndim == 3 else X[:, -1]

        for i in range(len(SIDs)):
            s = int(SIDs[i])
            if s not in service_vals:
                service_vals[s] = []
            service_vals[s].append(u_now[i])

    return {s: np.quantile(v, TRAINING.THETA_BASE) for s, v in service_vals.items()}


SERVICE_BASELINES = get_service_baselines()

train_ds = ShardedWindowsDataset(
    PATHS.WINDOWS_DIR,
    "train",
    PREPROCESSING.INPUT_LEN,
    PREPROCESSING.PRED_HORIZON,
    use_weights=False,
)
val_ds = ShardedWindowsDataset(
    PATHS.WINDOWS_DIR,
    "val",
    PREPROCESSING.INPUT_LEN,
    PREPROCESSING.PRED_HORIZON,
    use_weights=False,
)


def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    gamma = trial.suggest_float("gamma", 10.0, 30.0)
    delta = trial.suggest_float("delta", 0.03, 0.15)
    under_penalty = trial.suggest_float("under_penalty", 1.0, 8.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds, batch_size=TRAINING.BATCH_SIZE, shuffle=True, num_workers=4
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
        rnn_type="lstm",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(TRAINING.EPOCHS):
        model.train()
        total_train_loss = 0
        for x, y, sid in train_loader:
            x, y = x.to(device), y.to(device)

            w = get_ground_truth_weights(
                y,
                sid,
                device,
                gamma,
                delta,
                theta_min=TRAINING.THETA_MIN,
                theta_mode=TRAINING.WEIGHT_MODE,
            )

            optimizer.zero_grad()
            preds = model(x)
            loss = weighted_mse(preds, y, w, under_penalty=under_penalty)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

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
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("\n🏆 Best Trial:", study.best_params)

import optuna
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.defaults import PATHS, PREPROCESSING, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster

def objective(trial):
    
    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    dropout = trial.suggest_float("dropout", 0.4, 0.5) 
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    
    train_ds = ShardedWindowsDataset(
        PATHS.WINDOWS_DIR, "train", PREPROCESSING.INPUT_LEN, PREPROCESSING.PRED_HORIZON, use_weights=True
    )
    val_ds = ShardedWindowsDataset(
        PATHS.WINDOWS_DIR, "val", PREPROCESSING.INPUT_LEN, PREPROCESSING.PRED_HORIZON, use_weights=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNNForecaster(
        input_size=train_ds[0][0].shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=PREPROCESSING.PRED_HORIZON,
        rnn_type="lstm"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(5): 
        model.train()
        for x, y, w in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            
            loss = criterion(preds, y) 
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()
        
        avg_val_loss = val_loss / len(val_loader)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n--- Best Trial ---")
    print(f"Value: {study.best_value}")
    print("Params: ")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
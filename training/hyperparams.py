from __future__ import annotations

import copy
import logging
import math
import math as _math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.defaults import PATHS, TRAINING
from core.losses import weighted_mse
from core.models import RNNForecaster
from training.resume_utils import load_resume_state, save_resume_state


def hyperparam_key(hyperparams):
    return (
        int(hyperparams["hidden_size"]),
        int(hyperparams["num_layers"]),
        round(float(hyperparams["dropout"]), 4),
        round(float(hyperparams["lr"]), 8),
    )


def sample_hyperparams(rng, used_keys):
    log_min = math.log10(TRAINING.LR_RANGE[0])
    log_max = math.log10(TRAINING.LR_RANGE[1])
    for _ in range(TRAINING.HYPERPARAM_SAMPLE_ATTEMPTS):
        candidate = {
            "hidden_size": rng.choice(TRAINING.HIDDEN_SIZE_OPTIONS),
            "num_layers": rng.choice(TRAINING.NUM_LAYERS_OPTIONS),
            "dropout": round(rng.uniform(*TRAINING.DROPOUT_RANGE), 4),
            "lr": round(10 ** rng.uniform(log_min, log_max), 8),
        }
        key = hyperparam_key(candidate)
        if key not in used_keys:
            used_keys.add(key)
            return candidate
    return None


def apply_hyperparams(args, hyperparams):
    args.hidden_size = hyperparams["hidden_size"]
    args.num_layers = hyperparams["num_layers"]
    args.dropout = hyperparams["dropout"]
    args.lr = hyperparams["lr"]


class SFOAOptimizer:
    D = 4
    BOUNDS_LOW = None
    BOUNDS_HIGH = None

    def __init__(self, eval_fn, N, Tmax, Gp, seed=42):
        if N < 3:
            raise ValueError("SFOA requires N >= 3 to sample distinct distances.")
        self.eval_fn = eval_fn
        self.N = int(N)
        self.Tmax = int(Tmax)
        self.Gp = float(Gp)
        self.rng = np.random.default_rng(seed)
        self.BOUNDS_LOW, self.BOUNDS_HIGH = self._build_bounds()
        if self.N < 6:
            logging.warning(
                "[SFOA] Population size N=%s is < 6; preying will sample fewer candidates.",
                self.N,
            )

    def _build_bounds(self):
        low = np.array(
            [
                0.0,
                0.0,
                TRAINING.DROPOUT_RANGE[0],
                _math.log10(TRAINING.LR_RANGE[0]),
            ],
            dtype=float,
        )
        high = np.array(
            [
                float(len(TRAINING.HIDDEN_SIZE_OPTIONS) - 1),
                float(len(TRAINING.NUM_LAYERS_OPTIONS) - 1),
                TRAINING.DROPOUT_RANGE[1],
                _math.log10(TRAINING.LR_RANGE[1]),
            ],
            dtype=float,
        )
        return low, high

    def _decode(self, x: np.ndarray) -> dict:
        """Convert continuous position vector (D,) to a hyperparams dict."""

        def clip(val, low, high):
            return max(low, min(high, float(val)))

        idx_hidden = int(
            round(clip(x[0], self.BOUNDS_LOW[0], self.BOUNDS_HIGH[0]))
        )
        idx_layers = int(
            round(clip(x[1], self.BOUNDS_LOW[1], self.BOUNDS_HIGH[1]))
        )
        dropout = round(float(clip(x[2], self.BOUNDS_LOW[2], self.BOUNDS_HIGH[2])), 4)
        lr = round(10 ** float(clip(x[3], self.BOUNDS_LOW[3], self.BOUNDS_HIGH[3])), 8)
        return {
            "hidden_size": TRAINING.HIDDEN_SIZE_OPTIONS[idx_hidden],
            "num_layers": TRAINING.NUM_LAYERS_OPTIONS[idx_layers],
            "dropout": dropout,
            "lr": lr,
        }

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.BOUNDS_LOW, self.BOUNDS_HIGH)

    def _evaluate_all(self, X: np.ndarray) -> np.ndarray:
        fitness = []
        for row in X:
            hyperparams = self._decode(row)
            try:
                fitness.append(float(self.eval_fn(copy.deepcopy(hyperparams))))
            except Exception as exc:
                logging.warning("[SFOA] Evaluation failed for %s: %s", hyperparams, exc)
                fitness.append(float("inf"))
        return np.asarray(fitness, dtype=float)

    def _explore(self, X: np.ndarray, T: int) -> np.ndarray:
        theta = (_math.pi / 2) * (T / self.Tmax)
        Et = ((self.Tmax - T) / self.Tmax) * _math.cos(theta)
        X_new = X.copy()
        for i in range(self.N):
            p = self.rng.integers(0, self.D)
            candidates = [j for j in range(self.N) if j != i]
            k1, k2 = self.rng.choice(candidates, size=2, replace=False)
            A1 = self.rng.uniform(-1, 1)
            A2 = self.rng.uniform(-1, 1)
            y_p = Et * X[i, p] + A1 * (X[k1, p] - X[i, p]) + A2 * (
                X[k2, p] - X[i, p]
            )
            if self.BOUNDS_LOW[p] <= y_p <= self.BOUNDS_HIGH[p]:
                X_new[i, p] = y_p
        return X_new

    def _exploit(self, X: np.ndarray, best_pos: np.ndarray, T: int) -> np.ndarray:
        X_new = X.copy()
        for i in range(self.N):
            if i != self.N - 1:
                candidates = [j for j in range(self.N) if j != i]
                m_count = min(5, len(candidates))
                mp_indices = self.rng.choice(candidates, size=m_count, replace=False)
                distances = [best_pos - X[mp] for mp in mp_indices]
                sel_count = min(2, len(distances))
                sel = self.rng.choice(len(distances), size=sel_count, replace=False)
                dm1 = distances[sel[0]]
                dm2 = distances[sel[1]] if sel_count > 1 else distances[sel[0]]
                r1, r2 = self.rng.uniform(0, 1), self.rng.uniform(0, 1)
                y = X[i] + r1 * dm1 + r2 * dm2
                X_new[i] = self._clip_to_bounds(y)
            else:
                y = _math.exp(-T * self.N / self.Tmax) * X[i]
                X_new[i] = self._clip_to_bounds(y)
        return X_new

    def optimize(self, initial_state: dict = None) -> tuple[dict, float, dict]:
        if initial_state is not None and "X" in initial_state:
            X = np.asarray(initial_state["X"], dtype=float)
            if X.shape != (self.N, self.D):
                logging.warning(
                    "[SFOA] Resume state shape %s does not match (%d, %d); reinitializing.",
                    X.shape,
                    self.N,
                    self.D,
                )
                X = self.rng.random((self.N, self.D)) * (
                    self.BOUNDS_HIGH - self.BOUNDS_LOW
                ) + self.BOUNDS_LOW
        else:
            X = self.rng.random((self.N, self.D)) * (
                self.BOUNDS_HIGH - self.BOUNDS_LOW
            ) + self.BOUNDS_LOW

        fitness = self._evaluate_all(X)
        best_idx = int(np.argmin(fitness))
        best_pos = X[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        start_T = 1
        if initial_state is not None:
            start_T = int(initial_state.get("T", 0)) + 1

        last_T = start_T - 1
        if start_T <= self.Tmax:
            for T in range(start_T, self.Tmax + 1):
                rand_val = self.rng.uniform(0, 1)
                if rand_val > self.Gp:
                    X = self._explore(X, T)
                else:
                    X = self._exploit(X, best_pos, T)

                fitness = self._evaluate_all(X)
                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < best_fitness:
                    best_fitness = float(fitness[best_idx])
                    best_pos = X[best_idx].copy()

                best_params = self._decode(best_pos)
                logging.info(
                    "[SFOA] Iteration %d/%d | Best val loss: %.6f | hidden=%s, layers=%s, dropout=%.4f, lr=%.8f",
                    T,
                    self.Tmax,
                    best_fitness,
                    best_params["hidden_size"],
                    best_params["num_layers"],
                    best_params["dropout"],
                    best_params["lr"],
                )
                last_T = T

        final_state = {
            "X": X,
            "fitness": fitness,
            "best_pos": best_pos,
            "best_fitness": best_fitness,
            "T": last_T,
            "N": self.N,
            "Tmax": self.Tmax,
        }
        return self._decode(best_pos), best_fitness, final_state


def run_sfoa_search(args, train_ds, val_ds, device) -> dict:
    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    first_x, *_ = train_ds[0]
    input_size = first_x.shape[-1]
    eval_counter = {"count": 0}

    def eval_fn(hyperparams: dict) -> float:
        model = None
        optimizer = None
        eval_counter["count"] += 1
        torch.manual_seed(args.seed + eval_counter["count"])
        try:
            model = RNNForecaster(
                input_size=input_size,
                hidden_size=hyperparams["hidden_size"],
                num_layers=hyperparams["num_layers"],
                dropout=hyperparams["dropout"],
                horizon=args.pred_horizon,
                rnn_type=args.rnn_type,
                bidirectional=args.bidirectional,
                quantiles=None,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams["lr"],
                weight_decay=args.weight_decay,
            )

            for _ in range(TRAINING.SFOA_EVAL_EPOCHS):
                model.train()
                for batch in train_loader:
                    if args.use_weights:
                        x, y, w, _ = batch
                    else:
                        x, y, _ = batch
                        w = None
                    x = x.to(device)
                    y = y.to(device)
                    if w is not None:
                        w = w.to(device)

                    optimizer.zero_grad()
                    preds = model(x)
                    loss = weighted_mse(
                        preds, y, w, under_penalty=args.under_penalty
                    )
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss_accum = 0.0
            val_samples_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    if args.use_weights:
                        x, y, w, _ = batch
                    else:
                        x, y, _ = batch
                        w = None
                    x = x.to(device)
                    y = y.to(device)
                    if w is not None:
                        w = w.to(device)

                    preds = model(x)
                    loss = weighted_mse(
                        preds, y, w, under_penalty=args.under_penalty
                    )
                    val_loss_accum += loss.item() * x.size(0)
                    val_samples_seen += x.size(0)
            return val_loss_accum / max(1, val_samples_seen)
        except Exception as exc:
            logging.warning("[SFOA] Evaluation failed: %s", exc, exc_info=True)
            return float("inf")
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            torch.cuda.empty_cache()

    logging.info(
        "[SFOA] Starting hyperparameter search: N=%d, Tmax=%d, eval_epochs=%d",
        TRAINING.SFOA_POPULATION,
        TRAINING.SFOA_ITERATIONS,
        TRAINING.SFOA_EVAL_EPOCHS,
    )

    sfoa = SFOAOptimizer(
        eval_fn=eval_fn,
        N=TRAINING.SFOA_POPULATION,
        Tmax=TRAINING.SFOA_ITERATIONS,
        Gp=TRAINING.SFOA_GP,
        seed=args.seed,
    )

    resume_state = load_resume_state(PATHS.RESUME_STATE_FILE)
    initial_state = (
        resume_state.get("sfoa_state")
        if resume_state and "sfoa_state" in resume_state
        else None
    )
    best_hp, best_fitness, sfoa_state = sfoa.optimize(initial_state=initial_state)

    try:
        existing = load_resume_state(PATHS.RESUME_STATE_FILE) or {}
        existing["sfoa_state"] = sfoa_state
        save_resume_state(PATHS.RESUME_STATE_FILE, existing)
    except Exception as exc:
        logging.warning("Could not save SFOA state: %s", exc)

    logging.info(
        "[SFOA] Search complete. Best val loss: %.6f | Params: %s",
        best_fitness,
        best_hp,
    )
    return best_hp

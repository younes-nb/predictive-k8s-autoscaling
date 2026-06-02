import os
import sys
import argparse
import logging
import math
import math as _math
import random
import copy
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING, DEFAULT_CHECKPOINT_PATH
from core.dataset import ShardedWindowsDataset
from core.models import RNNForecaster


def setup_logging(mode="train", log_path=None):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

    try:
        tehran_tz = ZoneInfo("Asia/Tehran")
    except Exception:
        print("[WARN] Could not find 'Asia/Tehran' timezone. Using system time.")
        tehran_tz = None

    now = datetime.now(tehran_tz)
    if log_path is None:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{mode}_{timestamp}.log"
        log_path = os.path.join(PATHS.LOGS_DIR, log_filename)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    logging.info(f"=== {mode.upper()} SESSION STARTED ===")
    logging.info(f"Log file: {log_path}")
    logging.info(f"Timestamp: {now}")
    return log_path


def weighted_mse(preds, target, w=None, under_penalty=5.0):
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    value_loss = (sq_err * asym_weight).mean(dim=1)

    if w is None:
        return value_loss.mean()

    w = w.clamp(min=0.1, max=15.0)
    return (w * value_loss).sum() / w.sum().clamp_min(1e-6)


class PinballLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]):
        super().__init__()
        q = torch.tensor([float(q) for q in quantiles], dtype=torch.float32)
        self.register_buffer("quantiles", q)

    def forward(self, preds: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        if preds.dim() != 3:
            raise ValueError("PinballLoss expects preds with shape (batch, horizon, q).")
        q = self.quantiles.view(1, 1, -1)
        diff = target.unsqueeze(-1) - preds
        loss = torch.maximum(q * diff, (1.0 - q) * (-diff))
        per_sample = loss.mean(dim=(1, 2))

        if w is None:
            return per_sample.mean()

        w = w.clamp(min=0.1, max=15.0)
        return (w * per_sample).sum() / w.sum().clamp_min(1e-6)


def find_max_batch_size(
    model, input_size, args, device, loss_fn: Optional[nn.Module] = None, starting_batch=8192
):
    batch_size = starting_batch
    model.train()

    while batch_size > 0:
        try:
            dummy_x = torch.randn(batch_size, args.input_len, input_size, device=device)
            dummy_y = torch.randn(batch_size, args.pred_horizon, device=device)

            optimizer_dummy = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer_dummy.zero_grad()

            preds = model(dummy_x)
            if loss_fn is None:
                loss = ((preds - dummy_y) ** 2).mean()
            else:
                loss = loss_fn(preds, dummy_y)
            loss.backward()

            optimizer_dummy.zero_grad()
            del dummy_x, dummy_y, preds, loss
            torch.cuda.empty_cache()

            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e

    raise RuntimeError("Could not find a batch size that fits in memory.")


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


def load_resume_state(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        logging.warning("Failed to load resume state: %s", exc)
        return None


def save_resume_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


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
        dropout = round(
            float(clip(x[2], self.BOUNDS_LOW[2], self.BOUNDS_HIGH[2])), 4
        )
        lr = round(
            10 ** float(clip(x[3], self.BOUNDS_LOW[3], self.BOUNDS_HIGH[3])), 8
        )
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
                logging.warning(
                    "[SFOA] Evaluation failed for %s: %s", hyperparams, exc
                )
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
            y_p = (
                Et * X[i, p]
                + A1 * (X[k1, p] - X[i, p])
                + A2 * (X[k2, p] - X[i, p])
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


def train(args):
    accelerator = Accelerator(cpu=args.cpu)
    device = accelerator.device

    log_info = lambda msg: (
        logging.info(msg) if accelerator.is_local_main_process else None
    )

    if not hasattr(args, "resume_training"):
        args.resume_training = False

    resume_state = (
        load_resume_state(PATHS.RESUME_STATE_FILE) if args.resume_training else None
    )
    if resume_state and "args" in resume_state:
        args = argparse.Namespace(**resume_state["args"])
        args.resume_training = True
    if not hasattr(args, "probabilistic"):
        args.probabilistic = TRAINING.PROBABILISTIC_TRAINING
    # Resolve hyperparam optimizer setting (CLI arg takes precedence over default)
    hyperparam_optimizer = getattr(
        args, "hyperparam_optimizer", TRAINING.HYPERPARAM_OPTIMIZER
    )

    rng = random.Random(args.seed)

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging(
            "train", log_path=resume_state.get("log_path") if resume_state else None
        )

    used_keys = (
        {tuple(k) for k in resume_state.get("used_hyperparams", [])}
        if resume_state
        else set()
    )
    start_epoch = resume_state.get("epoch", 0) + 1 if resume_state else 1
    best_score = (
        resume_state.get("best_score", float("inf")) if resume_state else float("inf")
    )
    last_train_loss = resume_state.get("last_train_loss") if resume_state else None
    no_change_streak = (
        resume_state.get("no_change_streak", 0)
        if resume_state and last_train_loss is not None
        else 0
    )

    if resume_state:
        log_info("=== RESUMED TRAINING SESSION ===")
        log_info(
            f"Resuming training from epoch {start_epoch} using {PATHS.RESUME_STATE_FILE}"
        )

    if start_epoch > args.epochs:
        log_info("\n--- Configuration Inputs ---")
        for key, value in vars(args).items():
            log_info(f"{key:<20}: {value}")
        log_info("-" * 30)
        log_info("Resume state indicates training has already completed.")
        return

    log_info("\n--- Loading Datasets ---")

    train_ds = ShardedWindowsDataset(
        args.windows_dir, "train", args.input_len, args.pred_horizon, args.use_weights
    )

    val_ds = ShardedWindowsDataset(
        args.windows_dir, "val", args.input_len, args.pred_horizon, args.use_weights
    )

    log_info(f"Train samples (Total): {len(train_ds)}")
    log_info(f"Val samples (Total):   {len(val_ds)}")

    if len(train_ds) > 0:
        first_x, _, *_ = train_ds[0]
        input_size = first_x.shape[-1]
        log_info(f"Inferred Input Size: {input_size}")
    else:
        raise RuntimeError("Train dataset is empty.")

    current_hyperparams = resume_state.get("hyperparams") if resume_state else None
    if current_hyperparams is not None:
        used_keys.add(hyperparam_key(current_hyperparams))

    if current_hyperparams is None:
        if hyperparam_optimizer == "sfoa" and accelerator.is_local_main_process:
            log_info("Running SFOA hyperparameter search before main training...")
            current_hyperparams = run_sfoa_search(args, train_ds, val_ds, device)
            if current_hyperparams is None:
                logging.warning(
                    "[SFOA] Search did not return hyperparameters; falling back to random sampling."
                )
                current_hyperparams = sample_hyperparams(rng, used_keys)
        if hyperparam_optimizer == "sfoa" and accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

            def encode_hyperparams(hyperparams):
                hidden_idx = TRAINING.HIDDEN_SIZE_OPTIONS.index(
                    hyperparams["hidden_size"]
                )
                layer_idx = TRAINING.NUM_LAYERS_OPTIONS.index(hyperparams["num_layers"])
                log_lr = _math.log10(hyperparams["lr"])
                return torch.tensor(
                    [hidden_idx, layer_idx, hyperparams["dropout"], log_lr],
                    device=device,
                    dtype=torch.float32,
                )

            def decode_hyperparams(tensor):
                def clip(val, low, high):
                    return max(low, min(high, float(val)))

                hidden_idx = int(
                    round(
                        clip(
                            tensor[0].item(),
                            0,
                            len(TRAINING.HIDDEN_SIZE_OPTIONS) - 1,
                        )
                    )
                )
                layer_idx = int(
                    round(
                        clip(
                            tensor[1].item(),
                            0,
                            len(TRAINING.NUM_LAYERS_OPTIONS) - 1,
                        )
                    )
                )
                dropout = round(
                    clip(
                        tensor[2].item(),
                        TRAINING.DROPOUT_RANGE[0],
                        TRAINING.DROPOUT_RANGE[1],
                    ),
                    4,
                )
                lr = round(
                    10
                    ** clip(
                        tensor[3].item(),
                        _math.log10(TRAINING.LR_RANGE[0]),
                        _math.log10(TRAINING.LR_RANGE[1]),
                    ),
                    8,
                )
                return {
                    "hidden_size": TRAINING.HIDDEN_SIZE_OPTIONS[hidden_idx],
                    "num_layers": TRAINING.NUM_LAYERS_OPTIONS[layer_idx],
                    "dropout": dropout,
                    "lr": lr,
                }

            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    if accelerator.is_local_main_process:
                        hyper_tensor = encode_hyperparams(current_hyperparams)
                    else:
                        hyper_tensor = torch.zeros(SFOAOptimizer.D, device=device)
                    dist.broadcast(hyper_tensor, src=0)
                    current_hyperparams = decode_hyperparams(hyper_tensor)
                else:
                    logging.warning(
                        "[SFOA] Distributed broadcast unavailable; falling back to random sampling on non-main processes."
                    )
                    if not accelerator.is_local_main_process:
                        current_hyperparams = sample_hyperparams(rng, used_keys)
            except Exception as exc:
                logging.warning(
                    "[SFOA] Failed to broadcast hyperparameters: %s", exc
                )
                if not accelerator.is_local_main_process:
                    current_hyperparams = sample_hyperparams(rng, used_keys)
        elif hyperparam_optimizer != "sfoa":
            current_hyperparams = sample_hyperparams(rng, used_keys)

    if current_hyperparams is not None:
        used_keys.add(hyperparam_key(current_hyperparams))

    if current_hyperparams is None:
        raise RuntimeError(
            "Unable to select a unique hyperparameter set after "
            f"{TRAINING.HYPERPARAM_SAMPLE_ATTEMPTS} attempts."
        )

    apply_hyperparams(args, current_hyperparams)

    log_info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        log_info(f"{key:<20}: {value}")
    log_info("-" * 30)

    log_info(f"Device: {device} | Distributed Processes: {accelerator.num_processes}")
    torch.manual_seed(args.seed)

    quantiles = TRAINING.QUANTILES
    pinball_loss = PinballLoss(quantiles).to(device) if args.probabilistic else None

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        quantiles=quantiles if args.probabilistic else None,
    ).to(device)


    if device.type != "cpu":
        log_info("Tuning per-GPU batch size to hardware limits...")
        max_batch = find_max_batch_size(model, input_size, args, device, pinball_loss)

        safe_batch_size = int(max_batch * 0.8)
        safe_batch_size = 2 ** int(math.log2(max(1, safe_batch_size)))

        log_info(
            f"Auto-selected per-GPU Batch Size: {safe_batch_size} (Global Batch Size: {safe_batch_size * accelerator.num_processes})"
        )
        args.batch_size = safe_batch_size

    system_cores = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() or 1
    optimal_workers = min(system_cores, 4 * gpu_count)
    log_info(
        f"Dynamically set num_workers to {optimal_workers} (Cores: {system_cores}, GPUs: {gpu_count})"
    )

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if resume_state:
        model_state = resume_state.get("model_state_dict")
        if model_state:
            model.load_state_dict(model_state)
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    log_info("\n--- Starting Training Loop ---")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = datetime.now()
        model.train()
        train_loss_accum = 0.0
        train_samples_seen = 0

        for batch in train_loader:
            if args.use_weights:
                x, y, w, _ = batch
            else:
                x, y, _ = batch
                w = None

            optimizer.zero_grad()
            preds = model(x)

            if args.probabilistic:
                loss = pinball_loss(preds, y, w)
            else:
                loss = weighted_mse(
                    preds, y, w, under_penalty=args.under_penalty
                )

            accelerator.backward(loss)

            if args.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss_accum += loss.item() * x.size(0)
            train_samples_seen += x.size(0)

        local_avg_train = train_loss_accum / max(1, train_samples_seen)
        sync_train_tensor = torch.tensor(local_avg_train, device=device)
        avg_train_loss = accelerator.reduce(sync_train_tensor, reduction="mean").item()

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

                preds = model(x)

                if args.probabilistic:
                    loss = pinball_loss(preds, y, w)
                else:
                    loss = weighted_mse(
                        preds, y, w, under_penalty=args.under_penalty
                    )

                val_loss_accum += loss.item() * x.size(0)
                val_samples_seen += x.size(0)

        local_avg_val = val_loss_accum / max(1, val_samples_seen)
        sync_val_tensor = torch.tensor(local_avg_val, device=device)
        avg_val_loss = accelerator.reduce(sync_val_tensor, reduction="mean").item()

        epoch_duration = (datetime.now() - start_time).total_seconds()

        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            if accelerator.is_local_main_process:
                os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "model_state_dict": unwrapped_model.state_dict(),
                        "args": vars(args),
                        "input_size": input_size,
                        "best_val_loss": best_score,
                    },
                    args.checkpoint_path,
                )
            log_msg += " [Checkpoint Saved]"

        log_info(log_msg)

        delta = None
        if last_train_loss is not None:
            delta = abs(avg_train_loss - last_train_loss)
            if delta < TRAINING.LOSS_CHANGE_THRESHOLD:
                no_change_streak += 1
            else:
                no_change_streak = 0

        if hyperparam_optimizer != "sfoa":
            if (
                no_change_streak >= TRAINING.HYPERPARAM_CHECK_INTERVAL
                and epoch < args.epochs
            ):
                new_hyperparams = sample_hyperparams(rng, used_keys)
                if new_hyperparams is not None:
                    delta_display = f"{delta:.4f}" if delta is not None else "N/A"
                    log_info(
                        "Train loss change below threshold for "
                        f"{no_change_streak} consecutive epochs (Δ={delta_display}). "
                        "Switching hyperparameters."
                    )
                    log_info(f"New hyperparameters: {new_hyperparams}")

                    apply_hyperparams(args, new_hyperparams)

                    new_model = RNNForecaster(
                        input_size=input_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        horizon=args.pred_horizon,
                        rnn_type=args.rnn_type,
                        bidirectional=args.bidirectional,
                        quantiles=quantiles if args.probabilistic else None,
                    ).to(device)

                    new_optimizer = torch.optim.Adam(
                        new_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                    )

                    model, optimizer = accelerator.prepare(new_model, new_optimizer)

                    current_hyperparams = new_hyperparams
                    last_train_loss = None
                else:
                    log_info(
                        "No unused hyperparameter combinations remain; keeping current settings."
                    )
                    last_train_loss = avg_train_loss
                no_change_streak = 0
            else:
                last_train_loss = avg_train_loss
        else:
            last_train_loss = avg_train_loss

        if accelerator.is_local_main_process:
            resume_payload = {
                "epoch": epoch,
                "args": vars(args),
                "hyperparams": current_hyperparams,
                "sfoa_hyperparams": (
                    current_hyperparams if hyperparam_optimizer == "sfoa" else None
                ),
                "used_hyperparams": list(used_keys),
                "best_score": best_score,
                "last_train_loss": last_train_loss,
                "no_change_streak": no_change_streak,
                "log_path": log_path,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_resume_state(PATHS.RESUME_STATE_FILE, resume_payload)

    log_info("\n--- Training Completed ---")
    log_info(f"Best Validation Loss: {best_score:.4f}")
    log_info(f"Final Model Saved to: {args.checkpoint_path}")
    if log_path:
        log_info(f"Full Log Saved to:    {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--use_weights", action="store_true")
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--grad_clip", type=float, default=TRAINING.GRAD_CLIP)
    p.add_argument("--weight_decay", type=float, default=TRAINING.WEIGHT_DECAY)
    p.add_argument("--under_penalty", type=float, default=TRAINING.UNDER_PENALTY)
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument(
        "--resume_training",
        action="store_true",
    )
    p.add_argument(
        "--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL
    )
    p.add_argument(
        "--probabilistic",
        action="store_true",
        default=TRAINING.PROBABILISTIC_TRAINING,
    )
    p.add_argument(
        "--hyperparam_optimizer",
        default=TRAINING.HYPERPARAM_OPTIMIZER,
        choices=["random", "sfoa"],
    )

    try:
        train(p.parse_args())
    except Exception as e:
        logging.error("Fatal Error during training", exc_info=True)
        sys.exit(1)

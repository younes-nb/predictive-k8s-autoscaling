import copy
import logging
import math as _math
import os
import time as _time

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.config_paths import PATHS
from shared.config_training_defaults import TRAINING
from shared.logging_utils import setup_logging
from .loss import weighted_mse
from .train_helpers import load_resume_state, save_resume_state


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m}m{s}s"


class SFOAOptimizer:

    D = 4
    BOUNDS_LOW = None
    BOUNDS_HIGH = None

    def __init__(
        self, eval_fn, N, Tmax, Gp, seed=42, parallel_eval=False, accelerator=None
    ):
        if N < 3:
            raise ValueError("SFOA requires N >= 3 to sample distinct distances.")
        self.eval_fn = eval_fn
        self.N = int(N)
        self.Tmax = int(Tmax)
        self.Gp = float(Gp)
        self.parallel_eval = bool(parallel_eval)
        self.accelerator = accelerator
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
        def clip(val, low, high):
            return max(low, min(high, float(val)))

        idx_hidden = int(round(clip(x[0], self.BOUNDS_LOW[0], self.BOUNDS_HIGH[0])))
        idx_layers = int(round(clip(x[1], self.BOUNDS_LOW[1], self.BOUNDS_HIGH[1])))
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
        def _evaluate_one(idx: int, row: np.ndarray) -> float:
            hyperparams = self._decode(row)
            try:
                return float(
                    self.eval_fn(copy.deepcopy(hyperparams), candidate_idx=idx)
                )
            except Exception as exc:
                logging.warning(
                    "[SFOA] Evaluation failed for candidate #%d (%s): %s",
                    idx,
                    hyperparams,
                    exc,
                )
                return float("inf")

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            num_procs = self.accelerator.num_processes
            rank = self.accelerator.process_index

            local_fitness = np.full(len(X), float("inf"), dtype=float)
            rank_t0 = _time.time()
            for i in range(len(X)):
                if i % num_procs == rank:
                    cand_t0 = _time.time()
                    local_fitness[i] = _evaluate_one(i, X[i])
                    elapsed = _time.time() - cand_t0
                    logging.info(
                        "[SFOA-R%d] Candidate %d evaluated in %.1fs (inf=%d/%d)",
                        rank, i, elapsed,
                        int(np.sum(np.isinf(local_fitness))), len(X),
                    )

            try:
                import torch.distributed as dist

                device = self.accelerator.device
                local_tensor = torch.full((len(X),), float("nan"), device=device)
                for i in range(len(X)):
                    if not np.isinf(local_fitness[i]):
                        local_tensor[i] = local_fitness[i]

                all_tensors = [torch.empty_like(local_tensor) for _ in range(num_procs)]
                dist.all_gather(all_tensors, local_tensor)

                global_fitness = np.full(len(X), float("inf"), dtype=float)
                for j, t in enumerate(all_tensors):
                    vals = t.cpu().numpy()
                    mask = ~np.isnan(vals)
                    global_fitness[mask] = vals[mask]

                gather_elapsed = _time.time() - rank_t0
                logging.info(
                    "[SFOA-R%d] all_gather OK in %.1fs — global fitness: min=%.6f @ idx=%d",
                    rank, gather_elapsed,
                    float(np.min(global_fitness[~np.isinf(global_fitness)]))
                    if np.any(~np.isinf(global_fitness)) else float("inf"),
                    int(np.argmin(global_fitness)),
                )
                return global_fitness
            except Exception as e:
                elapsed = _time.time() - rank_t0
                inf_count = int(np.sum(np.isinf(local_fitness)))
                logging.error(
                    "[SFOA-R%d] all_gather FAILED after %.1fs. "
                    "Local fitness: %d/%d inf, CUDA alloc=%.0fMB, reserved=%.0fMB",
                    rank, elapsed, inf_count, len(X),
                    torch.cuda.memory_allocated(device) / 1e6
                    if torch.cuda.is_available() else 0,
                    torch.cuda.memory_reserved(device) / 1e6
                    if torch.cuda.is_available() else 0,
                )
                return local_fitness

        if not self.parallel_eval or len(X) <= 1:
            return np.asarray(
                [_evaluate_one(i, row) for i, row in enumerate(X)], dtype=float
            )

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        results_map: dict[int, float] = {}

        def _worker(idx: int, row: np.ndarray) -> tuple[int, float]:
            return idx, _evaluate_one(idx, row)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=n_gpu) as pool:
            futures = {pool.submit(_worker, i, row): i for i, row in enumerate(X)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, fit = fut.result()
                    results_map[idx] = fit
                except Exception as exc:
                    logging.warning(
                        "[SFOA] Parallel candidate #%d raised: %s", idx, exc
                    )
                    results_map[idx] = float("inf")

        return np.asarray([results_map[i] for i in range(len(X))], dtype=float)

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
            y_p = Et * X[i, p] + A1 * (X[k1, p] - X[i, p]) + A2 * (X[k2, p] - X[i, p])
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
                X = (
                    self.rng.random((self.N, self.D))
                    * (self.BOUNDS_HIGH - self.BOUNDS_LOW)
                    + self.BOUNDS_LOW
                )
        else:
            X = (
                self.rng.random((self.N, self.D)) * (self.BOUNDS_HIGH - self.BOUNDS_LOW)
                + self.BOUNDS_LOW
            )

        iter_t0 = _time.time()
        fitness = self._evaluate_all(X)
        logging.info("[SFOA] Initial evaluation done in %s", _format_elapsed(_time.time() - iter_t0))
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
                mode = "explore" if rand_val > self.Gp else "exploit"

                logging.info(
                    "[SFOA] Iteration %d/%d (%s) — evaluating %d candidates",
                    T, self.Tmax, mode, self.N,
                )

                if rand_val > self.Gp:
                    X = self._explore(X, T)
                else:
                    X = self._exploit(X, best_pos, T)

                iter_t0 = _time.time()
                fitness = self._evaluate_all(X)
                logging.info(
                    "[SFOA] Iteration %d/%d (%s) done in %s",
                    T, self.Tmax, mode, _format_elapsed(_time.time() - iter_t0),
                )

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


def run_sfoa_search(
    args,
    train_ds,
    val_ds,
    *,
    rank_seed: int = 42,
    accelerator=None,
) -> dict:
    import torch.distributed as dist
    from core.models import RNNForecaster

    is_distributed = accelerator is not None and accelerator.num_processes > 1
    rank = accelerator.process_index if (is_distributed and accelerator is not None) else 0

    logging.info("[SFOA-R%d] GPU health check starting...", rank)
    
    if len(train_ds) > 0:
        first_x, *_ = train_ds[0]
        input_size = first_x.shape[-1]
    else:
        raise RuntimeError("[SFOA] train_ds is empty — cannot derive input_size.")
        
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            mem_alloc_gb = torch.cuda.memory_allocated(0) / 1e9
            mem_free_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / 1e9
            logging.info(
                "[SFOA-R%d] GPU: %s | alloc=%.1fGB free=%.1fGB",
                rank, props.name, mem_alloc_gb, mem_free_gb,
            )
        except Exception as e:
            logging.error("[SFOA-R%d] GPU health check FAILED on device 0: %s", rank, e)
            raise RuntimeError(f"GPU 0 health check failed on rank {rank}") from e
    else:
        logging.info("[SFOA-R%d] Running on CPU (no CUDA)", rank)
    logging.info("[SFOA-R%d] GPU health check passed.", rank)

    if is_distributed and accelerator is not None:
        eval_device = accelerator.device
    elif torch.cuda.is_available():
        eval_device = torch.device("cuda:0")
    else:
        eval_device = torch.device("cpu")

    eval_counter = {"count": 0}

    def _decode_and_log(hyperparams: dict, candidate_idx: int) -> str:
        return (
            f"cand#{candidate_idx} "
            f"hidden={hyperparams['hidden_size']} "
            f"layers={hyperparams['num_layers']} "
            f"dropout={hyperparams['dropout']:.4f} "
            f"lr={hyperparams['lr']:.8f}"
        )

    def eval_fn(hyperparams: dict, *, candidate_idx: int = 0) -> float:
        model = None
        optimizer = None
        eval_counter["count"] += 1
        system_cores = os.cpu_count() or 1
        gpu_count = torch.cuda.device_count() or 1
        workers = min(system_cores, 4 * gpu_count)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

        torch.manual_seed(rank_seed + candidate_idx)
        label = _decode_and_log(hyperparams, candidate_idx)

        is_main_rank = (
            accelerator.is_local_main_process
            if (accelerator is not None and is_distributed)
            else True
        )
        if is_main_rank:
            logging.info(
                "[SFOA] cand#%d | START %s | device=%s",
                candidate_idx, label, eval_device,
            )

        try:
            logging.info("[SFOA] cand#%d before model creation", candidate_idx)
            model = RNNForecaster(
                input_size=input_size,
                hidden_size=hyperparams["hidden_size"],
                num_layers=hyperparams["num_layers"],
                dropout=hyperparams["dropout"],
                horizon=args.pred_horizon,
                rnn_type=args.rnn_type,
                bidirectional=args.bidirectional,
                quantiles=None,
            ).to(eval_device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams["lr"],
                weight_decay=args.weight_decay,
            )
            
            for epoch in range(TRAINING.SFOA_EVAL_EPOCHS):
                model.train()
                epoch_loss_sum = 0.0
                epoch_batches = 0
                epoch_start = _time.time()
                for batch in train_loader:
                    if args.use_weights:
                        x, y, w, _ = batch
                    else:
                        x, y, _ = batch
                        w = None
                    x = x.to(eval_device)
                    y = y.to(eval_device)
                    if w is not None:
                        w = w.to(eval_device)

                    optimizer.zero_grad()
                    preds = model(x)
                    loss = weighted_mse(preds, y, w, under_penalty=args.under_penalty)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_sum += loss.item() * x.size(0)
                    epoch_batches += 1

                avg_train = epoch_loss_sum / max(1, epoch_batches)
                epoch_duration = _time.time() - epoch_start

                model.eval()
                val_accum = 0.0
                val_cnt = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if args.use_weights:
                            xb, yb, wb, _ = batch
                        else:
                            xb, yb, _ = batch
                            wb = None
                        xb = xb.to(eval_device)
                        yb = yb.to(eval_device)
                        if wb is not None:
                            wb = wb.to(eval_device)
                        preds = model(xb)
                        vloss = weighted_mse(preds, yb, wb, under_penalty=args.under_penalty)
                        val_accum += vloss.item() * xb.size(0)
                        val_cnt += xb.size(0)
                avg_val = val_accum / max(1, val_cnt)

                is_main_rank = (
                    accelerator.is_local_main_process
                    if (accelerator is not None and is_distributed)
                    else True
                )
                if is_main_rank:
                    logging.info(
                        "[SFOA] cand#%d %s | device=%s | epoch %d/%d | "
                        "train_loss=%.6f val_loss=%.6f | %.2fs",
                        candidate_idx, label, eval_device,
                        epoch + 1, TRAINING.SFOA_EVAL_EPOCHS,
                        avg_train, avg_val, epoch_duration,
                    )
                else:
                    if epoch == TRAINING.SFOA_EVAL_EPOCHS - 1:
                        logging.info(
                            "[SFOA-R%d] cand#%d training complete (val_loss=%.6f)",
                            rank, candidate_idx, avg_val,
                        )

            if is_main_rank:
                logging.info(
                    "[SFOA] cand#%d | device=%s | DONE val_loss=%.6f",
                    candidate_idx, eval_device, avg_val,
                )
            return avg_val

        except Exception as exc:
            logging.warning(
                "[SFOA] Evaluation failed for %s: %s", label, exc, exc_info=True
            )
            return float("inf")
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_ranks = accelerator.num_processes if (is_distributed and accelerator is not None) else 1
    logging.info(
        "[SFOA] Starting search: %d rank(s), N=%d, Tmax=%d, eval_epochs=%d",
        n_ranks,
        TRAINING.SFOA_POPULATION,
        TRAINING.SFOA_ITERATIONS,
        TRAINING.SFOA_EVAL_EPOCHS,
    )

    logging.info(
        "[SFOA] Evaluating candidates in parallel across ranks (DDP all_gather)",
    )
    
    if eval_device.type != "cpu":
        logging.info("[SFOA] Profiling worst-case model architecture boundaries to establish global SFOA batch size...")
        from training.train_helpers import find_max_batch_size
        
        max_possible_hidden = max(TRAINING.HIDDEN_SIZE_OPTIONS)
        max_possible_layers = max(TRAINING.NUM_LAYERS_OPTIONS)
        
        worst_case_model = RNNForecaster(
            input_size=input_size,
            hidden_size=max_possible_hidden,
            num_layers=max_possible_layers,
            dropout=0.0,
            horizon=args.pred_horizon,
            rnn_type=args.rnn_type,
            bidirectional=args.bidirectional,
            quantiles=None,
        ).to(eval_device)
        
        max_batch = find_max_batch_size(worst_case_model, input_size, args, eval_device, loss_fn=None)
        safe_batch_size = int(max_batch * 0.8)
        args.batch_size = 2 ** int(_math.log2(max(1, safe_batch_size)))
        
        del worst_case_model
        torch.cuda.empty_cache()
        logging.info("[SFOA] Global SFOA batch size set to %d based on hardware ceilings.", args.batch_size)

    sfoa = SFOAOptimizer(
        eval_fn=eval_fn,
        N=TRAINING.SFOA_POPULATION,
        Tmax=TRAINING.SFOA_ITERATIONS,
        Gp=TRAINING.SFOA_GP,
        seed=rank_seed,
        parallel_eval=True,
        accelerator=accelerator if is_distributed else None,
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
        existing["sfoa_done"] = True
        existing["hyperparam_optimizer"] = "sfoa"
        save_resume_state(PATHS.RESUME_STATE_FILE, existing)
    except Exception as exc:
        logging.warning("Could not save SFOA state: %s", exc)

    logging.info(
        "[SFOA] Search complete. Best val loss: %.6f | Params: %s",
        best_fitness,
        best_hp,
    )

    if is_distributed:
        try:
            dist.barrier()
        except Exception as e:
            logging.error("[SFOA-R%d] Barrier before broadcast failed: %s", rank, e)
            raise
        best_hp_list = [best_hp]
        dist.broadcast_object_list(best_hp_list, src=0)
        best_hp = best_hp_list[0]

    return best_hp

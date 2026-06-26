import copy
import logging
import math as _math
import os
import time as _time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from shared.config_paths import PATHS
from shared.config_training_defaults import TRAINING
from shared.features import target_features_for_feature_set
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


def _to_cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _move_optimizer_state_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _int_arg(args, name: str, default) -> int:
    try:
        return int(getattr(args, name, default))
    except (TypeError, ValueError):
        return int(default)


def _use_file_system_tensor_sharing() -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError as exc:
        logging.warning("[SFOA] Could not set torch sharing strategy: %s", exc)


def _head_slice_dataset(dataset, max_samples: int):
    total = len(dataset)
    if max_samples <= 0 or total <= max_samples:
        return dataset
    return Subset(dataset, range(int(max_samples)))


def _same_hyperparams(left, right) -> bool:
    if not left or not right:
        return False
    try:
        return (
            int(left.get("hidden_size")) == int(right.get("hidden_size"))
            and int(left.get("num_layers")) == int(right.get("num_layers"))
            and round(float(left.get("dropout")), 4) == round(float(right.get("dropout")), 4)
            and round(float(left.get("lr")), 8) == round(float(right.get("lr")), 8)
        )
    except (TypeError, ValueError):
        return False


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

    def _evaluate_all(
        self,
        X: np.ndarray,
        *,
        fitness=None,
        T: int = 0,
        completed_T: int = 0,
        best_pos=None,
        best_fitness: float = float("inf"),
        phase: str = "initial",
        mode=None,
        checkpoint_fn=None,
        initial_state: dict = None,
    ) -> np.ndarray:
        if fitness is not None and len(fitness) == len(X):
            current_fitness = np.asarray(fitness, dtype=float).copy()
        else:
            current_fitness = np.full(len(X), np.nan, dtype=float)

        def _candidate_payload_from_resume(candidate_resume: dict, hyperparams: dict) -> dict:
            payload = {
                "candidate_hyperparams": copy.deepcopy(hyperparams),
                "candidate_epoch": 0,
            }
            if not candidate_resume:
                return payload
            for key in (
                "candidate_epoch",
                "candidate_val_loss",
                "candidate_model_state_dict",
                "candidate_optimizer_state_dict",
                "candidate_torch_rng_state",
                "candidate_cuda_rng_state",
            ):
                if key in candidate_resume:
                    payload[key] = candidate_resume[key]
            return payload

        def _checkpoint(active_candidate=None, candidate_payload=None) -> None:
            if checkpoint_fn is None:
                return
            interim_state = {
                "X": X.copy(),
                "fitness": current_fitness.copy(),
                "best_pos": None if best_pos is None else np.asarray(best_pos, dtype=float).copy(),
                "best_fitness": float(best_fitness),
                "T": int(completed_T),
                "active_T": int(T),
                "active_candidate": None if active_candidate is None else int(active_candidate),
                "phase": phase,
                "mode": mode,
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "N": self.N,
                "Tmax": self.Tmax,
            }
            if candidate_payload:
                interim_state.update(candidate_payload)
            checkpoint_fn(interim_state)

        def _candidate_resume_state(idx: int):
            if not isinstance(initial_state, dict):
                return None
            if initial_state.get("active_T") is None:
                return None
            if int(initial_state.get("active_T", -1)) != int(T):
                return None
            if initial_state.get("active_candidate") is None:
                return None
            if int(initial_state.get("active_candidate", -1)) != int(idx):
                return None
            if not np.isnan(current_fitness[idx]):
                return None
            return initial_state

        def _evaluate_one(idx: int, row: np.ndarray) -> float:
            if not np.isnan(current_fitness[idx]):
                return float(current_fitness[idx])
            hyperparams = self._decode(row)
            candidate_resume = _candidate_resume_state(idx)

            def _candidate_checkpoint(candidate_payload: dict) -> None:
                payload = {"candidate_hyperparams": copy.deepcopy(hyperparams)}
                if candidate_payload:
                    payload.update(candidate_payload)
                _checkpoint(active_candidate=idx, candidate_payload=payload)

            try:
                _checkpoint(
                    active_candidate=idx,
                    candidate_payload=_candidate_payload_from_resume(
                        candidate_resume, hyperparams
                    ),
                )
                return float(
                    self.eval_fn(
                        copy.deepcopy(hyperparams),
                        candidate_idx=idx,
                        candidate_state=candidate_resume,
                        checkpoint_fn=_candidate_checkpoint,
                    )
                )
            except Exception as exc:
                logging.warning(
                    "[SFOA] Evaluation failed for candidate #%d (%s): %s",
                    idx,
                    hyperparams,
                    exc,
                )
                return float("inf")

        _checkpoint()

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            num_procs = self.accelerator.num_processes
            rank = self.accelerator.process_index

            local_fitness = current_fitness.copy()
            rank_t0 = _time.time()
            for i in range(len(X)):
                if np.isnan(local_fitness[i]) and i % num_procs == rank:
                    cand_t0 = _time.time()
                    local_fitness[i] = _evaluate_one(i, X[i])
                    current_fitness[i] = local_fitness[i]
                    _checkpoint()
                    elapsed = _time.time() - cand_t0
                    logging.info(
                        "[SFOA-R%d] Candidate %d evaluated in %.1fs (pending=%d/%d)",
                        rank, i, elapsed,
                        int(np.sum(np.isnan(local_fitness))), len(X),
                    )

            try:
                import torch.distributed as dist

                device = self.accelerator.device
                local_tensor = torch.as_tensor(local_fitness, dtype=torch.float64, device=device)

                all_tensors = [torch.empty_like(local_tensor) for _ in range(num_procs)]
                dist.all_gather(all_tensors, local_tensor)

                global_fitness = np.full(len(X), np.nan, dtype=float)
                for j, t in enumerate(all_tensors):
                    vals = t.cpu().numpy()
                    mask = ~np.isnan(vals)
                    global_fitness[mask] = vals[mask]

                current_fitness = global_fitness
                _checkpoint()

                gather_elapsed = _time.time() - rank_t0
                available = ~np.isnan(global_fitness)
                logging.info(
                    "[SFOA-R%d] all_gather OK in %.1fs - global fitness: min=%.6f @ idx=%d",
                    rank, gather_elapsed,
                    float(np.nanmin(global_fitness[available]))
                    if np.any(available) else float("inf"),
                    int(np.nanargmin(global_fitness)) if np.any(available) else -1,
                )
                return global_fitness
            except Exception as e:
                elapsed = _time.time() - rank_t0
                pending_count = int(np.sum(np.isnan(local_fitness)))
                logging.error(
                    "[SFOA-R%d] all_gather FAILED after %.1fs. "
                    "Local fitness: %d/%d pending, CUDA alloc=%.0fMB, reserved=%.0fMB",
                    rank, elapsed, pending_count, len(X),
                    torch.cuda.memory_allocated(device) / 1e6
                    if torch.cuda.is_available() else 0,
                    torch.cuda.memory_reserved(device) / 1e6
                    if torch.cuda.is_available() else 0,
                )
                return local_fitness

        if checkpoint_fn is not None or not self.parallel_eval or len(X) <= 1:
            for i, row in enumerate(X):
                if np.isnan(current_fitness[i]):
                    current_fitness[i] = _evaluate_one(i, row)
                    _checkpoint()
            return current_fitness

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        results_map: dict[int, float] = {}
        pending_indices = [i for i in range(len(X)) if np.isnan(current_fitness[i])]
        if not pending_indices:
            return current_fitness

        def _worker(idx: int, row: np.ndarray) -> tuple[int, float]:
            return idx, _evaluate_one(idx, row)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max(1, n_gpu)) as pool:
            futures = {pool.submit(_worker, i, X[i]): i for i in pending_indices}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, fit = fut.result()
                    current_fitness[idx] = fit
                except Exception as exc:
                    logging.warning(
                        "[SFOA] Parallel candidate #%d raised: %s", idx, exc
                    )
                    current_fitness[idx] = float("inf")

        return current_fitness

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

    def optimize(self, initial_state: dict = None, checkpoint_fn=None) -> tuple[dict, float, dict]:
        if initial_state is not None and "X" in initial_state:
            X = np.asarray(initial_state["X"], dtype=float)
            if X.shape != (self.N, self.D):
                logging.warning(
                    "[SFOA] Resume state shape %s != (%d, %d); "
                    "reinitializing X and discarding stale state.",
                    X.shape, self.N, self.D,
                )
                X = (
                    self.rng.random((self.N, self.D))
                    * (self.BOUNDS_HIGH - self.BOUNDS_LOW)
                    + self.BOUNDS_LOW
                )
                initial_state = None  # shape mismatch: treat as fresh run
        else:
            X = (
                self.rng.random((self.N, self.D)) * (self.BOUNDS_HIGH - self.BOUNDS_LOW)
                + self.BOUNDS_LOW
            )

        if initial_state is not None and initial_state.get("rng_state") is not None:
            try:
                self.rng.bit_generator.state = initial_state["rng_state"]
            except Exception as exc:
                logging.warning("[SFOA] Could not restore RNG state: %s", exc)

        cached_fitness = initial_state.get("fitness") if initial_state is not None else None
        if cached_fitness is not None and len(cached_fitness) == len(X):
            fitness = np.asarray(cached_fitness, dtype=float).copy()
        else:
            fitness = np.full(len(X), np.nan, dtype=float)

        completed_T = int(initial_state.get("T", 0)) if initial_state is not None else 0
        active_T_raw = initial_state.get("active_T") if initial_state is not None else None
        active_T = int(active_T_raw) if active_T_raw is not None else None

        best_pos = None
        if initial_state is not None and initial_state.get("best_pos") is not None:
            best_pos = np.asarray(initial_state["best_pos"], dtype=float)
            if best_pos.shape != (self.D,):
                best_pos = None
        best_fitness = (
            float(initial_state.get("best_fitness", float("inf")))
            if initial_state is not None
            else float("inf")
        )

        def _fitness_complete(values: np.ndarray) -> bool:
            return len(values) == len(X) and not np.any(np.isnan(values))

        def _mark_pending_as_failed(values: np.ndarray) -> np.ndarray:
            pending = int(np.sum(np.isnan(values)))
            if pending:
                logging.warning(
                    "[SFOA] Population evaluation returned with %d pending candidate(s); marking them as inf.",
                    pending,
                )
                return np.where(np.isnan(values), float("inf"), values)
            return values

        def _refresh_best_from_fitness() -> None:
            nonlocal best_pos, best_fitness
            available = ~np.isnan(fitness)
            if not np.any(available):
                return
            idx = int(np.nanargmin(fitness))
            if best_pos is None or fitness[idx] < best_fitness:
                best_fitness = float(fitness[idx])
                best_pos = X[idx].copy()

        def _checkpoint_complete(done_T: int) -> None:
            if checkpoint_fn is None:
                return
            complete_state = {
                "X": X.copy(),
                "fitness": fitness.copy(),
                "best_pos": None if best_pos is None else best_pos.copy(),
                "best_fitness": float(best_fitness),
                "best_hyperparams": self._decode(best_pos) if best_pos is not None else None,
                "T": int(done_T),
                "active_T": None,
                "active_candidate": None,
                "phase": "complete",
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "N": self.N,
                "Tmax": self.Tmax,
            }
            try:
                checkpoint_fn(complete_state)
            except Exception as exc:
                logging.warning("[SFOA] Completed-state checkpoint failed at T=%d: %s", done_T, exc)

        if active_T is not None and not _fitness_complete(fitness):
            phase = initial_state.get("phase", "initial" if active_T == 0 else "iteration")
            mode = initial_state.get("mode")
            logging.info(
                "[SFOA] Resuming %s evaluation at T=%d candidate=%s epoch=%s.",
                phase,
                active_T,
                initial_state.get("active_candidate"),
                initial_state.get("candidate_epoch"),
            )
            iter_t0 = _time.time()
            fitness = self._evaluate_all(
                X,
                fitness=fitness,
                T=active_T,
                completed_T=completed_T,
                best_pos=best_pos,
                best_fitness=best_fitness,
                phase=phase,
                mode=mode,
                checkpoint_fn=checkpoint_fn,
                initial_state=initial_state,
            )
            fitness = _mark_pending_as_failed(fitness)
            logging.info(
                "[SFOA] Resumed T=%d evaluation done in %s",
                active_T,
                _format_elapsed(_time.time() - iter_t0),
            )
            _refresh_best_from_fitness()
            completed_T = active_T
            _checkpoint_complete(completed_T)
            initial_state = None
            active_T = None

        if active_T is not None and _fitness_complete(fitness):
            logging.info(
                "[SFOA] Resume state has completed active_T=%d; finalizing iteration state.",
                active_T,
            )
            _refresh_best_from_fitness()
            completed_T = active_T
            _checkpoint_complete(completed_T)
            initial_state = None
            active_T = None

        if not _fitness_complete(fitness):
            iter_t0 = _time.time()
            fitness = self._evaluate_all(
                X,
                fitness=fitness,
                T=0,
                completed_T=0,
                best_pos=best_pos,
                best_fitness=best_fitness,
                phase="initial",
                mode=None,
                checkpoint_fn=checkpoint_fn,
                initial_state=initial_state,
            )
            fitness = _mark_pending_as_failed(fitness)
            logging.info("[SFOA] Initial evaluation done in %s", _format_elapsed(_time.time() - iter_t0))
            _refresh_best_from_fitness()
            completed_T = 0
            _checkpoint_complete(completed_T)
            initial_state = None
        else:
            _refresh_best_from_fitness()
            if initial_state is not None:
                logging.info("[SFOA] Reusing cached fitness from resume state.")

        if completed_T >= self.Tmax:
            logging.warning(
                "[SFOA] Resume state T=%d >= Tmax=%d - SFOA is already complete. "
                "Returning best known result without iterating. "
                "If you intended a fresh run, set resume=False (or delete %s).",
                completed_T, self.Tmax, PATHS.RESUME_STATE_FILE,
            )

        if best_pos is None:
            best_pos = X[int(np.nanargmin(fitness))].copy()
            best_fitness = float(np.nanmin(fitness))

        for T in range(completed_T + 1, self.Tmax + 1):
            rand_val = self.rng.uniform(0, 1)
            mode = "explore" if rand_val > self.Gp else "exploit"

            logging.info(
                "[SFOA] Iteration %d/%d (%s) - evaluating %d candidates",
                T, self.Tmax, mode, self.N,
            )

            if rand_val > self.Gp:
                X = self._explore(X, T)
            else:
                X = self._exploit(X, best_pos, T)

            fitness = np.full(len(X), np.nan, dtype=float)
            iter_t0 = _time.time()
            fitness = self._evaluate_all(
                X,
                fitness=fitness,
                T=T,
                completed_T=T - 1,
                best_pos=best_pos,
                best_fitness=best_fitness,
                phase="iteration",
                mode=mode,
                checkpoint_fn=checkpoint_fn,
                initial_state=None,
            )
            fitness = _mark_pending_as_failed(fitness)
            logging.info(
                "[SFOA] Iteration %d/%d (%s) done in %s",
                T, self.Tmax, mode, _format_elapsed(_time.time() - iter_t0),
            )

            _refresh_best_from_fitness()
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
            completed_T = T
            _checkpoint_complete(completed_T)

        final_state = {
            "X": X,
            "fitness": fitness,
            "best_pos": best_pos,
            "best_fitness": best_fitness,
            "best_hyperparams": self._decode(best_pos),
            "T": completed_T,
            "active_T": None,
            "active_candidate": None,
            "phase": "complete",
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
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
    resume: bool = False,
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

    _use_file_system_tensor_sharing()

    sfoa_train_pct = float(getattr(args, "sfoa_train_pct", TRAINING.SFOA_TRAIN_PCT))
    sfoa_val_pct = float(getattr(args, "sfoa_val_pct", TRAINING.SFOA_VAL_PCT))
    sfoa_train_ds = _head_slice_dataset(
        train_ds, int(len(train_ds) * sfoa_train_pct / 100.0)
    )
    sfoa_val_ds = _head_slice_dataset(
        val_ds, int(len(val_ds) * sfoa_val_pct / 100.0)
    )

    num_targets = len(target_features_for_feature_set(args.feature_set))

    system_cores = os.cpu_count() or 1
    sfoa_workers = max(
        0,
        min(system_cores, _int_arg(args, "sfoa_num_workers", TRAINING.SFOA_NUM_WORKERS)),
    )
    logging.info(
        "[SFOA] Evaluation data: train=%d/%d val=%d/%d samples | "
        "num_workers=%d | shuffle=False",
        len(sfoa_train_ds),
        len(train_ds),
        len(sfoa_val_ds),
        len(val_ds),
        sfoa_workers,
    )

    eval_counter = {"count": 0}

    def _decode_and_log(hyperparams: dict, candidate_idx: int) -> str:
        return (
            f"cand#{candidate_idx} "
            f"hidden={hyperparams['hidden_size']} "
            f"layers={hyperparams['num_layers']} "
            f"dropout={hyperparams['dropout']:.4f} "
            f"lr={hyperparams['lr']:.8f}"
        )

    def eval_fn(
        hyperparams: dict,
        *,
        candidate_idx: int = 0,
        candidate_state: dict = None,
        checkpoint_fn=None,
    ) -> float:
        model = None
        optimizer = None
        eval_counter["count"] += 1
        pin_memory = eval_device.type == "cuda"

        train_loader = DataLoader(
            sfoa_train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=sfoa_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            sfoa_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=sfoa_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )

        torch.manual_seed(rank_seed + candidate_idx)
        label = _decode_and_log(hyperparams, candidate_idx)
        saved_hyperparams = (
            candidate_state.get("candidate_hyperparams")
            if isinstance(candidate_state, dict)
            else None
        )
        can_resume_candidate = isinstance(candidate_state, dict) and (
            saved_hyperparams is None or _same_hyperparams(saved_hyperparams, hyperparams)
        )
        if isinstance(candidate_state, dict) and not can_resume_candidate:
            logging.warning(
                "[SFOA] Ignoring candidate resume state for cand#%d because hyperparameters differ.",
                candidate_idx,
            )

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
            model = RNNForecaster(
                input_size=input_size,
                hidden_size=hyperparams["hidden_size"],
                num_layers=hyperparams["num_layers"],
                dropout=hyperparams["dropout"],
                horizon=args.pred_horizon,
                rnn_type=args.rnn_type,
                bidirectional=args.bidirectional,
                num_targets=num_targets,
                quantiles=None,
            ).to(eval_device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams["lr"],
                weight_decay=args.weight_decay,
            )

            resume_epoch = 0
            avg_val = None
            if can_resume_candidate:
                requested_epoch = int(candidate_state.get("candidate_epoch", 0) or 0)
                has_model_state = candidate_state.get("candidate_model_state_dict") is not None
                has_optimizer_state = candidate_state.get("candidate_optimizer_state_dict") is not None
                if requested_epoch > 0 and not (has_model_state and has_optimizer_state):
                    logging.warning(
                        "[SFOA] Candidate cand#%d resume state is missing model/optimizer state; restarting candidate.",
                        candidate_idx,
                    )
                else:
                    resume_epoch = min(requested_epoch, TRAINING.SFOA_EVAL_EPOCHS)
                    if has_model_state:
                        model.load_state_dict(candidate_state["candidate_model_state_dict"])
                    if has_optimizer_state:
                        optimizer.load_state_dict(candidate_state["candidate_optimizer_state_dict"])
                        _move_optimizer_state_to_device(optimizer, eval_device)
                    avg_val = candidate_state.get("candidate_val_loss")
                    if candidate_state.get("candidate_torch_rng_state") is not None:
                        torch.set_rng_state(candidate_state["candidate_torch_rng_state"])
                    if (
                        eval_device.type == "cuda"
                        and candidate_state.get("candidate_cuda_rng_state") is not None
                    ):
                        torch.cuda.set_rng_state(
                            candidate_state["candidate_cuda_rng_state"],
                            device=eval_device,
                        )
                    if resume_epoch > 0 and is_main_rank:
                        logging.info(
                            "[SFOA] cand#%d | resuming after epoch %d/%d",
                            candidate_idx,
                            resume_epoch,
                            TRAINING.SFOA_EVAL_EPOCHS,
                        )

            def _checkpoint_candidate(done_epoch: int, val_loss) -> None:
                if checkpoint_fn is None:
                    return
                payload = {
                    "candidate_epoch": int(done_epoch),
                    "candidate_val_loss": None if val_loss is None else float(val_loss),
                    "candidate_model_state_dict": _to_cpu_tree(model.state_dict()),
                    "candidate_optimizer_state_dict": _to_cpu_tree(optimizer.state_dict()),
                    "candidate_torch_rng_state": _to_cpu_tree(torch.get_rng_state()),
                }
                if eval_device.type == "cuda":
                    payload["candidate_cuda_rng_state"] = _to_cpu_tree(
                        torch.cuda.get_rng_state(eval_device)
                    )
                checkpoint_fn(payload)

            _checkpoint_candidate(resume_epoch, avg_val)

            if resume_epoch >= TRAINING.SFOA_EVAL_EPOCHS:
                if avg_val is None:
                    avg_val = float("inf")
                if is_main_rank:
                    logging.info(
                        "[SFOA] cand#%d | device=%s | DONE val_loss=%.6f",
                        candidate_idx, eval_device, avg_val,
                    )
                return float(avg_val)
            
            for epoch in range(resume_epoch, TRAINING.SFOA_EVAL_EPOCHS):
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

                _checkpoint_candidate(epoch + 1, avg_val)

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
            if 'train_loader' in locals():
                del train_loader
            if 'val_loader' in locals():
                del val_loader
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
            num_targets=num_targets,
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

    initial_state = None
    if resume:
        _rs = load_resume_state(PATHS.RESUME_STATE_FILE)
        if _rs and "sfoa_state" in _rs:
            initial_state = _rs["sfoa_state"]
            logging.info(
                "[SFOA-R%d] Resuming from saved state (T=%d, active_T=%s, candidate=%s, epoch=%s).",
                rank,
                initial_state.get("T", 0),
                initial_state.get("active_T"),
                initial_state.get("active_candidate"),
                initial_state.get("candidate_epoch"),
            )
        else:
            logging.info(
                "[SFOA] resume=True but no saved sfoa_state found; starting fresh."
            )
    else:
        logging.info(
            "[SFOA] resume=False — ignoring any saved sfoa_state and starting fresh."
        )

    # Keys that belong exclusively to the main training loop.
    # They must not survive in RESUME_STATE_FILE while SFOA is still running
    # or has just completed, because a subsequent --resume_training would
    # misinterpret stale epoch / model / log-path data from a prior run.
    _STALE_TRAINING_KEYS = (
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "log_path",
        "best_score",
        "last_train_loss",
        "no_change_streak",
        "hyperparams",
        "sfoa_hyperparams",
    )

    def _sfoa_checkpoint(sfoa_interim_state: dict) -> None:
        """
        Called by SFOAOptimizer.optimize() during population/candidate evaluation.
        Writes a partial sfoa_state so the run can be resumed if interrupted.
        Sets sfoa_done=False to prevent train.py from skipping SFOA on resume.
        Purges all stale main-training-loop keys from the file so that a
        --resume_training after an interrupt cannot accidentally pick up old
        epoch / model / log-path data.
        Only rank 0 writes; non-main ranks return immediately.
        """
        if is_distributed and not accelerator.is_local_main_process:
            return
        try:
            existing = load_resume_state(PATHS.RESUME_STATE_FILE) or {}
            existing["sfoa_state"]           = sfoa_interim_state
            existing["sfoa_done"]            = False   # still in progress
            existing["hyperparam_optimizer"] = "sfoa"
            for _k in _STALE_TRAINING_KEYS:
                existing.pop(_k, None)
            save_resume_state(PATHS.RESUME_STATE_FILE, existing)
            logging.info(
                "[SFOA] Search checkpoint saved: T=%s/%s active_T=%s candidate=%s epoch=%s/%s.",
                sfoa_interim_state.get("T", "?"),
                sfoa_interim_state.get("Tmax", "?"),
                sfoa_interim_state.get("active_T"),
                sfoa_interim_state.get("active_candidate"),
                sfoa_interim_state.get("candidate_epoch"),
                TRAINING.SFOA_EVAL_EPOCHS,
            )
        except Exception as _exc:
            logging.warning("[SFOA] Mid-search checkpoint write failed: %s", _exc)

    # In a multi-rank run, only rank 0 has read the resume state from disk.
    # Broadcast it so every rank constructs its SFOAOptimizer with the same
    # `X` / cached `fitness` / `T`. Without this, ranks 1..N-1 would treat
    # the run as fresh (full ~15h initial re-eval on random X) and the
    # round-robin `_evaluate_all` would mix correct and garbage candidate
    # fitnesses into one incoherent global vector.
    if is_distributed and initial_state is not None:
        if accelerator.is_local_main_process:
            _payload = [initial_state]
        else:
            _payload = [None]
        dist.broadcast_object_list(_payload, src=0)
        initial_state = _payload[0]
        logging.info(
            "[SFOA-R%d] Received broadcast initial_state (T=%d) from rank 0.",
            rank, initial_state.get("T", 0) if isinstance(initial_state, dict) else -1,
        )

    best_hp, best_fitness, sfoa_state = sfoa.optimize(
        initial_state=initial_state,
        checkpoint_fn=_sfoa_checkpoint,
    )

    try:
        if is_distributed:
            # Make sure all ranks have finished calling optimize() before any
            # rank touches the resume file. Only rank 0 reads-modifies-writes
            # so we don't race-clobber a half-written state.
            dist.barrier()
            if accelerator.is_local_main_process:
                existing = load_resume_state(PATHS.RESUME_STATE_FILE) or {}
                existing["sfoa_state"]           = sfoa_state
                existing["sfoa_done"]            = True
                existing["hyperparam_optimizer"] = "sfoa"
                for _k in _STALE_TRAINING_KEYS:
                    existing.pop(_k, None)
                existing["hyperparams"] = best_hp
                existing["sfoa_hyperparams"] = best_hp
                save_resume_state(PATHS.RESUME_STATE_FILE, existing)
            dist.barrier()
        else:
            existing = load_resume_state(PATHS.RESUME_STATE_FILE) or {}
            existing["sfoa_state"]           = sfoa_state
            existing["sfoa_done"]            = True
            existing["hyperparam_optimizer"] = "sfoa"
            for _k in _STALE_TRAINING_KEYS:
                existing.pop(_k, None)
            existing["hyperparams"] = best_hp
            existing["sfoa_hyperparams"] = best_hp
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

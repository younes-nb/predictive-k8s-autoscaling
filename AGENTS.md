# predictive-k8s-autoscaling — Agent Guide

## Entrypoints & execution

All scripts use `sys.path.insert(0, REPO_ROOT)` — not pip-installable. Always run from repo root.

```bash
# Full pipeline (preprocessing → training → evaluation)
python pipelines/full_pipeline.py [--start_date 0d0] [--end_date 7d0] [--feature_set cpu_mem_both] [--rnn_type lstm] [--hyperparam_optimizer sfoa] [--skip_*] [--cpu] [--resume_training] [--bidirectional] [--probabilistic] [--max_services N] [--train_pct 100] [--val_pct 100] [--test_pct 100] [--smote_tomek]

# Preprocessing only
python pipelines/preprocessing_pipeline.py [--start_date 0d0] [--end_date 7d0] [--feature_set cpu_mem_both] [--skip_*] [--max_services N] [--smote_tomek]

# Training only
python training/train.py --windows_dir <dir> [--checkpoint_path] [--feature_set] [--rnn_type] [--hyperparam_optimizer sfoa] [--resume_training] [--cpu]

# Evaluation only
python training/evaluate.py --windows_dir <dir> --checkpoint_path <path> [--test_pct] [--cpu]
```

Multi-GPU (>1 GPU): `full_pipeline.py` auto-detects and wraps training/evaluation with `accelerate launch`. CPU/1-GPU: uses plain `python`.

## Critical configuration quirks

- **`train_pct`/`val_pct`/`test_pct`**: values are raw percentages — 25 = 25%, not 0.25. 100 or ≤0 uses all samples.
- **`hyperparam_optimizer`**: defaults to `"sfoa"` (expensive population-based search). Set `--hyperparam_optimizer none` to skip search and use `TrainingDefaults` directly.
- **`use_weights`** defaults to `False` in `TrainingDefaults`. Enabled via `--use_weights` CLI flag AND `TRAINING.USE_WEIGHTS = True`.
- **Feature sets**: `cpu`, `cpu_mem`, `node_cpu_mem`, `cpu_mem_mcr`, `threshold_analysis`, `cpu_delta_upstream`, `cpu_mem_both` (default). `cpu_delta_upstream` requires callgraph data.
- Default `INPUT_LEN=60`, `PRED_HORIZON=15`, `STRIDE=10`.

## Data paths (from `shared/config_paths.py`)

| Data | Path |
|------|------|
| Raw traces | `/dataset/raw/{msresource,node,msrtmcre,mscallgraph}` |
| Parquet | `/dataset/parquet/{...}` |
| Threshold | `/dataset/threshold/{msresource,msrtmcre}` |
| Preprocessed windows | `/dataset/windows` (sharded `.npy` files) |
| Model checkpoint | `/proj/k8sautoscaledl-PG0/models/model_global.pt` |
| Resume state | `/proj/k8sautoscaledl-PG0/train_resume_state.pt` |
| Logs | `/proj/k8sautoscaledl-PG0/logs/` |

## Windowed dataset format

Numpy memory-mapped shards: `part-NNNNN_X_{train,val,test}.npy` + `_y_`, `_sid_`, `_w_` (optional weights) counterparts. Dataset (`core/dataset.py`) loads via `np.load(..., mmap_mode='r')`.

## Resume training

`--resume_training` saves per-epoch state (model weights, optimizer, epoch counter, hyperparams, best score) to `RESUME_STATE_FILE` after every epoch. On restart with `--resume_training`, continues from last saved epoch. CLI overrides for `batch_size`, `epochs`, `train_pct`, `val_pct`, `sfoa_*` take precedence over saved args.

## Deployment (K8s CustomPodAutoscaler)

1. Place trained model at `deploy/model.pt`
2. `deploy/build_and_push.sh` — builds Docker image and pushes to `docker.io/younesnb/predictive-k8s-autoscaler:v1.0.0`
3. `deploy/deploy_all_cpas.sh` — applies CPA CRDs for all deployments in `online-boutique` namespace (skips `loadgenerator`)

CPA container uses CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`). Configured via env vars: `PROMETHEUS_URL`, `FEATURE_SET`, `RESIDUAL`, `TARGET_DEPLOYMENT`, `TARGET_NAMESPACE`.

## Load testing

```bash
cd load_testing && bash run_test.sh  # runs locust against https://online-boutique.younesnb.linkpc.net
```

## Analytics

```bash
cd analytics && bash collect.sh  # kubectl cp experiment_metrics.csv from CPA pods
```

## Misc

- **Logging** uses `Asia/Tehran` timezone (falls back to system time if unavailable).
- **AMP (FP16)**: enabled automatically via accelerate when GPU is available. Controlled by `accelerate` config, not `TRAINING` defaults.
- **No CI, no tests framework, no pre-commit, no Makefile** in this repo.
- **Experiments sub-project** at `experiments/cvcbm/` has its own pipeline (`run_pipeline.py`) and config; uses separate data paths (`/dataset/cvcbm_preprocess`, `/proj/k8sautoscaledl-PG0/models/cvcbm`).

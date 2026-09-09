# AGENTS.md

## Run
- No packaging, lint, typecheck, or test suite (`requirements.txt` only; no `pyproject`/`setup.py`, no CI, no `pytest`/`conftest`).
- Run from repo root: `python pipelines/full_pipeline.py ...`, `python training/train.py ...`, `python training/evaluate.py ...`. Scripts prepend repo root to `sys.path`; do not add extra path hacks.
- Install deps with `pip install -r requirements.txt` (plus `load_testing/requirements.txt` only for locust).

## Config: env vars beat defaults
- `shared/config_env.py:get_env` + `shared/config_{paths,preprocessing,training}_defaults.py` are the source of truth; `config/defaults.py` just re-exports.
- Default paths are absolute cluster paths (`/dataset/...`, `/proj/k8sautoscaledl-PG0/...` in `shared/config_paths.py`). Override locally via env: `RAW_ROOT PARQUET_ROOT WINDOWS_DIR MODELS_DIR LOGS_DIR ANALYTICS_OUT_DIR RESUME_STATE_FILE`.
- Same pattern for hyperparams: `INPUT_LEN=128 PRED_HORIZON=5 STRIDE=5 FEATURE_SET=cpu_mem_both BATCH_SIZE=4096 EPOCHS=1000 SEED=42`, etc.

## Pipeline order and entrypoints
- Full flow: `pipelines/full_pipeline.py` → `pipelines/preprocessing_pipeline.py` (fetch → ingest per-table → `build_windows.py` → smoothing/swt/cskv) → `training/train.py` → `training/evaluate.py` → `analytics/simulate_alibaba_predictive_hpa.py`.
- `full_pipeline.py` auto-selects launcher: `accelerate launch` if >1 GPU, plain `python` if 0–1 GPU; pass `--cpu` to force CPU. `train.py`/`evaluate.py` use Accelerate (fp16 on GPU, deterministic algorithms on).
- Keep `feature_set` (see `shared/features.py:FEATURE_SETS`) and `preprocess_approach none|smoothing|swt|cskv` identical across stages. Checkpoint records both plus `args/hyperparams`; `evaluate.py` rebuilds the model from the checkpoint, not from CLI flags.
- `*_pct` flags (`--train_pct/--val_pct/--test_pct`, `--sfoa_*_pct`) are percentages 0–100 (`100` = all, `<=0` = all), not fractions. Slicing is a head-prefix (`train_helpers.head_slice_dataset_by_pct`).
- `HYPERPARAM_OPTIMIZER` defaults to `none`; `sfoa` runs population 10 × 5 iters × 10 epochs — expensive, use small `*_pct` first.

## Caching gotchas (windows)
- `build_windows.py` caches aggressively: `part-*.done` markers, `_service_arrays.npy` + `_service_index.json` (signature includes feature_set/targets/msname/input_len/parquet fingerprint). Changing splits/targets/service without `--recompute` reuses stale shards.
- Force rebuild with `--recompute` (build_windows), `--recompute_windows` / `--recompute_preprocessing` (pipelines). `full_pipeline.py` only auto-forces rebuild when `input_len` differs from default.
- Changing `--msname` against an existing cache fails fast ("not in the service-array cache") — delete `_service_arrays.*` or use a fresh `--windows_dir`.
- swt/cskv/smoothing outputs go to `<windows_dir>/<approach>/`; `train.py`/`evaluate.py` then need `--preprocess_dir` there (wired automatically by `full_pipeline.py`). swt levels default to 5/5 (`preprocessing/swt/config.py`); CLI `--swt_level/--mem_swt_level` overrides. `--preprocess_dir` is required for swt/cskv eval.

## Data quirks
- `core/dataset.py:ShardedWindowsDataset` reads `part-*_X_<split>.npy` + `_y_` + `_sid_` (+ `_ylast_` for persistence reference) and returns half-precision tensors; train casts to float when not fp16.
- `--csv_path` fast path skips fetch+ingest; mapping is `_CSV_COLUMN_MAP` in `preprocessing/build_windows.py`, timestamp format `%Y-%m-%d %H:%M:%S`, default tz `Asia/Tehran`. Services with timestamp gaps/dups or too-short series are skipped.
- Resume: `--resume_training` reads `PATHS.RESUME_STATE_FILE`; inspect with `python tooling/inspect_resume_state.py [--path ...]`. CLI restores saved args except a small override set (see `train.py` `_cli_overrides`).
- Logs go to `LOGS_DIR` as `<mode>_<timestamp>.log` (timestamps in Asia/Tehran).

## Verification (no test suite)
- Cheap check: `python training/smoke_test_batch_profile.py --windows_dir <dir> --num_samples 100000 --mode eval` (or `--mode train`).
- Focused runs: `--max_services N` + `--msname <svc>` + small `--train_pct/--val_pct/--test_pct`, `--split val`, `--inference_bench_samples N`.

## Deploy / load testing
- `deploy/build_and_push.sh` requires `deploy/model.pt` + `core/models.py`, builds `docker.io/younesnb/predictive-k8s-autoscaler:v1.0.0`. `deploy/deploy_all_cpas.sh` pins `FEATURE_SET=cpu_mem_both MODEL_TYPE=dpam PREPROCESS=swt WINDOW=128 HORIZON=5` per deployment (skips `loadgenerator`, `redis-cart`).
- `load_testing/run_test.sh` requires `MCR_CSV` (from `analytics/analyze_http_mcr_oscillation.py`); tune via `TARGET_URL MAX_REQUESTS USER_POOL START_HOURS TEST_HOURS`.

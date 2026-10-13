# Madu workload predictor (SoCC'22 — training & evaluation only)

Implements the workload-learning half of Madu ("The Power of Prediction:
Microservice Auto Scaling via Workload Learning"). Deployment/scaling
-utilization analysis (Eq.11-12), autoscaling optimization (Eq.13-14),
performance profiling (Eq.9-10) — is intentionally out of scope.

## Paper mapping

| Paper | Code |
|---|---|
| Features x_t = {C_t, t_m, t_h, t_w} (§4.1.2) | `madu/data.py`: CPM + minute/hour/day-of-week, §4.1.2 |
| Seq2Seq encoder–decoder, n=60, m=5 (Fig.7, §5.1.1) | `madu/model.py: MaduSeq2Seq`, `--n/--m` |
| Gaussian NLL loss, Eq.(1) (§4.1.3) | `madu/loss.py: gaussian_nll` |
| Stochastic attention, Eq.(3)-(6) (§4.1.4) | `model.py`, `--attention stochastic` (sampled in train, mean in eval); `deterministic`/`none` reproduce Fig.12 ablations |
| Final prediction Ŷ = μ + σ, Eq.(8) (§4.1.6) | `model.py: predict` (clamped at 0) |
| Individual model per microservice (§6.2.1) | `train.py --msname/--top_k`, checkpoint per MS |
| Accuracy Eq.(15) by CPM percentile, Table 2 | `metrics.py: evaluate_by_percentile`, `evaluate.py` (per-MS tables + cross-MS summary; Table-2 Avg row = mean over MSs of pooled `overall` accuracy) |
| Accurate ±10% (Fig.9a), overestimation Eq.(16) (Fig.9b) | `metrics.py` |
| Baselines ARIMA / Seq2Seq (Table 1) | `baselines.py` (ARIMA), `--no_uncertainty` (deterministic Seq2Seq); persistence naive included |

## Data

CPM = received calls/minute/MS from `/dataset/ms_call_counts/per_minute`
(`minute = timestamp_ms // 60000`, all rpctypes incl. USER/MQ summed,
UM/DM double-logging deduped). Missing (minute, dm) rows mean 0 calls.

## Usage

```bash
python madu/train.py --msname MS_37691 --epochs 30          # 1 GPU used if present
python madu/train.py --top_k 5 --cpu --epochs 2 --max_minutes 2000  # quick test
python madu/train.py --msname MS_37691 --no_uncertainty --attention none  # Seq2Seq baseline
python madu/evaluate.py --checkpoint /proj/k8sautoscaledl-PG0/madu/madu_MS_37691.pt
python madu/evaluate.py --checkpoint_dir /proj/k8sautoscaledl-PG0/madu --no-arima
```

## Full 100-MS pipeline (pipelines/madu_pipeline.py)

```bash
python pipelines/madu_pipeline.py --epochs 15            # select→aggregate→train→profile→evaluate→plot
python pipelines/madu_pipeline.py --skip_training --skip_testing  # rerun plots only
```

Stages: `madu/select_ms.py` (top-K by CPM coefficient of variation, needs
msresource coverage for §4.2) → `madu/aggregate_util.py` (per-minute mean
CPU/mem + container counts from msresource) → `madu/train.py` (one model
per MS) → `madu/profiling.py` (OLS fit of Eq.9-10 on train minutes) →
`madu/evaluate.py` → per-MS CPM/CPU/memory trace plots via
`analytics/evaluate_trace.py --madu_ckpt` (Madu trace mode, h=1 rollout:
row at minute m carries actuals at m and the forecast for m+1; predicted
util uses actual container counts at m+1).

## Deviations from the paper (documented)

- `t_w` is day-of-week (0-6) from the minute index (user instruction);
  the paper writes "week number". Trace start weekday is unknown.
- Split is chronological 70/10/20 — paper uses 30-day train / 10-day test,
  but our trace spans ~13 days (minutes 0-18719).
- Eq.(15) divides by raw CPM; a floor `max(y, 1)` avoids /0 on idle minutes.
  Median accuracy + MAE are reported alongside (mean is outlier-dominated
  when near-zero minutes exist — the paper's busy MSs did not have these).
- Normalization stats (max / log1p mean-std) use train input minutes
  only (`full[:n_train + n]`); extending the window to `n+m` would peek at
  the first validation minutes through window overlap.
- `evaluate.py --out_json` writes `{"per_ms": [...], "summary": {...}}`;
  `summary` holds the Table-2 cross-MS means (group cells + Avg row).
- ARIMA is `(p,1,0)` (default `(5,1,0)`), statsmodels fit every
  `--arima_refit` windows on trailing history, frozen AR params recursed
  between refits (exact for multi-step AR); per-window refits are infeasible.
- BNN/DUBNN baselines omitted (prior work, not the paper's contribution).
- Training hyperparameters follow `shared/config_training_defaults.py`
  (`--hidden 64 --layers 3 --dropout 0.5 --batch_size 4096 --lr 5e-4
  --epochs 1000 --patience 20`, ReduceLROnPlateau scheduler,
  weight decay 1e-4, seed 42; `--min_delta 1e-4` per user instruction,
  overriding the config's 1e-6). The paper specifies none of these
  (only a 100-3000 training-step curve); the 64/3-layer model has
  ~127k params.
- Added (beyond paper): `--transform log1p` (z-scored log1p CPM; required
  for bursty MSs where max-scaling collapses typical values to ~1e-5),
  `--feat full` (+cyclic time + rolling mean/std), `--resample N
  --resample_how {mean,max}` (N-minute bins; max-bins capture peaks and
  are predictable where single minutes are not), `--mu_only` eval
  (μ head alone; Eq.8's +σ systematically overestimates bursty MSs),
  and a pooled `overall` metric row (mean-of-group-medians overweights
  tiny groups).

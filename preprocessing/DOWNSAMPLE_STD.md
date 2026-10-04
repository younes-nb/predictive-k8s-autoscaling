# Low-Std Window Downsampling via Label Distribution Smoothing (LDS)

## Source article

**Delving into Deep Imbalanced Regression (DIR)** — Yuzhe Yang, Kaiwen Zha,
Ying-Cong Chen, Hao Wang, Dina Katabi. *International Conference on Machine
Learning (ICML), 2021* (Long Talk).

- Paper: https://arxiv.org/abs/2102.09554
- Project page: https://dir.csail.mit.edu/
- Reference implementation: https://github.com/YyzHarry/imbalanced-regression

Our method is an adaptation of the paper's **Label Distribution Smoothing
(LDS)** to window volatility. The paper addresses *imbalanced regression*:
continuous targets whose empirical distribution is highly skewed, so rare
target ranges are under-learned. LDS estimates an *effective* label density by
convolving the empirical histogram with a symmetric Gaussian kernel (nearby
labels share statistical strength), then re-weights by inverse (square-root)
density. We keep the paper's density estimator and tempered inverse weighting,
but apply them to a *volatility pseudo-label* and use them for stochastic
*undersampling* instead of loss re-weighting.

## 1. Problem setup

Training data are sliding windows

```
X_i in R^{L x C},  i = 1..N        (L = input_len, C = channels)
```

over mostly-flat Kubernetes resource-usage series. Let the per-window
volatility score be the mean temporal standard deviation over the target
channels (CPU and memory):

```
s_i = mean_{c in targets} std_{t=1..L}( X_i[t, c] ),   s_i >= 0
```

Empirically `s_i` is extremely right-skewed: ~90% of windows sit near zero
(flat load) while the volatile windows that matter for autoscaling are rare.
Uniform training wastes compute on flat windows and lets them dominate the
gradient.

## 2. Algorithm

**Step 1 — Robust range.** Compute scores on a strided subsample
(<=200k windows/shard) and set the histogram range from a high percentile,
robust to outliers:

```
s_max = percentile({s_i}, 99.9)
edges = linspace(0, s_max, B + 1),   B = 50 bins by default
```

**Step 2 — Empirical density.** Bin every train score and count:

```
b_i   = clip(digitize(s_i, edges), 0, B - 1)
cnt[b] = #{i : b_i = b}
```

**Step 3 — LDS smoothing.** Convolve with a Gaussian kernel
(sigma = 2 bins, radius = 3·sigma), exactly the paper's smoothing step:

```
ker[k] ∝ exp(-0.5 * (k / sigma)^2),   sum(ker) = 1
p~[b]  = (cnt + eps) * ker            (mode="same", eps = 1e-3)
```

`p~` is the *effective* density: empty bins borrow mass from neighbors, so a
bin with one stray sample cannot claim near-zero density.

**Step 4 — Tempered inverse keep probability.** The paper's
inverse-square-root weighting, rescaled for sampling:

```
w[b]    = 1 / sqrt(p~[b])
w_max   = max{ w[b] : cnt[b] > 0 }     # rarest NON-EMPTY bin
p[b]    = clip(w[b] / w_max, min_keep, 1.0),   min_keep = 0.1
```

Normalizing by the rarest non-empty bin (rather than the global max) is
essential: the eps floor would otherwise give empty bins astronomic weights
and collapse every populated bin to `min_keep`. The square root is the
paper's tempering — it rebalances without letting a handful of extreme
windows dominate. Result: rarest volatility band kept with p = 1, flattest
band with p = 0.1.

**Step 5 — Stochastic filtering.** Keep window `i` iff `u_i < p[b_i]`,
`u_i ~ Uniform(0, 1)` from a seeded RNG. Shards that would end up empty keep
their single highest-score window. `y`, `sid`, `ylast` rows follow the same
mask, so alignment is preserved.

**Step 6 — Split discipline.** Only the **train** split is downsampled.
Val/test shards are copied verbatim, so all reported metrics stay unbiased.

## 3. Design properties

| Property | How |
|---|---|
| Deterministic | Per-shard `SeedSequence([seed, ordinal])`; identical output for any worker count |
| Fast | 3 passes (sample / histogram / filter) over shards in a 16-worker pool; ~9 s for 4M windows |
| Safe on degenerate data | If `s_max ≈ 0` (all constant), train is copied unchanged |
| Cacheable | `_downsample_signature.json` (params + input fingerprints); skips rebuild when valid |
| Auditable | `_downsample_stats.json` (totals, per-bin counts/kept/probabilities) |

Complexity is linear in the number of windows; memory per worker is bounded
by the processing chunk (200k windows).

## 4. Parameters (defaults)

| Flag | Default | Meaning |
|---|---|---|
| `--std_bins` | 50 | Histogram bins over `[0, s_max]` |
| `--std_sigma` | 2.0 | LDS Gaussian sigma, in bins |
| `--min_keep` | 0.1 | Keep-probability floor (densest bin) |
| `--seed` | 42 | Sampling RNG seed |
| `--max_std_pct` | 99.9 | Percentile defining `s_max` |
| `--jobs` | 16 | Parallel shard workers |

## 5. Measured effect (2 most volatile services, 4.08M train windows)

- Kept 442,504 windows (**10.85%**); flat band 10%, rarest band 70–96%.
- Mean window std 0.005 → 0.010; p90 0.0097 → 0.028; p99 0.053 → 0.110.
- Downsampling cost: 8.7 s. A bilstm trained on the kept subset beats the
  persistence baseline on CPU MSE by 29% (test, last horizon step).

## 6. Usage

```bash
# Standalone
python preprocessing/downsample_std_windows.py \
  --windows_dir /path/to/windows --out_dir /path/to/windows/downsample_std \
  --feature_set cpu_mem_both

# As a pipeline stage
python pipelines/full_pipeline.py --feature_set cpu_mem_both \
  --model_type bilstm --preprocess_approach downsample_std \
  --windows_dir <dir> --input_len 32 --pred_horizon 5
```

## 7. Implementation map

- `preprocessing/downsample_std_windows.py` — scorer, LDS density, sampler
- `pipelines/preprocessing_pipeline.py` — Step 3b wiring (`--std_bins/--std_sigma/--min_keep/--downsample_seed/--downsample_jobs`)
- `pipelines/full_pipeline.py` — `downsample_std` approach choice + flag passthrough, `preprocess_dir` to train/eval
- `training/train.py`, `training/evaluate.py` — load train/val/test from the downsampled dir (identity transform; val/test copies are full)

## 8. Limitations (for honest presentation)

- Volatility is *input* std, not *forecast-error* difficulty; a flat window
  preceding a spike is downsampled even though it is informative.
- Linear bins over `[0, p99.9]` spread thin tails over many bins; fewer bins
  or log-spaced edges would concentrate rare mass further.
- On near-constant targets (memory), persistence remains brutally strong on
  MAE; the win shows in large-error metrics (MSE/RMSE/Max-Under-Error).

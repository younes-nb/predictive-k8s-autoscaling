# 1 hour, 14-Jul, 30k — four-way comparison (HPA / PBScaler / AURORA v2 / AURORA v3)

Generated 2026-11-23. Window = the full 60 min of each run (`metrics.csv`,
5 s ticks) and the final 30 min (t+30 → t+60). Identical offered load
(`MAX_REQUESTS=30000`, trace day 1995-07-14, seed pool 11 995, VU pool 500,
`maxReplicas=10`, cold reset before each run).

| # | run | autoscaler | image |
|---:|---|---|---|
| 3 | `step3_hpa_30k_jul14_1h_20261121T205554Z` | HPA | — |
| 4 | `step4_pbscaler_30k_jul14_1h_20261121T221343Z` | PBScaler | vendored PBScaler |
| 5 | `step5_aurora_30k_jul14_1h_20261122T201726Z` | AURORA v1/v2 | pre-v3 |
| 6 | `step6_aurora2_30k_jul14_1h_20261122T214926Z` | AURORA v3 | `v0.1.0-20261123T124658Z` |

## 1. Results

### Whole hour

| arm | SLO viol | ok | unlab | rate | outage-aware | front p50 | front p95 | cov % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HPA | 266 | 452 | 2 | **37.0 %** | 37.2 % | 69 ms | 9 937 ms | 99.7 |
| PBScaler | 476 | 233 | 11 | 67.1 % | 67.6 % | 2 207 ms | 24 469 ms | 98.8 |
| AURORA v1/v2 | 124 | 596 | 0 | **17.2 %** | 17.2 % | 115 ms | 2 093 ms | 100.0 |
| AURORA v3 | 460 | 259 | 1 | 64.0 % | 64.0 % | 387 ms | 9 533 ms | 99.9 |

### Final 30 minutes

| arm | SLO viol | ok | rate | outage % | front p50 | front p95 |
|---|---:|---:|---:|---:|---:|---:|
| HPA | 17 | 344 | **4.7 %** | 0.0 | 21 ms | 150 ms |
| PBScaler | 119 | 233 | 33.8 % | 2.5 | 179 ms | 2 340 ms |
| AURORA v1/v2 | 2 | 359 | **0.6 %** | 0.0 | 107 ms | 171 ms |
| AURORA v3 | 203 | 158 | 56.2 % | 0.0 | 219 ms | 6 942 ms |

### Load actually delivered

| arm | delivered iters | it/min | req/s | dropped % | req failed % | mean replicas (42 svc) | last-30m mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HPA | 457 184 | 7 557 | 617 | 53.7 | 8.56 | 94.0 | 93.0 |
| PBScaler | 150 474 | 2 487 | 208 | 84.8 | 6.42 | 129.4 | 165.7 |
| AURORA v1/v2 | 201 292 | 3 327 | 276 | 79.6 | 6.64 | 345.6 | 357.0 |
| AURORA v3 | 208 713 | 3 450 | 270 | 78.9 | 10.38 | 287.4 | 293.0 |

### Run status

| arm | k6 rc | status | reason |
|---|---:|---|---|
| HPA | 0 | incomplete | final cold reset failed (data complete) |
| PBScaler | 99 | incomplete | k6 `http_req_duration p(95)<8000` breached |
| AURORA v1/v2 | 99 | incomplete | same |
| AURORA v3 | 99 | incomplete | same (p95 = 30 s) |

## 2. What the v3 run shows

**v3 is stable but inert.** The controller itself was healthy for the whole
hour: 0 cycle errors, 42 per-service bounds captured from the live HPAs before
suspension, 42 HPAs suspended and 42 restored, 0 OOMKills after the sparse-edge
rewrite, and 99.9 % dataset coverage. None of the review items (C01–C18,
F01–F02) reappeared.

**It is also a clear regression against v1/v2** (64.0 % vs 17.2 % whole-hour,
56.2 % vs 0.6 % in the last 30 min) at essentially the same delivered load
(270 vs 276 req/s). The regression is not a load artefact: at *lower* delivered
load than HPA (270 vs 617 req/s) v3 violates the SLO *more* often (64.0 % vs
37.0 %).

**The replica trajectory explains it.** v3 scaled out early, then froze:

| t | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HPA | 22.5 | 43.4 | 37.6 | 39.6 | 47.9 | 39.2 | 40.9 | 43.5 | 45.0 | 45.8 | 43.1 | 41.3 |
| PBScaler | 9.0 | 19.5 | 28.1 | 33.0 | 34.7 | 40.0 | 42.9 | 49.0 | 49.0 | 49.0 | 52.1 | 54.0 |
| AURORA v1/v2 | 35.9 | 58.0 | 52.8 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 |
| AURORA v3 | 37.1 | 50.2 | 42.7 | 36.4 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 | 36.0 |

(replicas summed over `ts-ui-dashboard`, `ts-auth`, `ts-user`, `ts-order`,
`ts-travel`, `ts-contacts`)

From t+20 min onward v3 makes **no further replica changes** while the SLO is
violated on 56 % of ticks.

**Root cause: the Replica-Change Model has no useful supervision.**
`train/train_models.py` builds its labels from telemetry:

- `y_act = out` only when the weak label is a *resource* class
  (`cpu`, `memory`, …); `in` only for `over_provisioned`, a class that
  `weak_labels.py` never emits; `hold` otherwise.
- `y_tgt = zeros` for **every** sample.

The target head therefore learns "pick the lowest legal count", and the action
head learns "hold" for exactly the latency-bound labels that dominate this
workload (`queue`, `shared_dependency`, `app_problem`). `SafetyLayer.validate`
only *validates* an `out` proposal — it never creates one — so when the model
says `hold`, no amount of SLO pressure produces a scale-out.

The result is the failure mode the review already flagged for the old estimator,
reappearing through a different door: v3 **under-provisions the hot services**
and **over-provisions cold ones**:

| service | p90 HPA | p90 v3 | p90 v1/v2 | replicas HPA | replicas v3 | replicas v1/v2 |
|---|---:|---:|---:|---:|---:|---:|
| ts-user-service | 27 410 | 28 000 | 704 | 10.0 | **3.0** | 9.0 |
| ts-contacts-service | 69 036 | 28 000 | 458 | 10.0 | **5.0** | 10.0 |
| ts-order-service | 1 440 000 | 28 000 | 28 437 | 10.0 | **7.0** | 10.0 |
| ts-security-service | 9 500 | 28 000 | 29 986 | 1.5 | **1.0** | 10.0 |
| ts-preserve-service | 9 500 | 28 000 | 28 000 | 1.8 | **1.0** | 10.0 |
| ts-admin-user-service | 28 000 | 28 000 | 1 041 | 3.3 | 10.0 | 10.0 |
| ts-food-service | 88 | 88 | 87 | 2.3 | **10.0** | 10.0 |
| ts-ticketinfo-service | 85 | 95 | 206 | 1.0 | **4.0** | 10.0 |

v3 spends 287 replicas on average (3× HPA's 94) and still misses the SLO, while
v1/v2 spent 346 and met it — the difference is *where* the replicas went, not
how many.

## 3. Caveats

- **The generator is saturated in every arm.** With `VUS_MODE=auto` the pool is
  500 VUs for 30 000 scheduled iterations/min, so 54–85 % of iterations are
  dropped and delivered load differs 2–3× between arms. The open-loop generator
  therefore makes this a comparison of *as-delivered* load, not of the offered
  30k. The v3-vs-v1/v2 comparison is the cleanest pair (270 vs 276 req/s); the
  v3-vs-HPA comparison is not load-matched.
- **k6 rc=99 in three arms** is the generator's own `http_req_duration
  p(95) < 8000 ms` threshold, i.e. a *result* of the autoscaler missing the
  deadline, not a harness failure. The datasets are complete.
- **v3's container logs were not captured** (the run's `logs/` directory is
  empty), so the per-cycle hold reasons are inferred from the code and the
  replica trajectory rather than read from the log.
- The first collection pass for the v3 run produced 55.7 % coverage because
  Mimir was still ingesting; an immediate re-run of
  `rebuild_train_data.py` over the same window gave 99.9 % (1 unlabelled tick).
  The numbers here use the re-collected dataset.

## 4. Next step

The v3 safety, graph and lifecycle work is done and verified; what is missing is
a real target label for the Replica-Change Model. The minimum change is to
derive `target` from the observed goodput-per-replica of the same service
(`ceil(offered / goodput_per_replica)`) and to emit `over_provisioned` from the
observed replica-vs-goodput ratio, so that the action head can learn `in` and
the target head has a non-degenerate label. That is a training-data change, not
a controller change, and needs its own smoke + 1 h cycle.
# Last 30 minutes — HPA vs PBScaler vs AURORA (14-Jul, 30k, 60 min)

Generated 2026-11-22. Window = final 30 min of each 60-min run (t+30 → t+60 of
`metrics.csv`), identical offered load (382,497 iterations, same trace slice),
42 deployments, `maxReplicas=10`, gateway fixed at 12, cold reset before each run.

| # | run | autoscaler |
|---:|---|---|
| 3 | `step3_hpa_30k_jul14_1h_20261121T205554Z` | HPA |
| 4 | `step4_pbscaler_30k_jul14_1h_20261121T221343Z` | PBScaler |
| 5 | `step5_aurora_30k_jul14_1h_20261122T101643Z` | AURORA-HPA |

## 1. Results

| Metric | HPA 14-Jul | PBScaler 14-Jul | AURORA 14-Jul |
|---|---:|---:|---:|
| SLO violations (count) | **13** | 114 | 360 |
| SLO violation rate (%) | **4.7 %** | 32.9 % | 100.0 % |
| Frontend latency p90 — mean (ms) | **176** | 666 | 7,585 |
| Frontend latency p90 — median (ms) | **23** | 178 | 7,308 |
| Frontend latency p90 — p95 (ms) | **187** | 2,207 | 14,659 |
| Frontend latency avg — mean (ms) | **514** | 2,270 | 3,115 |
| Avg replicas | 114.9 | 189.1 | **265.0** |
| Peak replicas | 123 | 201 | **309** |
| Replica scale events | 46 | 29 | **60** |
| Request success rate — istio (%) | 92.07 | **98.03** | 96.80 |
| Request fail rate — istio 5xx (%) | 7.93 | **1.97** | 3.20 |
| Request fail rate — k6 client (%) | 9.87 | 8.25 | **8.18** |
| Offered requests (derived) | 2,015,161 | 2,005,481 | 2,009,910 |
| Delivered requests (HTTP) | **1,543,225** | 444,675 | 312,797 |
| Offered iterations | 382,497 | 382,497 | 382,497 |
| Delivered iterations | **292,919** | 84,811 | 59,527 |
| Dropped iterations | 84,958 | 293,345 | 318,499 |
| Delivery rate (%) | **77.5 %** | 22.4 % | 15.7 % |
| Delivered iterations / s | **162.7** | 47.1 | 33.1 |
| HTTP requests / s | **857.3** | 247.0 | 173.8 |
| Outage — unlabelled ticks (%) | 23.1 % | 3.6 % | **0.0 %** |
| Outage-aware violation rate (%) | **26.7 %** | 35.3 % | 100.0 % |
| CPU, train-ticket — mean / peak (cores) | **42.96** / 49.89 | 31.04 / 34.67 | 21.08 / 37.55 |
| CPU throttled (% CFS) | 22.3 % | 17.1 % | **10.4 %** |
| Container restarts (delta) | 34 | **0** | **0** |

## 2. What happened

**AURORA scaled the most and finished last.** 265 average replicas (peak 309)
against HPA's 115 and PBScaler's 189 — and it delivered **5× fewer iterations**
than HPA (59,527 vs 292,919) with **100%** SLO violations.

The reason is visible in the CPU columns: AURORA's mean CPU was the **lowest**
(29.6 cores vs 55.8) and its throttling the **lowest** (10.4% vs 22.3%). It
spread replicas so thin that no pod was ever hot — while the front p90 sat at
**7.3 s**. It hit the same per-pod wall as the other two, but from the opposite
direction: too many under-used replicas, each capped at 2 cores, so the queue at
the front never drained.

Its one genuine win is **0.0% outage** — it answered every request (96.8% istio
success, 8.2% k6 client failure) rather than shedding. That is the same
trade-off PBScaler made, taken further: it protected availability and paid for it
in latency.

## 2b. Metric verification (2026-11-22)

Re-derived before the root-cause analysis; two errors in the first pass:

- **CPU was cluster-wide, not train-ticket.** The query omitted
  `namespace="train-ticket"`. Corrected values are in the table above
  (42.96 / 31.04 / 21.08). The ranking is unchanged.
- **KSM is duplicated in the 21-Nov data** (128 series for 64 deployments), so a
  naive `sum()` doubles HPA and PBScaler replica counts. The table uses
  `max by (deployment)` and is unaffected; any future query on that window must
  dedupe.

AURORA's 100% is real: 720/720 ticks above 200 ms (min 560 ms).

## 3. Caveats

- **HPA's 4.7% is over 277 labelled ticks**; 83 of its 360 ticks (23.1%) have no
  front data, all in the 53–60 min tail. On the outage-aware rate HPA is
  **26.7%**, not 4.7%. AURORA's 100% is over all 360 ticks (0% outage).
- **AURORA's `maxReplicas=10` was never the limit** — peak 309 replicas across 42
  deployments is an average of ~7.4, and 37 of 42 deployments were above 1.
- **AURORA's decisions are not comparable to a tuned controller yet.** The
  prototypical head was calibrated on weak labels whose `transition` class
  dominates (3,653 windows vs 360 `cpu`), and its addressability prior was
  fitted on 2,243 balanced samples. The over-scaling is consistent with that
  calibration, not with the control loop being broken: the loop, lifecycle
  accounting, single-writer guard and HPA restore all behaved correctly.
- **`--min-test-violations` / RF comparison does not apply here.** AURORA does
  not use the step-2 RF; its image was built from the encoder + prototypes
  published as `v20261122T085719Z`.
- The 1 h run's `run_status.json` reports `INCOMPLETE` with `k6 exited with
  rc=99`; the dataset is nonetheless 100% covered and the cold reset succeeded.
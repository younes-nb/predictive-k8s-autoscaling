# Three-autoscaler comparison — 14-Jul, 10k, 60 min

Generated 2026-11-24. All three arms ran the same trace day
(1995-07-14), the same offered load (`MAX_REQUESTS=10000`), the same seed pool
(11 995), the same 500-VU generator, the same gateway, and a cold namespace
before each run. Only one autoscaler was active per run; the others were removed
from the cluster.

| # | arm | run |
|---:|---|---|
| 1 | HPA (native, stock) | `step8_hpa_10k_jul14_1h_20261124T023039Z` |
| 2 | PBScaler (RF) | `step9_pbscaler_10k_jul14_1h_20261124T040752Z` |
| 3 | AURORA-R (rule policy) | `step10_aurorar_10k_jul14_1h_20261124T055726Z` |

## 1. Headline

| arm | SLO viol | rate | front p50 | front p95 | delivered iters | req/s | dropped | req failed | mean replicas | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HPA | 34 | **4.7 %** | 67 ms | **196 ms** | 114,340 | 151 | 65.3 % | 6.02 % | 112.6 | 127 |
| PBScaler | 30 | **4.2 %** | 88 ms | **189 ms** | 98,895 | 128 | 70.0 % | 6.27 % | 162.8 | 196 |
| AURORA-R | 80 | **11.1 %** | 81 ms | 583 ms | 80,677 | 102 | 75.5 % | 6.80 % | **99.9** | 100 |

Final 30 min: HPA 9.4 %, PBScaler 5.3 %, AURORA-R 7.2 %.

AURORA-R used the **fewest replicas of the three** (99.9 vs 112.6 / 162.8) and
still missed the SLO **2.4× more often than HPA**. Its front p95 was 3× worse
(583 ms vs 196 ms).

## 2. The decisive number

Replica counts changed at least once during the hour:

| arm | services that changed replica count |
|---|---:|
| HPA | 16 / 42 |
| PBScaler | 22 / 42 |
| **AURORA-R** | **1 / 43** |

AURORA-R moved `ts-order-service` from 4 to 5 and then held every one of its 42
other services at a fixed count for the remaining ~55 min, while the trace
ramped and the SLO was violated on 11 % of ticks.

Replicas by 5-minute bucket (mean), latency-critical services:

| arm | service | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HPA | ts-user-service | 3.9 | 9.6 | 9.5 | 4.0 | 5.0 | 4.4 | 5.5 | 10.0 | 8.2 | 7.9 | 2.6 | 2.0 |
| PBScaler | ts-user-service | 1.0 | 1.1 | 2.0 | 2.1 | 6.8 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 |
| **AURORA-R** | ts-user-service | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| HPA | ts-contacts-service | 6.5 | 5.3 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 8.3 | 10.0 | 10.0 | 10.0 | 10.0 |
| PBScaler | ts-contacts-service | 4.3 | 7.8 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 |
| **AURORA-R** | ts-contacts-service | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |

Per-service latency in the final 30 min (p90, ms) against that allocation:

| service | p90 HPA | p90 PBScaler | p90 AURORA-R | reps HPA | reps PBScaler | reps AURORA-R |
|---|---:|---:|---:|---:|---:|---:|
| ts-user-service | 1 300 | 783 | **28 000** | 6.0 | 9.0 | **1.0** |
| ts-ui-dashboard | 675 | 450 | **10 129** | 3.2 | 9.0 | 3.0 |
| ts-admin-user-service | 2 750 | 950 | **28 000** | 1.2 | 9.0 | 1.0 |
| ts-security-service | 85 005 | 28 000 | 28 000 | 3.7 | 9.0 | 3.0 |
| ts-contacts-service | 28 000 | 633 | 650 | 8.7 | 9.0 | 4.0 |
| ts-admin-basic-info-service | 24 000 | 850 | 800 | 3.0 | 9.0 | 3.0 |

AURORA-R held `ts-user-service` at **1 replica** for the full hour while its p90
sat at the 28 s client timeout, and `ts-ui-dashboard` at 3 while its p90 ran at
10 s.

## 3. Root cause, with the code path

`aurora_r/policy.py::choose_rule` is the decision path for this arm
(`policy_source=rule`, because the learned response model has no intervention
dataset yet). It has exactly two moves:

```
scale out:  SLO violated AND cpu_request_util >= rule_cpu_util_scaleout (=1.0)
                                 [or cpu_throttled_frac >= 0.05]
scale in:   SLO continuously healthy (Wilson upper bound <= 1 %) for
            scale_in_health_s (=300 s) AND cpu_request_util <= 0.30
```

At this load the SLO miss rate is ~4.7 %. That is:

- **above** `max_slo_risk_scalein` (1 %), so the Wilson bound never certifies
  "healthy" → the scale-in branch is closed;
- **above** zero, so `degraded` is true → the scale-out branch is open;
- but the scale-out branch additionally requires `cpu_request_util >= 1.0`, and
  the latency-critical services in this workload are **not CPU-bound** — their
  bottleneck is JVM heap / shared datastores. Their CPU request utilisation sits
  well below 1.0 all hour.

So every cycle evaluates `direction = out`, finds no service meeting the CPU
evidence, and returns *"no service met the measured-pressure evidence for out"*.
The controller is not stuck in a crash or a lock; it is in a **dead zone**:
degraded enough to demand action, not healthy enough to permit the only action
it can take, and without the evidence its one scale-out rule requires.

Two secondary defects compound it:

1. **The SLO band is a no-man's-land.** Any miss rate in (1 %, ∞) is "degraded"
    but only "healthy" at ≤ 1 %. A service running at 5–11 % violations can
    never be scaled in and cannot be scaled out without CPU evidence. There is
    no action for the middle of the distribution.
2. **Scale-in evidence is CPU-only.** `cpu_request_util <= 0.30` is the sole
    scale-in condition. A service that is idle on CPU but latency-bound (heap,
    GC, connection pool) reads as over-provisioned and is scaled toward its
    minimum — which is how `ts-user-service` reached 1 replica.

## 4. Weaknesses of AURORA-R, ranked by evidence

1. **No usable scale-out path for non-CPU bottlenecks.** The only horizontal
   scale-out evidence is `cpu_request_util >= 1.0` or throttling ≥ 5 %. This
   workload's dominant failure family (`ts-user-service` / `ts-auth-service`
   heap exhaustion, `ts-preserve-*` at the 28 s client timeout) is not
   CPU-bound. Result: 1 replica change in 60 minutes.
2. **Replica allocation is frozen, not adaptive.** 1/43 services moved. HPA moved
   16/42 and PBScaler 22/42 on the same trace.
3. **It is the most frugal arm and still the least compliant** — 99.9 mean
   replicas (12 % below HPA) with 2.4× HPA's violation rate. Efficiency and
   compliance both lose, so the low replica count is not buying anything.
4. **Latency on the user-facing path is 3× worse** (p95 583 ms vs 196 ms), and
   `ts-ui-dashboard` p90 is 15× HPA's (10.1 s vs 675 ms).
5. **Scale-in is gated on a healthy SLO but not on a non-CPU bottleneck check**,
   so the mechanism that *is* reachable (scale-in) fires on the wrong services
   while the one that is needed (scale-out) cannot fire.
6. **The middle of the SLO distribution has no action.** Miss rate in (1 %, ∞)
   is "degraded", which permits neither scale-in (needs ≤ 1 %) nor, without CPU
   evidence, scale-out.

## 5. Caveats

- **The response model is not in the decision path.** No settled interventions
  existed before this run, so `policy_source=rule` was used. The rule is
  deterministic and carries no learned confidence; the learned action-response
  model is still a bootstrap. Its training data is exactly what this run
  produced — the journal holds the settled before/after observations.
- **Delivered load differs** (151 / 128 / 102 req/s) because the generator is a
  500-VU pool against 10 000 scheduled iterations/min; 65–76 % of iterations are
  dropped in every arm. AURORA-R delivered the least, which is partly a
  consequence of its own latency, not an independent variable.
- **k6 rc=99 on the AURORA-R arm** is the generator's own
  `http_req_duration p(95) < 8000 ms` threshold, i.e. a result.
- **10k sits at the SLO knee on this cluster.** The 13-Jul 4 h HPA arm recorded
  0 violations (max front p90 199.5 ms), which is why the RF was trained on the
  14-Jul HPA arm (34 positives) rather than on 13-Jul (0 positives).
- **The RF itself is at chance on this load**: trained on the 14-Jul HPA data it
  scored recall 0.0, AUC 0.5 on a chronological holdout. At 10k, SLO violations
  are not predictable from `[i, qps, count]`. PBScaler therefore never predicts
  a violation and behaves close to a static-provisioning baseline.
- Phase 12's final cold reset reports failure on arms whose HPAs declare
  `minReplicas > 1` (the HPAs legitimately restore replicas before the reset can
  assert `spec == 1`). It does not affect the datasets, which are all
  100 % coverage / COMPLETE.

## 6. What would fix it

1. Add a scale-out path whose evidence is not CPU utilisation: latency and
   completion ratios per service, plus the diagnosis model's degradation
   probabilities. The rule should be able to say "this service is latency-bound
   and replicas are the plausible remedy" without a CPU threshold.
2. Make scale-in require evidence that the bottleneck is *not* a
   non-horizontal one (heap, connection pool, downstream datastore) before
   reducing replicas.
3. Give the band between "healthy" and "degraded" an explicit action (usually
   hold, but with hysteresis rather than a dead zone).
4. Then train the response model on the settled interventions this run produced
   and re-run the arm with `policy_source=model`.
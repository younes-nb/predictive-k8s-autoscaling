# Same-Day A/B: PBScaler v1.2.5 vs Native HPA — Root-Cause Analysis

**Date:** 2026-10-26 (Tehran) · **Data cutoff:** 06:41Z = 10:11 Tehran (Run B in progress)
**Method:** automated measurement (`pbscaler/scripts/analyze_ab4h.py`) + live Mimir queries
**Status:** Run A complete (240 min). Run B at 112 of 240 min — all Run-B figures are
provisional and will be restated at completion (~12:18 Tehran).

---

## 1. Question and verdict

**Question:** is the observed PBScaler-vs-HPA difference an *implementation* issue, a
*cluster-status* issue, or is PBScaler *actually* underperforming HPA?

**Verdict: PBScaler is genuinely underperforming in this regime.** Implementation and
cluster status are both excluded below with direct measurements. The mechanism is
specific and reproducible: **the latency that drives the SLO label in this workload is
not replica-addressable, and PBScaler has no feedback loop to discover that fact.** It
therefore spends capacity monotonically on services where more replicas provably do not
help, while its redundancy module never fires to reclaim it.

| Hypothesis | Verdict | Strongest evidence |
|---|---|---|
| Implementation bug (v1.2.5 instrumentation / model / switch) | **Refuted** | `util/GA.py` byte-identical v1.2.4↔v1.2.5; `PBScaler.py` differs only by a timing helper; model md5 identical; 0 clamps, 0 oscillation |
| Cluster status (starvation, disk, telemetry, node health) | **Refuted** | node CPU 10–25%, mem 31–39%, 0 evictions, 0 pending, disk 50–58%, Mimir serving 1126 live series, controller 0 restarts |
| PBScaler genuinely worse | **Supported** | travel p90 = **28,000 ms @ 4 replicas (HPA)** and **28,000 ms @ 9 replicas (PBScaler)** — identical; PBScaler holds +12 % pods for −28 % delivered work |

---

## 2. Experiment configuration

Both arms, byte-identical offered load:

| parameter | value |
|---|---|
| `START_MIN` | 1932 (Jul-14 phase, same as Run 3 / Run 2′) |
| `TRACE_COMPRESS` | 8 |
| `MAX_REQUESTS` | 7600 |
| `USER_POOL` | 560 VUs (closed loop) |
| `TEST_MIN` | 255 = 15 min warmup + 240 min measured |
| collector | 5 s interval, istio front-P90 label, completion semantics (`rate[1m]`) |
| SLO | 200 ms (gateway → `ts-ui-dashboard` edge P90) |

**Offered profile** (recomputed from `http_mcr_NASA_jul95.csv`, grid `(1932 + m·8) mod 2880`):
3,474,304 requests over 255 min · req/s min 0, median 190, mean 227, max 836 ·
per-window min–max normalization puts the window peak at 7600 (raw max 836 req/s).

| | Run A | Run B |
|---|---|---|
| autoscaler | 43 native HPAs, **0** pbscaler pods | **0** HPAs, 1 pbscaler pod (`2/2 Running`, 0 restarts) |
| image | — | `docker.io/younesnb/pbscaler:v1.2.5` |
| k6 window | 20:45:26Z → 01:00:26Z (00:15→04:30 Tehran) | 04:33:21Z → 08:48:21Z (08:03→12:18 Tehran) |
| collector window | 21:00:27Z → 01:00:22Z | 04:48:21Z → 08:48Z |
| dataset | `pbscaler/data/runA_hpa_jul14_4h_20261025_204525` | `pbscaler/data/runB_pbscaler_jul14_4h_20261026_043321` |
| rows | 2880 (239.9 min, **exact 5.0 s cadence, 0 gaps**) | 1342 at cutoff (112 min) |
| unlabeled | 8 (0.28 %) | 0 |

Model in both arms: `RandomForestClassify.model`, md5 `0ff9f8c5ad1725f85ef9601cfec48714`,
43 services × `[i, qps, count]` = 129 features, paper-default class weight.

---

## 3. Headline result — aligned minute-by-minute

Identical trace minutes (first 1323 collector ticks = 110 min of both runs):

| metric | Run A (HPA) | Run B (PBScaler) | delta |
|---|---|---|---|
| violations / labeled | **309 / 1323** | **464 / 1323** | +155 |
| violation rate | **23.36 %** | **35.07 %** | **+11.71 pts (+50 % rel.)** |
| pods (median / max) | 115 / 116 | 161 / 168 | **+46 pods (+40 %)** |
| front P90 median | 10.0 ms | 30.4 ms | 3.0× |
| front P90 p95 | 15,857 ms | 16,438 ms | ≈ equal |
| front **AVG** median | 464.4 ms | 1196.2 ms | 2.6× |
| front **AVG** p95 | 3,951 ms | 4,218 ms | ≈ equal |

Per-10-minute blocks (A rate / B rate):

```
min   0: A 20.8% 115p | B 40.8% 129p      min  60: A 23.3% 115p | B 28.3% 162p
min  10: A 24.2% 114p | B 40.0% 135p      min  70: A 34.2% 115p | B 30.0% 163p
min  20: A 35.8% 116p | B 37.5% 143p      min  80: A 18.3% 115p | B 39.2% 164p
min  30: A 25.8% 116p | B 28.3% 153p      min  90: A 23.3% 115p | B 29.2% 166p
min  40: A 21.7% 116p | B 36.7% 160p      min 100: A 15.8% 115p | B 22.5% 167p
min  50: A 12.5% 115p | B 54.2% 161p      min 110: A 66.7% 115p | B  0.0% 168p (partial)
```

PBScaler wins 1 of 11 complete blocks (min 70) and loses 10. Its pod count rises
monotonically (129 → 168) and never falls.

**Client-facing impact, same 125 min of k6 (15 warmup + 110 measured):**

| | A (HPA) | B (PBScaler) |
|---|---|---|
| completed iterations | **791,667** | **649,952** (−17.9 %) |
| throughput | 105.6 it/s | 86.7 it/s |
| client request timeouts | 36,496 | 48,166 (**+32.0 %**) |
| timeout rate per attempt | 4.41 % | **6.90 %** (1.56×) |
| interrupted iterations | 0 | 0 |

Run A full 255 min: 2,104,680 completed, 74 interrupted, 77,883 timeouts (3.6 % of attempts).

---

## 4. Hypothesis 1 — implementation bug: **refuted**

| Check | Method | Result |
|---|---|---|
| Decision-logic drift v1.2.4→v1.2.5 | extract `/app/util/GA.py` from both images, `diff -q` | **byte-identical** |
| Controller drift | `diff` of `/app/PBScaler.py` v1.2.4 vs v1.2.5, timing lines filtered | only the `_log_timing` helper (5 lines) + 6 call sites |
| Model identity | `md5sum` in-pod vs on-disk | `0ff9f8c5ad1725f85ef9601cfec48714` — same RF that produced Run 2′ |
| Predictor feature set | in-pod `RandomForestClassify.svcs` | 1007 bytes, 43 services, banner `loaded predictor svc list … (43 svcs)` |
| Runaway recurrence (v1.2.0 class bug) | grep `clamped out-of-band action` | **0 occurrences** |
| Oscillation / flapping | direction flips over each service's write sequence | **0 flips**; ramps strictly monotonic (e.g. `[2,3,4,5,6,7,8,9]`) |
| Ghost / self-resurrecting deployments | pod count, restart count | 1 pod, 0 restarts, 0 HPAs present |
| Collector correctness | cadence + label sanity | exact 5.0 s, 0 gaps, 0.28 % unlabeled |

The only behavioral change in v1.2.5 is that it *prints timestamps*. The v1.2.1 safety
fixes (`get_all_counts()` for `svc_counts`, bound arrays built over the final root list,
post-GA clamp) are intact and demonstrably working.

**The predictor is also not at fault** — scored offline with the live model against each
arm's own recorded feature rows:

| scored on | n | true viol | AUC | acc | P(safe) median *when actually violating* | miss rate |
|---|---|---|---|---|---|---|
| training set (Run 1′, in-sample) | 4303 | 25.4 % | 0.999 | 0.988 | 0.050 | 2.8 % |
| **Run A states (HPA)** | 1156 | 23.6 % | **0.987** | 0.951 | 0.160 | 14.3 % |
| **Run B states (PBScaler)** | 1156 | 37.1 % | **0.979** | 0.912 | 0.160 | 12.1 % |

AUC 0.979 on PBScaler's own states means the model is **not blind** to the bad states
PBScaler creates — it correctly assigns P(safe) ≈ 0.16 to genuinely violating ticks.
The failure is downstream of perception: it is in the *action* the optimizer chooses.

---

## 5. Hypothesis 2 — cluster status: **refuted**

Measured during Run B (same queries valid for Run A):

| signal | value |
|---|---|
| node CPU utilization (6 nodes) | 10 % – 25 % (avg 20.2 %) |
| node memory utilization | 31 % – 39 % |
| node status | all `Ready`, uptime 6 d 22 h, **no reboots** |
| evicted / failed / unknown pods | **0** |
| pending pods (all namespaces) | **0** |
| master3 disk | 50–58 % (26–30 G free), no `DiskPressure` taint |
| mimir blocks / tsdb | 1.7 G / 15 G after launch prune |
| Mimir API | live, 1126 `istio_requests_total` series, port-forward + watchdog healthy |
| k6 interrupted iterations | 0 |
| controller restarts | 0 |

The cluster had **substantial spare capacity** throughout. This is not a
resource-starvation artifact — it is the opposite: PBScaler consumed *more* of the spare
capacity and produced *worse* results (§6.3).

**One acknowledged context difference:** the orchestrator's readiness-check bug (§9)
left the controller idle 3.5 h before Run B, during which waste detection pulled the
fleet to floor (113 pods). Run A began from an HPA-managed state (152 spec at warmup,
114 pods at collector tick 0); Run B began at 113 spec / 129 pods at collector tick 0.
Both then ran the same 15-min warmup. The divergence measured is steady-state, not a
startup transient: by min 30–40 PBScaler is already at 153–160 pods and still violating
more, and its pod count keeps climbing through min 110.

---

## 6. Hypothesis 3 — genuine underperformance: the mechanism

### 6.1 The dominant latency is not replica-addressable

The SLO label is driven by a small set of pathological edges whose latency is
**invariant to a 2.25× capacity increase** (medians over each run's window, `metrics.csv`):

| service | Run A (HPA) | Run B (PBScaler) | replica change → effect |
|---|---|---|---|
| `ts-travel-service` | p90 **28,000 ms** @ **4** reps | p90 **28,000 ms** @ **9** reps | +125 % → **0 % improvement** |
| `ts-food-service` | p90 26,490 ms @ 3 | p90 20,832 ms @ 9 | +200 % → −21 % |
| `ts-preserve-service` | p90 **2,462 ms** @ 3 | p90 **21,587 ms** @ 9 | +200 % → **8.8× WORSE** |

28,000 ms is not a queueing delay — it is the **k6 30 s client timeout**
(`load_testing/k6/trainticket.js:58,75,78` all set `timeout: '30s'`). These are hung
requests, so no number of additional replicas shortens them.

Replica count is decoupled from latency on these edges. `ts-food-service` at a
**constant 9 replicas** across consecutive 10-minute blocks:

```
p90: 26828  27334  27942  27091  26965  14251    145    156     81  (ms)
     └──────── 50 min pinned at ~27 s with 9 replicas ───────┘ └─ recovers on its own ─┘
```

`ts-preserve-service`, also at a constant 9 replicas:

```
p90: 24505  25069  26920    191    550  26999  25223  16843  27100  (ms)
                └ 191 ┘          └──── re-hangs at 27 s ────
```

And where PBScaler *did* ramp and the metric moved, it moved the wrong way:
`ts-station-service` 3 → 9 replicas, p90 5 → 34 → 9 ms, corr(replicas, p90) = **+0.40**.

### 6.2 Capacity misallocation (aligned 1342 ticks)

| raised by PBScaler | A → B | | lowered by PBScaler | A → B |
|---|---|---|---|---|
| `ts-admin-basic-info-service` | 3 → 9 | | `ts-ui-dashboard` | 10 → 9 |
| `ts-auth-service` | 3 → 9 | | `ts-order-service` | 4 → 3 |
| `ts-food-service` | 3 → 9 | | `ts-contacts-service` | 4 → 3 |
| `ts-preserve-service` | 3 → 9 | | `ts-ticket-office-service` | 3 → 2 |
| `ts-station-service` | 4 → 9 | | `ts-seat-service` | 3 → 2 |
| `ts-ticketinfo-service` | 4 → 9 | | `ts-route-service` | 3 → 2 |
| `ts-travel-service` | 4 → 9 | | `ts-order-other-service` | 3 → 2 |
| `ts-travel2-service` | 4 → 9 | | `ts-assurance-service` | 3 → 2 |
| `ts-basic-service` | 3 → 8 | | `ts-config-service` | 3 → 2 |
| `ts-payment-service` | 4 → 5 | | `ts-admin-order-service` | 3 → 2 |
| `ts-admin-travel-service` | 4 → 5 | | | |
| `ts-admin-route-service` | 1 → 3 | | | |
| `ts-rebook-service` | 1 → 3 | | | |
| `ts-verification-code-service` | 1 → 2 | | | |

Fleet totals: **A median 115 pods → B median 161 pods** · services at ≥9 replicas:
**2 (A) → 10 (B)** · services at ≤2 replicas: **18 (A) → 23 (B)**.

### 6.3 Efficiency inversion — more pods, less work

Per-service CPU *used* vs CPU *limit* (250 m per pod), averaged over each window
(cAdvisor + kube-state-metrics via Mimir):

| | Run A (HPA) | Run B (PBScaler) |
|---|---|---|
| pods | 165 | **185** (+12 %) |
| total CPU actually used | **17.6 cores** | 12.7 cores (**−28 %**) |
| delivered istio requests/s | **1473** | 939 (**−36 %**) |

HPA's top-of-ranking allocation:

| service | pods | cores used | limit | util |
|---|---|---|---|---|
| `ts-ui-dashboard` | 10 | 2.61 | 2.50 | **1.04** |
| `ts-travel-service` | 4 | 1.03 | 1.00 | **1.03** |
| `ts-food-service` | 3 | 0.76 | 0.75 | **1.01** |
| `ts-preserve-service` | 4 | 0.78 | 1.00 | 0.78 |
| `ts-travel2-service` | 7 | 1.01 | 1.75 | 0.58 |

PBScaler's:

| service | pods | cores used | limit | util |
|---|---|---|---|---|
| `ts-travel-service` | 9 | 2.32 | 2.25 | **1.03** |
| `ts-ui-dashboard` | 9 | 1.68 | 2.25 | 0.75 |
| `ts-food-service` | 9 | 1.67 | 2.25 | 0.74 |
| `ts-station-service` | 9 | 0.48 | 2.25 | 0.21 |
| `ts-admin-basic-info-service` | 9 | 0.47 | 2.25 | 0.21 |
| `ts-basic-service` | 8 | 0.37 | 2.00 | 0.18 |
| `ts-travel2-service` | 9 | 0.28 | 2.25 | 0.13 |
| `ts-preserve-service` | 9 | 0.24 | 2.25 | 0.11 |

PBScaler gave the bottleneck **2.25× more CPU** (travel 1.00 → 2.25 cores) and travel
remained saturated at util 1.03 — while five services sit at 8–9 replicas doing
0.11–0.21 of their limit. **~60 pods of allocation are inert.**

The collateral damage lands exactly on the measured label — the front edge (5-min-averaged
istio P90 by destination workload):

| | HPA | PBScaler |
|---|---|---|
| `ts-ui-dashboard` p90 | **33.8 ms** | **749.3 ms** (22×) |
| `ts-ui-dashboard` qps | 796.5 | 485.3 |

Mechanism (inference supported by the above): under a **closed-loop** 560-VU replay,
flows that block on the hung travel/preserve paths occupy front-tier worker threads, so
the *mix* of front requests shifts toward slow composite flows. The front P90 rises even
though the front service has spare CPU (util 0.75). PBScaler's response — ramping the
fan-in tier to the cap — neither removes the hang nor restores the mix.

### 6.4 The redundancy module is dead code under load, and every write is echoed

| controller behavior | measured |
|---|---|
| anomaly cycles (15 s target) | 348 over 112 min → **effective period 19.3 s** |
| cycles with ≥1 violating edge | **345 / 348 (99.1 %)** |
| `begin scale in` events | **0** |
| `redundancy_check` timed executions | **0** |
| `begin scale out` events | 56 |
| spec writes | 2464 over 56 commits = **44 writes per commit** |
| value-changing writes | 59 of 1848 sampled = **3.2 %** |
| no-op echo writes | **96.8 %** |
| "roots outside feature set" / "no actionable roots" / "all at cap" | 0 / 0 / 0 |

Two structural defects, both quantified:

1. **`waste_detection()` early-returns whenever any edge is abnormal**
   (`if len(ab_calls) > 0: return`). In a saturated regime 99.1 % of cycles have an
   abnormal edge, so the redundancy check *never executes* — PBScaler becomes
   scale-out-only and cannot reclaim the 46 surplus pods it accumulates.
2. **`execute_task()` re-patches the entire action map every cycle** — all 43 services
   plus the gateway, whether or not the value changed. 96.8 % of writes are no-ops:
   ~2400 pointless API-server round-trips per 2 h, and the same mechanism whose
   stale-read race produced the v1.2.0 runaway.

### 6.5 Measured module time costs (paper Table III) — real recordings

Source: `[timing] <UTC ISO> module=<name> seconds=<t>` lines written by
`time.perf_counter()` hooks inside the in-cluster v1.2.5 container, streamed to
`/tmp/opencode/runB_pbscaler_controller_raw.log`, filtered to the collector window.
**These are measured, not transcribed from the paper.**

| module (user naming) | internal key | n | mean s | med s | p90 s | p95 s | max s | Σ s |
|---|---|---|---|---|---|---|---|---|
| SLO Violation Detection | `slo_detection` | 410 | 0.402 | 0.347 | 0.657 | 0.841 | 1.704 | 164.8 |
| Redundancy Checking | `redundancy_check` | **0** | — | — | — | — | — | 0.0 |
| Performance Bottleneck Analysis | `bottleneck_analysis` | 346 | **3.675** | 3.648 | 4.320 | 4.472 | 5.454 | **1271.5** |
| Decision Maker (GA) | `decision_maker_ga` | 57 | **10.703** | 10.450 | 12.098 | 12.798 | 13.662 | 610.1 |
| Action Commit (k8s write, aux) | `action_commit` | 57 | 0.419 | 0.420 | 0.466 | 0.476 | 0.562 | 23.9 |

Share of the 2070 s of measured module time: **bottleneck analysis 61.4 %, GA 29.5 %,
SLO detection 8.0 %, commit 1.2 %.**

Consequences that follow directly from these numbers:

* **A full decide-and-act cycle costs 15.2 s** (0.40 + 3.68 + 10.70 + 0.42) against a
  nominal `AB_CHECK_INTERVAL` of 15 s, so the loop achieves only ~19.3 s per cycle
  (observed) — **PBScaler's real control period is ~29 % slower than its design point**,
  and every scale-up needs ≥3 cycles (≈60 s) to reach 3 replicas.
* **The GA is ~3× more expensive than previously recorded** (10.7 s measured vs the
  ~3.58 s figure carried in earlier notes/`AGENTS.md`, which was not timer-derived). At
  `size_pop=50, max_iter=5` it evaluates ~300 predictor inferences per invocation; it is
  the largest *per-action* cost, while PageRank is the largest *aggregate* cost because
  it runs every cycle.
* Bottleneck analysis (PageRank over the abnormal subgraph) is the dominant aggregate
  consumer — 1271 s per 112 min — and it is precisely the module whose output is wrong
  (§6.1).

### 6.6 Victim amplification — the `ts-preserve-service` / `ts-food-service` case study

Why did preserve show p90 **2,462 ms at 3 replicas under HPA** but **21,587 ms at 9
replicas under PBScaler**? Because preserve was never the bottleneck — it is a **relay**
whose inbound p90 mixes two populations (fast security/contacts hops at 6–40 ms and a
hung travel hop at 78–169 s), and PBScaler moved the mixture by *opening the gate* on
the relay. Edges out of preserve (istio destination-side p90, 5-min means):

| preserve → | Run A (HPA) | Run B (PBScaler) |
|---|---|---|
| `ts-travel-service` | 77,963 ms @ 0.35 calls/s (UC flags 0.32/s) | 169,497 ms @ 0.96 calls/s (UC 0.89/s) |
| `ts-security-service` | 40 ms @ 2.84/s | 31 ms @ 4.37/s |
| `ts-contacts-service` | 7 ms @ 2.79/s | 6 ms @ 4.37/s |
| preserve inbound | 3.57 req/s → p90 2,462 ms | 4.35 req/s → p90 21,587 ms |
| preserve CPU util (of limit) | — | **0.11** (9 pods, 0.24 of 2.25 cores) |

Under HPA only 0.35 hung calls/s reached travel, so preserve's 90th percentile stayed in
the fast population (2.46 s). Under PBScaler the hung-hop rate rose ×2.7 while preserve's
own request rate rose only ×1.22 → the p90 crossed into the hung population (21.6 s).
The amplifier is `ts-food-service`:

| food → | Run A | Run B |
|---|---|---|
| food inbound / CPU util | 5.25 req/s, **util 1.01 @ 3 pods** | 20.40 req/s, util 0.74 @ 9 pods |
| food → travel | 0.05 calls/s @ 28,224 ms | **1.49 calls/s @ 141,697 ms** (×30 injection) |
| food → food-map | 0.15/s @ 12 ms | 6.01/s @ 9 ms |

Under HPA the victims were themselves mildly CPU-starved, which accidentally **gated**
how much load they injected into travel. PBScaler relieved the victims (food 3→9,
preserve 3→9) and the gate opened: food's throughput ×3.9, its calls into the
already-hung travel ×30. Travel's own saturation is unchanged in kind (per-pod CPU util
1.03 in both arms; inbound qps 7.22→12.00, +66 %; failed-flag qps 2.86→5.07/s, +77 %;
inbound p90 28,000 ms in both). Its dependencies exonerate it as a victim-of-victims:
ticketinfo 13,250→504 ms, route ~8 ms, mongo util 0.02–0.03, mysql 0.01. Client-side
ranking agrees travel is the epicenter in both arms (k6 timeout endpoints:
travelservice 53,207 A / 33,166 B; foodservice 14,644 / 15,671; preserveservice 8,197 /
5,115 — B's log is shorter, ordering identical).

The loop:

```
travel per-pod CPU-throttled → hangs to ~30 s (k6 client timeout)
   ↑ injected load +66 %              ↓ composite latency inherited by
victims un-gated (food 3→9,      preserve 2.5 s→21.6 s, food, ui 34→749 ms
 preserve 3→9)                       ↓
PageRank sees preserve/food/ui abnormal → ramps them further
   ↓
scale-in never fires (0 in 121 min; 99.1 % of cycles have an abnormal edge)
   ↓
pods 115→168, delivered work −36 %, violation rate 23 %→34 %
```

So preserve was abnormal *because* the real bottleneck was misdiagnosed as
replica-addressable — and the diagnosis was useless anyway because the GA's only lever
cannot fix a per-pod CPU wall (`max_pod 10 × 250 m = 2.5 cores`/service). The novel
piece: **scaling a relay increases the rate at which it drowns the true bottleneck** — a
negative externality invisible to any per-service optimizer.

### 6.7 Why PBA could not rank `ts-travel-service` as *the* bottleneck: bug ledger vs missing oracles

Short answer: **the detector did see travel — three failure layers prevented it from
acting on that knowledge correctly.** (a) the ranking statistic degenerates under
saturation and tie-breaks alphabetically; (b) the root filter permanently hides any node
that reaches 9 replicas, so travel became invisible exactly when the graph evidence for
it matured; (c) the GA's fitness oracle (the RF) is provably **flat** in travel's replica
dimension because the training set never showed travel above 5 replicas — so the
"optimizer" adds nothing over the clamp's forced `cur+1`.

**(a) What the anomaly detector saw** (`get_call_latency`, irate[1m], threshold
200 ms × 1.1 = 220 ms — the controller's own query, evaluated live against Mimir):

| instant | abnormal edges (source → destination : p90) |
|---|---|
| 05:20Z | ui→preserve 28,000 · ui→travel 28,000 · gw→ui 15,848 · ui→auth **235** |
| 06:00Z | **preserve→travel 174,000** · **food→travel 102,857** · ui→preserve 26,001 · ui→travel 25,510 · ui→food 20,205 · gw→ui 1,308 · ui→auth 235 |
| 06:40Z | ui→preserve 28,000 · ui→travel 28,000 · ui→food 26,000 · ui→auth 235 |

Travel is in every set. The 174 s and 103 s downstream hops prove the mesh *measured*
travel as the slow endpoint. Localization input was never the problem.

**(b) Ranking collapse — flat personalization on a star graph, alphabetical ties.**
`cal_topology_potential` scores each node as `ef(node) + 0.37·Σ ef(predecessors)` where
`ef` = count of 60-s-window samples whose **aggregate inbound** p90 (summed over all
sources) exceeds 200 ms. In saturation every hot node's aggregate inbound is 20–170 s ≫
200 ms → `ef` is maximal for all of them → the personalization vector is **flat**. With
flat personalization PageRank reduces to graph structure, and during the ramp window
(04:34–04:43Z) the structure was a star `ui→{auth, preserve, travel, food, …}` whose
leaves are structurally identical sinks → exact ties → the stable `sorted(reverse=True)`
over the alphabetically-sorted `ab_mss` resolves them **alphabetically**. The revealed
preference (value-changing writes, 88 total):

| reached 9 replicas | at | note |
|---|---|---|
| `ts-auth-service` | 04:37:56Z | 1.5 QPS, p90 235 ms — the weakest signal in the set |
| `ts-travel2-service` | 04:39:28Z | |
| `ts-ui-dashboard` | 04:40:31Z | |
| `ts-preserve-service` | 04:42:04Z | relay (util 0.11) |
| `ts-travel-service` | 04:43:30Z | **the actual bottleneck — 5th** |
| `ts-food-service` | 04:46:25Z | relay (util 0.74) |

auth — a 1.5 QPS service at 235 ms, barely over threshold — was promoted to the cap
before travel, purely from the tie-break. (The `ef` definition also *rewards*
inheritance: a relay's score includes its children's hang because the series is
aggregate-inbound.)

**(c) Cap blindness — the filter hides solved nodes forever.** Root selection is
`filter(root: svc_counts[root] + 1 < max_num)` with `max_num = 10`: at 9 replicas the
condition is false, so a capped node is **removed from candidacy permanently** (nothing
scales in — §6.4). Travel's write sequence ends at `04:43:30Z 8→9`; zero further travel
writes in the remaining ~2 h despite 345/348 cycles still finding abnormal edges. The
cruel sequencing: the DAG evidence that would have ranked travel first
(`preserve→travel` 174 s, `food→travel` 103 s) only appears at ~06:00Z — **after** travel
was already invisible. PBA's strongest signal arrived for a node it could no longer pick.

**(d) The missing oracle — the RF is flat in the dimension the GA searches.** Sweeping
travel's replica count 1→10 against the live model, holding all other services at the
recorded state:

| tick | P(violation) at travel = 1..10 | sensitivity |
|---|---|---|
| Run B 05:57:07Z (recorded viol) | 0.60 ×10 | **+0.000** |
| Run A 23:00:02Z (recorded viol) | 0.97 ×10 | **+0.000** |

Identical flatness for preserve and food; ui barely sensitive (−0.04); setting *all four*
hot services to 10 makes violation **more** likely on the B tick (0.60→0.63). The reason
is the training distribution (Run 1′, HPA Jul-13):

| service | train count min/med/max | frac ≥ 7 | distinct values |
|---|---|---|---|
| `ts-travel-service` | **5 / 5 / 5** | 0 % | **1** |
| `ts-preserve-service` | 3 / 3 / 3 | 0 % | 1 |
| `ts-food-service` | 3 / 3 / 4 | 0 % | 2 |
| `ts-ui-dashboard` | 8 / 10 / 10 | 100 % | 3 |

The RF has **zero support above travel = 5** — it never observed a world where more
travel replicas exist, so it cannot express an opinion about one. The flatness is
epistemic; the measured outcome agrees with it physically (28,000 ms @ 4 = 28,000 ms @ 9,
§6.1), so the model is accidentally right and the GA accidentally harmless — but for the
wrong reasons, and because the clamp sets `lo = cur + 1`, **every committed cycle
increments regardless of fitness**: the ramps walk monotonically to the cap with 0
direction flips and 96.8 % echo writes. When the oracle is flat, the decision maker is
the clamp.

**(e) Machinery notes.** `cal_weight` correlates the *source's* latency series with the
*destination's* CPU/mem over 60 s (~12 samples); travel's own latency series is a flat
28 s line → Pearson ≈ 0 → its edge weights carry no signal. The resource series is
fetched (`get_svc_metric_range`) but **never used as a node-level saturation score** —
the one measurement that separates travel (util 1.03, hung) from preserve (util 0.11,
hung-but-relay) is in hand at decision time and thrown away.

**Ledger — bug vs missing oracle:**

| layer | classification | evidence |
|---|---|---|
| alphabetical tie-break on flat personalization | **bug** (deterministic, fixable) | auth first to 9 at 1.5 QPS |
| capped nodes permanently excluded | **bug** | travel writes stop 04:43:30Z; signal arrives 06:00Z |
| `ef` on aggregate inbound (inheritance rewards relays) | **design flaw** | preserve ef maximal at util 0.11 |
| no self-time / saturation oracle | **missing oracle** | CPU data fetched, used only as edge corr |
| no action-efficacy feedback | **missing oracle** | 4→9 ramp, Δp90 = 0, never checked |
| GA oracle flat (zero train support > 5) | **missing oracle** (data) | counterfactual + support tables |

### 6.8 Source classification: implementation bugs, not paper claims

Every finding in §6.1–§6.7 was reproduced against the **upstream open-source
repository** (`~/PBScaler`, commit `83820c5`, IEEE TSC 2024). None are paper claims and
none are lab adaptations. Verbatim matches:

| finding | upstream location | match |
|---|---|---|
| flat personalization (`sigma=1` ⇒ e⁻¹ for both graph levels) | `PBScaler.py:269 cal_topology_potential` | **verbatim** |
| alphabetical tie-break | same fn, `ab_mss.sort()` + stable `sorted(reverse=True)` | **verbatim** |
| cap blindness (`svc_counts[root] + 1 < max_num`) | `PBScaler.py:135` | **verbatim** |
| `ef` on aggregate inbound (rewards relays) | `PBScaler.py` `anomaly_score_map` block | **verbatim** |
| waste early-return | `PBScaler.py` `waste_detection` | **verbatim** |
| `execute_task` echo-patch (whole action map every cycle) | `PBScaler.py` `execute_task` | **verbatim** |
| `cal_weight` source-latency ↔ dest-CPU/mem corr | `PBScaler.py:250` | **verbatim** |
| GA `fitness()` uses `action[index]` as the count feature | `util/GA.py` `set_env` | **verbatim** |
| `max_iter=5, size_pop=50` defaults | `PBScaler.py:173` | **verbatim** |
| `schedule.run_pending()` blocking loop | `PBScaler.py` `start()` | **verbatim** |

Two constants differ from upstream but are **not** the cause of any finding: upstream
sets `max_pod = 8` (lab uses 10) and the upstream training set lives at
`/home/ubuntu/xsy/experiment/autoscaling/…` (lab retrained on the Oct-25 dataset). The
flat-oracle result (§6.7d) is a property of the *training distribution*, not of the
lab's model specifically — upstream's paper-mode `[i, qps, count]` layout inherits the
same support limitation.

**Classification of the six layers:**

| layer | bug / design flaw / missing oracle | where |
|---|---|---|
| alphabetical tie-break on flat personalization | **bug** — deterministic, fixable by tie-breaking on `ef` or a saturation signal | upstream `cal_topology_potential` |
| capped nodes permanently excluded | **bug** — `+1 < max_num` filter removes a node from candidacy forever | upstream `choose_action` |
| `ef` on aggregate inbound (inheritance rewards relays) | **design flaw** — scores relays higher than the nodes that cause their latency | upstream `build_abnormal_subgraph` |
| no self-time / saturation oracle | **missing oracle** — CPU/mem fetched (`get_svc_metric_range`) but used only as an edge correlation weight | upstream `cal_weight` |
| no action-efficacy feedback | **missing oracle** — 4→9 ramp with Δp90 = 0 is never checked | upstream `choose_action` |
| GA oracle flat (zero train support > 5 replicas) | **missing oracle (data)** — the classifier was never trained on a world where more replicas exist | upstream training set + `RandomForestClassify.py` |

**Caveat I cannot close:** the repo ships only a BibTeX citation
(`@article{xie2024pbscaler, … IEEE TSC 2024}`), no paper PDF. I can confirm the code is
upstream, but I cannot confirm whether the paper's text implies a richer oracle that the
code never realized. If the paper specifies a per-service resource signal or an
intervention-based fitness, that would be a *paper-vs-code gap* rather than a code bug —
worth checking against the actual PDF before calling any of this a defect in the paper.

---

## 7. Reconciling with the earlier "+7.38 pt improvement"

The 2026-10-25 pipeline reported PBScaler 42.08 % vs KHPA 49.46 % (−7.38 pts). Listing
all arms on their own terms shows the earlier comparison paired arms from different
sessions, where the **HPA baseline was the unstable quantity**:

| arm | phase | autoscaler | violation rate | session |
|---|---|---|---|---|
| Run 1′ | Jul-13 | HPA | 25.42 % | Oct 25 03:55Z |
| Run 3 | Jul-14 | HPA | **49.46 %** | Oct 24 21:16Z |
| **Run A** | Jul-14 | HPA | **21.14 %** | Oct 25 20:45Z |
| Run 2′ | Jul-14 | PBScaler | 42.08 % | Oct 25 11:10Z |
| **Run B** | Jul-14 | PBScaler | **35.07 %** (110 min) | Oct 26 04:48Z |

The HPA arm on the *same* Jul-14 phase moved **49.46 % → 21.14 %** (−28 pts) between
sessions, while the PBScaler arm stayed in a 35–42 % band. The earlier headline was
therefore driven mostly by an inflated HPA baseline (Run 3 launched minutes after 43
HPAs were re-created on a cluster PBScaler had just driven to extremes, and ran with
degraded Mimir: 434 unlabeled ticks). Under same-session pairing the sign **reverses**:
PBScaler is 11.71 pts *worse*.

This is exactly the confound the same-day rerun was designed to remove — and it did.

---

## 8. Implications for the bottleneck-approach proposal

Ranked by expected impact, each tied to a measurement above:

1. **Add an action-efficacy feedback loop (highest value).** After ramping root *r*,
   compare *r*'s edge p90 before/after over the next K cycles; if no improvement, mark
   *r* non-addressable, stop ramping it, and re-rank. Today PBScaler ramps travel
   4 → 9 replicas for **zero** latency change (§6.1) and has no mechanism to notice.
   This alone would free ~60 inert pods.
2. **Make root selection capacity-aware, not just latency-aware.** PageRank ranks
   services with slow *edges*; services slow because they are *starved* are identified
   by utilization against their own limit (`used / limit`). Only 2 of PBScaler's 10
   maxed services exceeded 0.5 utilization; the other 8 sat at 0.11–0.21. Intersect the
   latency graph with a per-pod saturation signal.
3. **Decouple the waste check from the anomaly early-return** so redundancy reduction can
   run while other services are abnormal (`begin scale in` = 0 in 112 min).
4. **Suppress no-op writes** in `execute_task` (96.8 % of 2464 writes changed nothing);
   this also removes the stale-read race the v1.2.0 runaway exploited.
5. **Reduce GA cost or run it asynchronously.** 10.7 s per invocation is why the control
   period is 19.3 s instead of 15 s. Options: shrink `size_pop`/`max_iter`, cache
   predictor inference for repeated candidate vectors, or act on the greedy bound
   (`min_array`) immediately and refine with the GA asynchronously.
6. **Recognize the addressability ceiling.** With a 250 m per-pod CPU limit and
   `max_pod = 10`, any one service is capped at 2.5 cores — a wall **both** arms hit on
   `ts-travel-service`. For hung-path workloads replication cannot fix the SLO; the
   predictor should be able to return "no feasible action" instead of the cap.

7. **Refresh the candidate set; never permanently hide a capped node.** The
   `svc_counts[root] + 1 < max_num` filter made travel invisible from 04:43:30Z onward —
   exactly when the `preserve→travel` (174 s) and `food→travel` (103 s) edges matured
   (§6.7c). Keep capped nodes in the ranking (they are evidence about *other* nodes'
   causality) and make "hold at cap" an explicit action instead of exclusion.
8. **Score nodes by self-time, not inherited latency.** Replace `ef` (aggregate-inbound
   count > SLO) with a per-hop decomposition (`self_time ≈ inbound − Σ children
   call_time`), or at minimum gate it by per-pod CPU utilization (§6.7e). This is the
   measurement that separates travel (self-time dominant, util 1.03) from preserve/food
   (relays, util 0.11–0.74) — and it is already being fetched and discarded.
   All six layers are upstream code, not lab adaptations (§6.8).
9. **Ground the GA in an intervention-capable oracle.** The RF has zero training support
   above travel = 5 (§6.7d), so its fitness surface is flat exactly where the controller
   operates. Retrain on data with replica variance (augment with PBScaler-driven states
   or synthetic count sweeps) or replace the classifier with a latency-response model;
   until then, treat flat-fitness dimensions as "not replica-addressable" and fall back
   to the efficacy test of item 1.

**Paper-facing caveat:** a material share of the violation rate in this regime is
application-level — requests hanging to the 30 s client timeout on the travel/preserve/
food paths, present in *both* arms. That compresses the achievable gap between
autoscalers and inflates any absolute violation rate relative to the paper's regime
(Table II: 5.6–19.4 %). This should be stated explicitly rather than reporting 21 % vs
35 % as a pure control-plane comparison.

---

## 9. Incident log for this run

| time (UTC / Tehran) | event |
|---|---|
| Oct 25 20:45:24 / 00:15 | orchestrator launched; preflight clean (43 HPAs, 0 pbscaler, 0 corpses, disk 70 %, Mimir PF ready) |
| Oct 25 21:00:26 / 00:30 | Run A collector started — picked up the `front_avg` patch 12 min earlier, so **both** arms record avg in-flight at 5 s |
| Oct 26 01:00:26 / 04:30 | Run A complete: 2880 rows, 0 gaps, `RESULT: OK — dataset usable` |
| Oct 26 01:01–01:08 / 04:31 | **switch bug:** readiness probe used `containerStatuses[1]` on a single-container pod → empty jsonpath → `FATAL: pbscaler pod not ready` → orchestrator aborted before Stage B. Cluster state was in fact correct (0 HPAs, v1.2.5 up, md5 verified) |
| Oct 26 01:01–04:33 / 04:31–08:03 | controller idle 3.5 h under no load (waste detection drove fleet to 113 pods); no data loss |
| Oct 26 04:33:21 / 08:03 | Stage B relaunched via fixed `stageB_relaunch.sh`; purity + md5 + 43-svc verified; launch prune (`-mmin +180`) → disk 55 % |
| Oct 26 04:48:21 / 08:18 | Run B collector started on schedule |
| Oct 26 05:19 / 08:49 | checkpoint: 379 ticks, B 41.7 % vs A 24.9 % — divergence detected, investigated in §6 |
| Oct 26 06:22 / 09:52 | checkpoint: 1135 ticks, B 37.8 % vs A 23.2 % — divergence stable, cluster healthy |
| Oct 26 06:38 / 10:08 | this document's data cutoff; Run B continues to 08:48Z / 12:18 |

---

## 10. Reproduction and artifacts

**Datasets**
* Run A: `pbscaler/data/runA_hpa_jul14_4h_20261025_204525/{train_data.csv,metrics.csv,meta.json}`
* Run B: `pbscaler/data/runB_pbscaler_jul14_4h_20261026_043321/{…}`
* Prior arms for §7: `data/run1prime_hpa_jul13_20261025_035556`,
  `data/run3_hpa_jul14_20261024_113741`, `data/run2prime_pbscaler_jul14_20261025_111004`

**Raw captures** (`/tmp/opencode/`)
* `runA_hpa_jul14_4h_{k6,collector,replica_snapshots}.log`
* `runB_pbscaler_jul14_4h_{k6,collector,replica_snapshots}.log`
* `runB_pbscaler_controller_raw.log` — the `[timing]` / `scale` / `clamped` stream
* `ab4h_orchestrator.log` (Run A + failed switch), `stageB_relaunch.log`
* `hpas_backup.yaml` — the 43 Run-A HPAs, for end-state restoration

**Code**
* `pbscaler/PBScaler.py` — timing hooks (`_log_timing` + 6 call sites)
* `pbscaler/util/PrometheusClient.py` — `get_front_avg()` / `get_svc_avg()`
* `pbscaler/scripts/collect_training_data.py` — `front_avg` + per-service `&avg`
* `pbscaler/scripts/analyze_ab4h.py` — report/timing/replica/load-shape generator
* `pbscaler/scripts/reconstruct_front_avg.py` — Mimir fallback (unused: both arms
  recorded avg in-flight)

**Regenerate the full report + plots** (run after 12:18 Tehran):

```bash
cd pbscaler && ./.venv/bin/python scripts/analyze_ab4h.py \
  --runA data/runA_hpa_jul14_4h_20261025_204525 \
  --runB data/runB_pbscaler_jul14_4h_20261026_043321 \
  --outdir /proj/k8sautoscaledl-PG0/logs/pbscaler/ab4h_rerun
```

---

## 11. Open items

1. Restate §3/§6/§7 with the complete 240-min Run B at ~12:18 Tehran.
2. Restore the 43 HPAs (`hpas_backup.yaml`) and verify cluster end state.
3. Decide whether to re-run the pair with the travel/preserve hang mitigated (raised
   per-pod CPU limits or a shorter client timeout) so the A/B measures control-plane
   quality rather than an application pathology.
4. If §8.1 (efficacy feedback) is implemented, re-run this A/B — the prediction from
   this document is that it converts PBScaler's +46-pod penalty into a net win, because
   the 60 inert pods are the entire difference in delivered work (§6.3).


# A/B @ MAX_REQUESTS=5000 (NASA Jul-14 slice, open-loop): PBScaler vs native HPA

**Date:** 2026-11-02 → 2026-11-03 (all times UTC) · **Cluster:** lab TrainTicket (3 master + 3 worker,
31.4 CPU cores / 64 GB each) · **Experiment target:** http://10.10.1.2:30080 (private LAN only)
**Status:** complete. All four runs used the identical offer: Jul-14 255-min slice, 15 min
warmup + 240 min measured, `TRACE_COMPRESS=1`, `OPEN_LOOP=1`, `USER_POOL=200`,
`MAX_REQUESTS=5000`, collector 5 s cadence inside the cluster (pod `pbscaler-collector`).

---

## 1. Question and verdict

**Question:** at the same offered load (Jul-14 slice @ MAX_REQUESTS=5000, peak ≈ 68 rps at the
front), how do native HPA (43 HPAs, `maxReplicas=8`, CPU 80 %) and PBScaler
(`PBS_MAX_POD=8`, RF SLO-violation predictor + TopoRank + GA) compare on (a) SLO violations of
the front-end 200 ms P90, and (b) the replicas they actually assign?

**Verdict: PBScaler kept the SLO perfectly but bought it with a ~2× larger fleet and a
never-released one-way ratchet.** HPA violated the SLO in one concentrated window
(198 of 2880 samples = 16.5 min of 240) driven by the gateway hitting its cap of 8; PBScaler
never exceeded the physical baseline (9.5–9.9 ms) all 240 min — but grew 43 → 147 pods, held a
median of +49 pods over the 1-replica baseline, issued **79 scale-out rounds and 0 scale-ins
while load was live**, and never used its configured max of 8 (every saturation landed at 7).

| Hypothesis | Verdict | Evidence |
|---|---|---|
| PBScaler fixes HPA's SLO misses at 5000 | **Supported** | 6.88 % → 0.00 % violations; front P90 max 965.3 ms → 9.9 ms |
| PBScaler does it efficiently (paper's waste term) | **Refuted** | 22,034 vs 11,201 pod-minutes (×1.97); +49 mean extra pods vs +3.7; 27/43 services scaled vs 5/43 |
| Both autoscalers ran under an equal-cap regime | **Refuted in detail** | HPAs genuinely used 8 (`train-ticket-gateway-istio` held 8 for 71 min); PBScaler's per-service peak was **7** every time (0 commands `to 8`) |

---

## 2. Experiment configuration

| parameter | Jul-14 HPA @5000 | Jul-14 PBScaler @5000 |
|---|---|---|
| workload | `mcr_jul14_slice.csv` (255 min, mean/peak 0.572), replay via k6 `ramping-arrival-rate` | same, byte-identical file & settings |
| autoscaler | 43 native HPAs, target CPU 80 %, min 1 / **max 8** | HPAs suspended (`PBS_SUSPEND_HPAS=false` path), all stateless at 1 |
| PBScaler caps | — | `PBS_MAX_POD=8`, `PBS_MIN_POD=1`, `SLO=200 ms`, `LAMBDA=0.5` |
| RF predictor | — | `simulation/train_ticket/RandomForestClassify.model`, 43 svc × [i, qps, count] |
| controller loops | — | 15 s SLO/anomaly loop + 120 s redundancy loop (`schedule`) |
| k6 summary | 726,049 iters / 4h15m / 54 iters/s / 0 interrupted | 728,339 iters / 4h15m / 54 iters/s / 0 interrupted |
| collector window | 14:09:33Z → 18:09:28Z, 2880 samples | 20:49:49Z → 00:49:44Z, 2880 samples |
| dataset (pod `/data/`) | `exp_jul14_hpa_20261102_140929` | `exp_jul14_pbscaler_20261102_204943` |

Jul-13 reference (same offer, `MAX_REQUESTS=5000`, HPA active): 2880 samples, **0 violations**,
front P90 max 10.5 ms, pods median 62 / max 109 — the Jul-13 slice is structurally gentler
(mean/peak 0.478) and never violates even at 5000. It therefore cannot serve as RF training
data (single class).

**Model/evaluation leakage caveat:** the deployed RF was trained on the three Jul-14 HPA @5000
runs (8,640 rows, 594 violation labels) → same trace, same epoch that PBScaler was then scored
on. In-sample. Metrics: acc 0.9995, recall 0.9936, AUC 0.9967 (cut 25 % holdout). The intended
cross-day plan (train Jul-13, test Jul-14) is impossible with the current Jul-13 data because
the HPA produced no violation labels there.

---

## 3. Headline result

| metric (2880 labels each) | Jul-14 HPA @5000 | Jul-14 PBScaler @5000 | delta |
|---|---|---|---|
| SLO violations (front P90 > 200 ms) | **198 (6.88 %)** | **0 (0.00 %)** | −6.88 pts |
| front P90 median | 9.7 ms | 9.7 ms | 0 |
| front P90 mean | 67.8 ms | 9.7 ms | −58.1 ms |
| front P90 p95 / p99 | 780.6 / 946.5 ms | 9.8 / 9.8 ms | ≈ ×10/×100 |
| front P90 max | 965.3 ms | 9.9 ms | −955 ms |
| violation-phase function | 7 bursts, longest 65 samples (≈ 5.4 min), window min 57–130 | none | — |
| total pods, median / mean / max | 43 / 46.7 / 60 | 87 / 91.8 / 147 | ×1.97 median |
| extra pods over the 1-replica baseline (43) | mean **3.7**, max 17 | mean **48.8**, max 104 | ×13 |
| pod-minutes over 240 min | 11,201 | 22,034 | ×1.97 |
| median total CPU (cores, all svcs) | 8.99 | 8.17 | ≈ equal |
| median total mem (`container_memory_usage_bytes`, MB) | 38,257 | 74,012 | ×1.94 |
| k6 delivered iterations | 726,049 | 728,339 | +0.3 % |

Interpretation in one line: PBScaler reached the SLO by almost doubling the fleet; HPA stayed
cheap but gave up 16.5 min of front-P90 SLO in a single multi-burst window. The active CPU the
two systems consume is nearly identical — the difference is *held capacity*, not *used compute*.

---

## 4. Chronology

**HPA violation window (minutes 57–130 of the run).** The offered Jul-14 slice peaks there;
front P90 crosses 200 ms, reaching 965 ms. `train-ticket-gateway-istio` — the only workload that
is entry-limited — sits at its HPA cap **8 for 856 samples (71.3 min)**, which is exactly this
window: the gateway saturated, everything behind it queued, and CPU-based scaling of the
remaining services never became the constraint (they mostly stayed at 1: median total pods 43
across the whole window). After min 150 the HPAs added ~12 pods (55 total, mostly
`ts-ui-dashboard` at 5 and late scaling of `ts-order-other-service`/`ts-security-service`), i.e.
the fleet grew *after* the violations had passed.

**PBScaler ratchet.** Scale-out rounds fire 79 times; the fleet grows monotonically:

```
hour        0     1     2     3     4
PBScaler   43    62    87   116   147 pods
HPA        43    43    43    54    55 pods
```

Even as offered qps declines through the second half (block medians 897 → 553 → 613 ≈ half the
initial intensity), nothing is ever scaled back. Both `begin scale in` rounds in the log occur
at 00:51–00:53Z — *after* k6 terminated at 00:49:44Z. During 240 loaded minutes the controller
reduced exactly zero replicas, and per the paper's redundancy loop that reclaiming never
triggered while load was live.

Internal edges still showed brief anomalies under PBScaler ("[anomaly] SLO-violating calls: 1,
SLO=200.0 ms" repeatedly in the controller log) — the front end never did.

---

## 5. Assigned replicas — per service

Services ever above 1 replica: HPA **5 of 43** vs PBScaler **27 of 43**.

**Jul-14 HPA @5000** (limiting set was the ingress path):

| service | max | mean | samples at max | % time >1 |
|---|---|---|---|---|
| train-ticket-gateway-istio **(HPA cap 8)** | **8** | 3.08 | 856 | 30 % |
| ts-ui-dashboard | 5 | 2.06 | 483 | 30 % |
| ts-order-other-service | 4 | 1.15 | 102 | 8 % |
| ts-security-service | 3 | 1.09 | 64 | 7 % |
| ts-auth-service | 2 | 1.29 | 835 | 29 % |
| all other 38 services | 1 | 1.00 | — | 0 % |

**Jul-14 PBScaler @5000** (broad, flatter, capped at an effective 7):

| service | max | mean | samples at max | % time >1 |
|---|---|---|---|---|
| ts-ui-dashboard | 7 | 6.71 | 2732 | 96 % |
| ts-auth-service | 7 | 6.71 | 2732 | 96 % |
| ts-admin-basic-info-service | 7 | 5.11 | 1413 | 94 % |
| ts-preserve-other-service | 7 | 4.93 | 1287 | 92 % |
| ts-preserve-service | 7 | 4.85 | 1365 | 93 % |
| ts-contacts-service | 7 | 4.42 | 1173 | 94 % |
| ts-food-service | 7 | 4.07 | 468 | 86 % |
| ts-travel-service | 7 | 3.65 | 554 | 67 % |
| ts-security-service | 7 | 2.38 | 472 | 39 % |
| ts-basic-service | 7 | 3.18 | 347 | 73 % |
| ts-food-map-service | 7 | 2.08 | 21 | 43 % |
| ts-station-service | 6 | 2.05 | 69 | 69 % |
| ts-ticketinfo-service | 6 | 2.79 | 243 | 73 % |
| ts-admin-order-service | 5 | 2.43 | 768 | 47 % |
| ts-travel2-service | 5 | 2.54 | 368 | 92 % |
| ts-consign-service | 4 | 2.29 | 180 | 63 % |
| ts-admin-route-service | 4 | 1.47 | 147 | 35 % |
| ts-seat-service | 4 | 1.37 | 64 | 32 % |
| ts-news-service | 3 | 1.96 | 865 | 66 % |
| ts-order-other-service | 3 | 1.45 | 184 | 39 % |
| ts-train-service / ts-route-service / ts-consign-price-service / ts-rebook-service / ts-config-service / ts-payment-service / ts-order-service | 2 | 1.0–1.6 | — | 1–62 % |
| all other 16 services | 1 | 1.00 | — | 0 % |

Reading the two tables against each other: the topology-aware controller spread capacity
across the whole call graph (27 services) instead of the CPU-facing ingress cone (5 services),
and pinned the hot ones at their effective ceiling for ~95 % of the run regardless of the load
curve. HPA meanwhile left 38 services untouched and concentrated on the gateway.

---

## 6. Answer: PBScaler's max replicas

* **Configured cap: `PBS_MAX_POD=8`** (`pbscaler/config/Config.py`, env-overridable) — equal to
  the live HPAs' `maxReplicas=8`, so the paper-style cap of 8 was honored symmetrically.
* **Effective observed per-service peak: 7.** Across all 277 scale actions there are
  **0 commands to 8** and 482 commands to ≥ 6; every hot service maxed out at exactly 7
  (e.g. `scale ts-ui-dashboard to 7` was commanded 74 times).
* **Why:** two properties interact, **both inherited verbatim from the upstream code**
  (`~/PBScaler`, the IEEE TSC 2024 reference implementation) rather than introduced by the
  lab adaptation — upstream `PBScaler.py:135` carries the identical filter and `util/GA.py`
  the identical fitness, and upstream even hardcodes the same default `max_pod = 8`.
  1. `PBScaler.py:205` — `self.roots = [root for root in res if svc_counts[root] + 1 < max_num]`.
     A service at 7 has `7 + 1 = 8`, which is not `< 8`, so it may never be re-queued for
     another bump; a service is only *selected* while a further +1 keeps it below the cap, so
     the search never proposes 8 (reaching 8 would require re-selecting a service sitting at 7).
  2. The GA fitness
     `f = 0.5·pred_violation_reward + 0.5·(1 − Σaction/Σub)` stops at the *lowest* count the
     RF labels safe; given the predictor's learned boundary (trained on HPA-run data in which
     low counts at high qps are violation-labeled) that stops at 7 for the hot services.
  Net effect: **PBScaler effectively runs with a per-service ceiling of `max_pod − 1`** — with
  this run's cap of 8 that is 7, which is why the last step of the cap was left untouched. The
  vendored default (`PBS_MAX_POD=10`) would put the invisible ceiling at 9 — consistent with
  the earlier 2026-10-26 observation where PBScaler held `ts-travel` at exactly 9. With the
  cluster's HPA cap also at 8, the symmetric-cap claim still stands at the configured level,
  but only HPA actually used its last step.

---

## 7. Cost accounting

| cost unit (both over identical 240 min) | Jul-14 HPA @5000 | Jul-14 PBScaler @5000 | ratio |
|---|---|---|---|
| pod-minutes (all 43 stateless svcs) | 11,201 | 22,034 | **×1.97** |
| extra pod-minutes vs 1-replica baseline | 427 | 10,833 | **×25** |
| median held memory (GB) | 37.4 | 72.3 | ×1.94 |
| median active CPU (cores) | 8.99 | 8.17 | 0.91 |

The last row is the crux: PBScaler is not burning *more compute* — it is *holding* ~2× more
allocatable slots. On a shared cluster this is exactly the kind of cost invisibility that
makes "0 % violations" claims fragile; the efficient frontier here is *HPA ≈ 11.2k pod-min,
6.88 % violations* vs *PBScaler ≈ 22.0k pod-min, 0 % violations*.

---

## 8. PBScaler weaknesses (ranked, with mechanism)

1. **One-way scale ratchet — no live scale-down.** 79 begin-scale-out vs 2 begin-scale-in rounds
   in 240 min, both at post-load teardown; fleet grows 43→147 monotonically while offered qps
   halves. The 120 s redundancy/waste loop's reclaim path never fired under live load, violating
   the intent of the paper's waste objective.
2. **Predictor boundary is trained to over-provision.** Trained on violation-labeled low-count
   rows, the RF needs abnormally high counts to predict "no violation" at high qps; the GA then
   settles on the boundary (7) and holds it independent of the actual demand curve. The same
   RF+GA coupling is why 27 services climbed — breadth, not depth, tuned by the model.
3. **Effective cap = max_pod − 1 (upstream-inherited).** `PBScaler.py:205` prevents any service
   already at the cap from being re-queued, so the configured 8 is a deceptive ceiling: the
   system never actually reaches it (7 everywhere). This is the paper reference code's own
   behavior (byte-identical in `~/PBScaler/PBScaler.py:135`, default `max_pod=8` there too) —
   not an artifact of the lab port. Since the vendored copy is the deployed baseline, either
   document the `max_pod − 1` semantics or deliberately fix them
   (`<= max_num` on the filter + explicit ≥ cap handling in the GA bounds).
4. **`R2` waste term unscaled.** `1 − Σaction/Σub` weights each added replica equally regardless
   of the service's real contribution, and LAMBDA=0.5 gives a model-driven demand a huge budget;
   combined with (2) the fleet grows breadth-first. Retraining the RF on a slower-reacting load
   profile (HPA with `stabilizationWindowSeconds`), or lowering LAMBDA toward the waste term,
   would pull the boundary down.
5. **In-sample predictor.** The deployed RF (acc 0.9995/recall 0.9936/AUC 0.9967) trained on
   the *same* Jul-14 epoch it was evaluated on; the cross-day dataset that was intended for
   this (Jul-13 @5000) has 0 positive labels and cannot train a two-class classifier. The
   sweeping generalization question — does the boundary hold on an unseen trace? — remains
   open, and history (2026-10-26 run, 35 % violations at +40 % pods) shows it can invert.

---

## 8bis. Stability accounting — did anything break? (append 2026-11-03)

Sources: live pods + Mimir kube metrics queried over each 240-min window
(`kube_pod_container_status_restarts_total` increase, `*_last_terminated_reason`, node
`MemAvailable`), istio request totals per response code, and the collector's per-service
`success_rate` series. k6: 0 interrupted iterations in all arms.

Stability: **clean.** Zero restarts in both 240-min windows, 0 OOMKilled and 0 Evicted
last-terminated states; min node memory available 52.8 GB (HPA window) / 45.7 GB (PBScaler
window); hot services at 7 replicas held per-pod `container_memory_usage_bytes` of
0.87–1.30 GB (includes reclaimable page cache — working set is lower; no OOM judge state
appeared). CPU at the caps: per-pod p95 ≤ 0.25 core everywhere (PBScaler fleet), ≤ 0.57 core
(`train-ticket-gateway-istio` at its single-replica config under PBScaler; under HPA the
gateway p95 was 0.10 core/pod × 8). **At neither arm's replica cap was any pod
resource-saturated** — the HPA window's violations were queue/pool effects (istio, tomcat,
DB), not CPU or memory cliffs; the gateway at its 8-cap ran at ~8 % CPU/pod.

Request-failure accounting (istio totals per 240-min window, destination-reported):

| workload | code | HPA arm | PBScaler arm |
|---|---|---|---|
| ts-ui-dashboard | 200 | 7,507,755 | 4,999,412 |
| ts-ui-dashboard | 404 | 565,458 | 376,179 |
| ts-ui-dashboard | 403 | 237,878 | 158,686 |
| ts-ui-dashboard | 500 | **207,735** | **138,974** |
| ts-ui-dashboard | 400 | 119,375 | 79,293 |
| ts-ui-dashboard | 503 | 4,834 | 1 |
| ts-preserve-service | 500 | 88,944 | 59,187 |
| ts-preserve-other-service | 500 | 59,407 | 39,817 |

Reading the numbers:

1. **~2.6 % of front-end calls end in 500 in *both* arms** (207.7k/8.0 M vs 139.0k/5.4 M) —
   app-level constant, present from the first collector sample of every run, unchanged by the
   autoscaler. The 404/403/400 shares likewise match between arms.
2. **`ts-preserve-service` and `ts-preserve-other-service` return 100 % 500s in every arm**
   (incl. the identically clean Jul-13 run): the booking endpoints are broken app-side,
   deterministically, ~6 qps. Fix before the next experiment set — the k6 "book" leg has been
   failing silently in 100 % of all journeys across every experiment in this campaign.
3. **Stress amplification is HPA-specific:** per journey (completed k6 iteration), the HPA arm
   produced 8.0 M/726k ≈ **11.0** front-end calls vs PBScaler's 5.4 M/728k ≈ **7.4**. The
   +48 % is consistent with istio/client retries and re-polls against a congested fleet — the
   congestion itself manufactures load. This residual also explains why the HPA arm's raw
   internal qps ran 1000–1400 while PBScaler's ran 550–1100 at the same offered journey rate.
4. **Front-end success floor under stress:** HPA arm dipped to min 0.9023 (a ~10 % 5xx burst
   for 12 consecutive 5-s samples at 15:37Z, inside the 57–130-min violation window);
   PBScaler arm never fell below 0.9720 (its app baseline). The 503 counter (gateway-side
   unavailability): 4,834 during the HPA arm vs **1** during the PBScaler arm.
5. Ten-minute-jet artifacts: services'idle `p99` columns report the istio histogram's
   3,600,000 ms top-bucket edge when there is no traffic in the interval — excluded from
   every latency conclusion above.

---


## 8ter. Headroom — room for a higher MAX_REQUESTS?

| arm | verdict on raising above 5000 | basis |
|---|---|---|
| Jul-14 HPA | **No.** Already over the knee: 6.88 % of intervals violating, 16.5 min cumulative, deeper 5xx dips (success min 0.902, 4,834 gateway 503s) — more offer only deepens the same failure mode | this run |
| Jul-14 PBScaler | **Yes, on all measured signals — up to an unmeasured cliff.** Front P90 finished the run at 9.5–9.9 ms (20× under SLO), per-pod CPU p95 ≤ 0.25 core, per-pod mem ≤ 1.30 GB incl. cache, fleet still able to grow (effective cap 7 was untouched), 100 % success other than the app-level baseline | this run |
| PBScaler caution | The region between ~83 rps peak (this run) and the 2026-10-26 collapse evidence (~126 rps peak offer with compress=8 / closed-loop 560 VUs / older model: 35 % violations, front p95 ≈ 16 s) is unmeasured with the current setup | prior ab4h report |

Recommended before any decisive statement: a staircase of short PBScaler probes at
MAX_REQUESTS = 5500 → 6000 → 6500 → 7000 (15 min each, HPA arm parked at 5000 as the fixed
reference) to bracket the knee of the queue/retry-amplification curve; the first stair that
shows `ts-ui-dashboard` 500-rate rising above ~2.8 % or front P90 leaving the 9.x-ms band is
the practical ceiling for this offer shape.

---

## 9. Threats to validity

* **Sequential runs.** The two arms ran back-to-back (14:09Z / 20:49Z), not simultaneously;
  collector start offsets against the trace replay differ (per-block `total_qps` medians diverge:
  HPA 684→1415 vs PBScaler 1077→612), so minute-aligned qps comparisons are approximate.
  Headline per-run aggregates are unaffected.
* **Collector resumption.** The PBScaler collector was restarted manually twice after dying
  (loguru stream died with runner-side exec sessions); the analysis here uses the full 2880-row
  first collector window.
* **Open-loop arrival replay** remains demand-anchored: `front_p90` reacts to *offered* rps, not
  closed-loop pressure, so maxlen bursts under HPA reflect server slowness at fixed offered
  intensity (the proper comparison condition).
* **Single trace pair.** Jul-13 vs Jul-14 differ structurally (mean/peak 0.478 vs 0.572); all
  conclusions are slice-specific.

---

## 10. Artifacts & locations

| artifact | path |
|---|---|
| HPA arm metrics | pod `/data/exp_jul14_hpa_20261102_140929/{train_data,metrics}.csv`; runner `/tmp/opencode/metrics_jul14_hpa.csv` |
| PBScaler arm metrics | pod `/data/exp_jul14_pbscaler_20261102_204943/…`; runner `/tmp/opencode/metrics_jul14_pbscaler.csv` |
| Jul-13 HPA @5000 (reference, no violations) | pod `/data/exp_jul13_hpa_5000_20261102_192854/…`; runner `/tmp/opencode/metrics_jul13_hpa5000.csv` |
| PBScaler controller log (79 out / 2 in rounds) | `/tmp/opencode/exp_jul14_pbscaler_pod_controller5.log` |
| k6 end logs | `/tmp/opencode/exp_jul14_hpa_final.log`, `/tmp/opencode/exp_jul14_pbscaler_final.log`, `/tmp/opencode/exp_jul13_hpa_5000.log` |
| RF model + feature order | `pbscaler/simulation/train_ticket/RandomForestClassify.{model,svcs}` |
| RF training log | `/tmp/opencode/rf_jul14_final.log` |
| controller runtime | pod `pbscaler-controller` (monitoring ns, `/tmp/pbscaler`), `PBS_MIMIR_URL=http://mimir.monitoring.svc.cluster.local:8080` |

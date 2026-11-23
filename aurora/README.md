# AURORA-HPA — Adaptive, Uncertainty-aware, Resource-Oriented Replica Autoscaler

AURORA-HPA is an application-independent Kubernetes horizontal autoscaler. It
decides **replica counts only**, using a shared self-supervised temporal encoder,
few-shot bottleneck classification, per-service capacity estimation and a
lifecycle-aware reconciler, on a 15-second control loop fed by 5-second
telemetry.

This document describes the implementation as it exists in this tree. It is the
authority for component responsibilities and data flow; where code and this
document disagree, the code wins and this document is stale.

---

## 1. Design contract (non-negotiable)

| # | Rule | Where enforced |
|---|---|---|
| 1 | **No forecasting.** CPU, memory and future resource requirements are never predicted. | `models/`, `decision/engine.py` |
| 2 | CPU/memory are **inputs** describing current condition and bottlenecks, never prediction targets. | `features/groups.py` |
| 3 | Single replica writer per managed workload. | `k8s/scale_client.py`, `lifecycle/reconciler.py` |
| 4 | Pending pods are **committed**, never **serving**. | `lifecycle/reconciler.py` |
| 5 | Never stack a second scale-out while viable pods are still starting. | `lifecycle/reconciler.py` |
| 6 | No global retraining inside the 15 s loop. | `drift/adaptation.py` (L3 is offline) |
| 7 | Unverified outcomes never become labels. | `feedback/ledger.py`, `drift/adaptation.py` |
| 8 | A masked observation is not a zero. | `telemetry/schema.py` |
| 9 | Runtime-agnostic: cAdvisor + kube-state-metrics + Istio only. | `features/groups.py` |
10 | 5 s resolution preserved end to end; decisions are not made on 15 s averages. | `features/window.py` |
| 11 | Only the Scale subresource is written. | `k8s/scale_client.py` |

### 1.1 The forecasting rule in detail

`DecisionEngine` scores candidate replica counts with

```
U(N) = w_perf * P(N * capacity >= offered_now)
     - w_cost * N / max_replicas
     - w_risk * sigma_capacity
     - w_change * |N - N_committed| / max_replicas
```

`offered_now` is the **newest observed** request rate. `capacity` is a posterior
over **observed** goodput per replica. There is no term for a future value, no
horizon, and no forecast head. The same applies to the lifecycle reconciler,
which reads live Scale/pod state.

---

## 2. Repository layout

```
aurora/
  config.py                  env-driven configuration (AURORA_*)
  main.py                    entry point: load model, suspend HPAs, run loop
  telemetry/
    schema.py                Observation / Batch, availability mask semantics
    mimir.py                 one bulk range query per metric family
  features/
    groups.py                22 features, their queries, and derived ratios
    build.py                 dataframe -> model-ready [T, F] windows
    window.py                rolling per-service history + running normaliser
  models/
    encoder.py               TS2Vec-style encoder + contrastive loss
    proto.py                 prototypical bottleneck head + addressability head
    registry.py              versioned, atomic, rollback-able model store
  capacity/estimator.py      partial-pooled goodput-per-replica posterior
  decision/engine.py         expected-utility action selection
  lifecycle/reconciler.py    N_committed, pending viability, single-writer guard
  drift/detector.py          MMD input drift, CUSUM capacity drift, config/topology hashes
  drift/adaptation.py        Level 1/2 online adaptation; Level 3/4 offline
  feedback/ledger.py         append-only action/outcome ledger
  graph/topology.py          dependency graph from observed Istio traffic
  profiles/store.py          deployment + service profiles, persisted
  k8s/scale_client.py        Scale subresource, pod accounting, HPA suspend/restore
  control/loop.py            the serialised 15 s control loop + stage timings
  train/
    harvest.py               Mimir -> parquet corpus
    weak_labels.py           telemetry-derived bottleneck labels
    selfsup.py               self-supervised encoder pretraining
    calibrate.py             prototype + addressability calibration, publish
  deploy/
    Dockerfile               python:3.10-slim + CPU torch
    rbac.yaml                ServiceAccount/Role/RoleBinding
    deployment.yaml          PVC + Deployment (PBScaler-identical resources)
    build_and_push.sh        build, bake the published model, push
  tests/
    test_core.py             37 tests: schema, window, capacity, decision, lifecycle, drift, registry
    test_models.py           encoder, heads, weak labels
```

---

## 3. Timing model

| Parameter | Value | Constant |
|---|---|---|
| Telemetry sampling | 5 s | `AURORA_TELEMETRY_STEP_S` |
| Decision interval | 15 s | `AURORA_DECISION_INTERVAL_S` |
| New observations per decision | 3 (normal) | derived |
| History length | 24 observations = 2 min | `AURORA_HISTORY_LEN` |
| Stale after | 2 intervals (10 s) | `AURORA_STALE_AFTER_INTERVALS` |

```
t=0   obs1 ─┐
t=5   obs2  ├─ appended to each service's rolling window
t=10  obs3 ─┘
t=15  -----> decision cycle (read newest complete observations, infer, decide, reconcile)
```

The loop is **serialised**: `cycle()` runs to completion before the next one
starts. If a cycle exceeds 15 s the next begins immediately and the overrun is
logged. Two concurrent writers are never possible, which is what makes rule 3
safe.

Scrape jitter is tolerated: each sample is placed on the 5 s grid by its **own
timestamp**, and a row with no sample within ±3.75 s keeps `mask=0`. Missing
telemetry never becomes a zero, and never becomes a scaling decision — the
service is skipped and counted as a hold.

---

## 4. Telemetry and the feature schema

### 4.1 Collection

`MimirClient.fetch()` issues **one range query per metric family**, each
aggregating over all services (`sum by (container)`, `sum by (deployment)`,
`sum by (destination_workload)`, `sum by (pod)`), then slices the result per
service. 21 queries per cycle, ~1.4 queries/second — deliberately not one query
per service, which at 42 services would be ~900 queries per cycle and would
make the collector the slowest stage.

Pod-level metrics (network, pod start time) carry no container label, so they
are resolved back to a service by longest-prefix match
(`_resolve_pod` in `telemetry/mimir.py`).

### 4.2 Features (22)

| Group | Features |
|---|---|
| workload | `req_rate`, `err_rate`, `lat_p50`, `lat_p90`, `lat_p99` |
| CPU | `cpu_cores`, `cpu_limit`, `cpu_throttled`, `cpu_periods` |
| memory | `mem_working_set`, `mem_limit`, `mem_page_faults`, `oom_events` |
| network | `net_rx`, `net_tx` |
| storage | `fs_read`, `fs_write`, `io_time` |
| lifecycle | `replicas`, `ready`, `available`, `pod_start_time` |

Derived (computed, never fetched): `success_rate`, `completion_ratio`,
`cpu_limit_pct`, `mem_limit_pct`, `cpu_throttled_pct`, `cpu_per_request`,
`lat_ratio_p90_p50`, `inflight` (Little's law), `inflight_per_replica`,
`pending`, `cpu_per_replica`, `req_per_replica`, `cpu_headroom`, `mem_headroom`.

Two of these exist specifically because the SLO is blind to outage:

- `completion_ratio = 1 − (5xx + conn_fail) / req_rate` — requests that
  actually **finished**. A timeout (`response_code="0"`) counts as a
  non-completion, because it is an outage, not a slow success.
- `latency_available` / `blindness_ratio` — a tick with arrivals but no latency
  sample is **blind**, not fast. The estimator excludes the latency signal when
  the blindness ratio is high instead of reading missing data as a good result.

Every derived ratio returns `None` unless **both** inputs are present, so a
missing denominator can never masquerade as a healthy zero.

`pod_start_time` is converted to `pod_age_s` before it reaches the model — the
absolute epoch is not a useful feature.

### 4.3 Availability mask

Each feature carries three states:

| State | value | mask | meaning |
|---|---|---|---|
| measured | float | 1 | real observation, possibly genuinely 0 |
| unavailable | `None` | 0 | metric absent for this pod |
| stale | — | 0 (whole row) | nothing ingested within the staleness bound |

`tests/test_core.py::TestSchema` locks the zero-vs-missing distinction.

---

## 5. Model

### 5.1 Shared temporal encoder

`TS2VecEncoder` (models/encoder.py) — projection to `d_model=64`, three dilated
`Conv1d` blocks (dilations 1/2/4) with residual GELU and LayerNorm. Input
`(B, T, F)`, output per-timestep embeddings `(B, T, D)`; a masked mean gives the
sequence vector.

Trained self-supervised on **unlabelled** windows from all services
(`ts2vec_loss`: instance discrimination — two augmented views, same timestamp is
the positive, other timestamps are negatives). Augmentation is small jitter plus
random temporal masking. No labels and no forecasting objective are used.

This is the "shared learning" level: one encoder, many services and
deployments. A new deployment inherits it.

### 5.2 Prototypical bottleneck head

`PrototypicalHead` (models/proto.py) holds one prototype vector per class:

```
cpu, memory, network, storage, queue,
shared_dependency, app_problem, infra_problem,
transition, healthy, unknown
```

Logits are `-distance / temperature`. Adding or updating a service's knowledge
is `set_prototype()` / `add_prototype()` — an EMA update on the prototype
vector, **no encoder retraining**. That is the few-shot mechanism.

### 5.3 Addressability head

`AddressabilityHead` is an independent 2-layer MLP answering
`P(adding replicas improves this service)`. It is deliberately separate from the
bottleneck classifier: detecting `cpu` does not imply horizontal scaling helps,
and detecting `shared_dependency` must not cause a scale-out.

### 5.4 Outputs

`AuroraModel.decide()` returns, per service:

| Output | Meaning |
|---|---|
| `representation` | sequence embedding (D) |
| `bottleneck` | class name |
| `bottleneck_probs` | full distribution |
| `addressable_prob` | P(horizontal scaling helps) |
| `confidence` | max softmax probability |
| `ood` | nearest-prototype distance above the calibrated threshold |

No future values are produced.

### 5.5 Calibration

`train/calibrate.py` fits prototypes from telemetry-derived weak labels
(balanced per class, capped at 200 windows each), sets the OOD threshold at the
99th percentile of nearest-prototype distance, and fits the addressability head
against the class prior (`cpu`, `memory`, `network`, `storage`, `queue` are
addressable-capable). The result is published as an immutable version.

---

## 6. Training pipeline

```
run dirs (params.json + launch_epoch)
        │  train/harvest.py
        ▼
corpus.parquet          422,067 rows × 25 cols, 64 services, 5 s resolution
        │  train/weak_labels.py
        ▼
corpus_labelled.parquet  + bottleneck column (telemetry-derived weak labels)
        │  train/selfsup.py
        ▼
encoder.pt               self-supervised weights + global norm statistics
        │  train/calibrate.py
        ▼
data/registry/<version>/ model.pt, feature_schema.json, validation.json
        │  deploy/build_and_push.sh
        ▼
docker image (model baked in)
```

### 6.1 Weak supervision

`train/weak_labels.py` converts each raw feature to a **within-service
percentile**, so "high" means "unusual for this service", not "above a universal
constant". Priority order resolves ties:

1. `transition` — pending replicas, or pod age < 60 s
2. `memory` — working set at the service's 97th percentile **and** page faults/OOM high
3. `cpu` — `cpu_limit_pct` and `cpu_throttled_pct` both at the 90th percentile
4. `shared_dependency` — own p90 high while CPU is calm
5. `queue` — own p90 high with CPU near the limit
6. `network`, `storage` — respective pressure at the high end
7. `healthy` — nothing above

These thresholds exist **only to create training labels**. Runtime classification
comes from the learned prototypes plus the service's own baselines; the
labeller's percentiles never gate a scaling decision.

Measured corpus: `healthy 315,464 · transition 44,020 · network 31,274 ·
storage 21,831 · cpu 4,265 · shared_dependency 4,004 · memory 986 · queue 223`.

---

## 7. Service state and capacity estimator

`capacity/estimator.py`. The previous design reduced a service to one scalar,
`capacity = goodput / replicas`, and the decision ignored CPU, memory and
latency entirely. Goodput is now **one input among six**, and capacity is
estimated per binding signal.

Per service, per tick, normalised per replica:

```
arr_per_rep   = req_rate / replicas
cpu_lim_rep   = cpu_limit / replicas
cpu_use_rep   = cpu_cores / replicas
mem_lim_rep   = mem_limit / replicas
mem_use_rep   = mem_working_set / replicas
```

| signal | capacity (arrivals·s⁻¹·replica⁻¹) | meaning |
|---|---|
| `cap_cpu` | `cpu_lim_rep · u_cpu / (cpu_use_rep / arr_per_rep)` | arrivals before CPU binds |
| `cap_memory` | `mem_lim_rep · u_mem / (mem_use_rep / arr_per_rep)` | …before memory binds |
| `cap_latency` | `arr_per_rep · (lat_baseline / p90)` | …before latency degrades vs **its own** baseline |
| `cap_throttle` | `arr_per_rep · (0.20 / throttled_fraction)` | …before CFS throttling dominates |
| `cap_queue` | `arr_per_rep · (inflight_baseline / inflight_per_rep)` | …before in-flight exceeds its own baseline |
| `cap_goodput` | `completed / replicas` | observed completion — **reported, never part of the minimum** |

**The binding capacity is `min(cap_cpu, cap_memory, cap_latency, cap_throttle,
cap_queue)` and the argmin *is* the bottleneck.** No separate guess.

Three properties make this different from a fixed threshold:

- **`u_cpu` and `u_mem` are measured, not assumed.** They are the highest
  utilisation the service has sustained while *not* throttling / without page
  faults. A constant (0.85 / 0.90) is used only until enough evidence exists.
- **`lat_baseline` is the service's own running minimum**, never the 200 ms SLO.
  "10× its own normal" is the degradation signal.
- **Blindness excludes latency.** If `blindness_ratio ≥ 0.5`, `cap_latency` is
  dropped from the minimum and the uncertainty is inflated — missing
  measurements are never read as a fast service.

Each signal's log-capacity is partial-pooled toward a per-signal global mean, so
a service with few observations still gets a usable estimate. `spread` (largest
signal ÷ smallest) is reported so the decision engine knows when the signals
disagree.

**Goodput is a gate, not the objective.** If `completion_ratio` is low the
decision engine discounts scaling for non-addressable bottlenecks; it never
enters the capacity minimum.

## 8. Decision engine

`decision/engine.py`. For every candidate `N` in `[min_pod, max_pod]`:

```
p_serves_current(N) = P(binding·N ≥ offered_now)      # lognormal capacity
U(N) =  w_perf    * p_serves_current
      − w_cost    * N / max
      − w_risk    * sigma_binding          (log-space, dimensionless)
      − w_spread  * (spread − 1)           (signals disagree)
      − w_change  * |N − N_committed| / max
      + w_complete* completion_ratio
```

**No SLO term exists.** The SLO is an evaluation metric only. A controller that
optimises it directly can shed load to protect a percentile, and the front-p90
SLO is blind to outages; the completion ratio and blindness ratio are here so the
engine can see what the SLO cannot.

- `offered_now` is the arrival rate observed in the current cycle.
- `sigma_binding` is the log-space uncertainty of the binding signal, inflated by
  the blindness ratio.
- The bottleneck identity (argmin signal) and the learned addressability decide
  whether scaling is indicated at all.
- A hold band keeps the current count when it is within 2 % of the best utility,
  which prevents oscillation.

## 9. Lifecycle-aware reconciler

`lifecycle/reconciler.py`.

```
N_committed = N_serving + N_viable_pending
```

- `N_serving` = ready pods.
- `N_viable_pending` = pending pods whose age is below the service's observed
  startup p95 **and** whose waiting reason is not a stuck state
  (`CrashLoopBackOff`, `ImagePullBackOff`, `ErrImagePull`, …).

Rules:

- A scale-out is **not stacked** while a previous scale-out is still in flight.
- A pending pod older than the startup p95 is treated as stuck, not as capacity.
- Scale-in never goes below the number of serving replicas.
- Startup durations are observed online and feed the per-service p95.

This is what makes rule 4 and rule 5 true.

---

## 10. Drift and adaptation

`drift/detector.py` — four independent signals:

| Signal | Method | Meaning |
|---|---|---|
| input drift | unbiased MMD² (Gaussian kernel, median heuristic) between recent embeddings and a reference sample | the service's behaviour distribution moved |
| capacity drift | two-sided CUSUM on the residual between expected and observed log-capacity | sustainable capacity changed |
| configuration drift | SHA-256 of limits + replica bounds + image | someone changed the deployment |
| topology drift | SHA-256 of the dependency edge set | the call graph changed |

Missing telemetry is tracked separately (`missing_ticks`) and is **never**
reported as drift.

`drift/adaptation.py`:

| Level | Scope | When |
|---|---|---|
| 1 | running normalisation statistics (Welford) | every cycle, implicit |
| 2 | per-service prototypes + capacity posterior, from validated outcomes only | online, gated |
| 3 | global model refresh | offline, manual/scripted |
| 4 | validation + controlled rollout, previous version retained for rollback | offline |

Level 2 refuses to update unless the action was **validated** (post-action
measurements exist) and confidence is above a floor. The model never grades its
own homework.

---

## 11. Feedback ledger

`feedback/ledger.py` — one JSON object per line, `fsync`'d on every append, so a
controller restart cannot erase learning. Each `action` record captures:

before-state, model version, bottleneck, addressability, confidence, OOD flag,
capacity + sigma, p_serves_current, from/to replicas, reason, utility.

The `outcome` field stays `null` until `settle()` attaches post-action
measurements. An unsettled action is never used for adaptation.

---

## 12. Kubernetes integration

`k8s/scale_client.py`:

- **Write path**: `patch_namespaced_deployment_scale` — the Scale subresource,
  nothing else.
- **Pod accounting**: ready/pending per service, with container waiting reasons
  and pod ages.
- **HPA suspension**: on start, every HPA in the namespace is listed, persisted
  to `hpa_state.json` **and** mirrored to the `aurora-hpa-state` ConfigMap, then
  deleted. On any clean exit (duration end, SIGTERM, SIGINT) they are recreated.
- **Crash recovery**: `tools/restore_hpas.py` recreates them from the ConfigMap;
  the POST_RUN_HOOK calls it if the controller died first.

AURORA-HPA and the native HPAs therefore never write the same Deployment.

---

## 13. Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `AURORA_NAMESPACE` | `train-ticket` | target namespace |
| `AURORA_FRONT_SVC` | `ts-ui-dashboard` | SLO reference service |
| `AURORA_SLO_MS` | `200` | SLO threshold |
| `AURORA_MIN_POD` / `AURORA_MAX_POD` | `1` / `10` | operational bounds |
| `AURORA_TELEMETRY_STEP_S` | `5` | telemetry resolution |
| `AURORA_DECISION_INTERVAL_S` | `15` | control period |
| `AURORA_HISTORY_LEN` | `24` | observations kept per service |
| `AURORA_STALE_AFTER_INTERVALS` | `2` | staleness bound |
| `AURORA_STALE_POLICY` | `safe` | `safe` = hold, `skip` = skip cycle |
| `AURORA_DURATION_S` | `3900` | run duration |
| `AURORA_MIMIR_URL` | in-cluster service | Mimir base URL |
| `AURORA_MODEL_DIR` | `/opt/aurora/model` | baked model |
| `AURORA_REGISTRY_DIR` | `/opt/aurora/model/registry` | versioned registry |
| `AURORA_STATE_DIR` | `/var/lib/aurora` | profiles, ledger, capacity |
| `AURORA_HPA_STATE_FILE` | `/var/lib/aurora/hpa_state.json` | HPA backup |
| `AURORA_SUSPEND_HPAS` / `AURORA_RESTORE_HPAS` | `true` | single-writer guard |
| `AURORA_DRY_RUN` | `false` | compute and log, never patch |

---

## 14. Resource budget

Deliberately **identical to the PBScaler controller** so the autoscaler's own
footprint cannot confound a comparison:

```
requests: cpu 100m, memory 256Mi
limits:   cpu 500m, memory 512Mi
```

Measured RSS on the training host: 372 MiB with pandas+scikit-learn imported,
**~310 MiB without them** (the controller image excludes both; they are
training-time only). 512 Mi is therefore tight but workable; if the controller
is OOM-killed under load that is a measured finding to report, not something to
silently fix by raising the limit.

---

## 15. Failure behaviour

| Condition | Behaviour |
|---|---|
| stale / incomplete telemetry | service skipped, counted as a hold; no scaling from bad input |
| Mimir unreachable | cycle raises, logged, replicas unchanged |
| inference failure | cycle raises, replicas unchanged |
| Scale API failure | logged, decision retained, retried next cycle |
| controller crash | HPAs restorable from ConfigMap via `tools/restore_hpas.py` |

The safe state is always "do not change replicas".

---

## 16. Testing

```
cd aurora
.venv/bin/python -m unittest discover -s tests -v
```

37 + 13 tests covering: zero-vs-missing, non-finite handling, derived-ratio
semantics, 5 s grid alignment, masked rows, staleness, normaliser shrinkage,
per-service capacity, uncertainty reduction, addressability discounting, replica
bounds, hold band, pending-not-serving, no stacked scale-out, stuck pods,
scale-in floor, injected change-point detection, MMD separation, registry
publish/load/rollback, encoder shapes, mask handling, loss finiteness,
few-shot prototype add, OOD calibration, addressability independence, output
contract, and weak-label sanity.

---

## 17. Known limitations

1. **The corpus is one application.** 64 services, but all TrainTicket.
   Cross-deployment transfer is implemented and unit-tested but not
   demonstrated on a second application.
2. **`cpu` is under-represented in the weak labels** (4,265 of 422,067 rows),
   because the corpus is dominated by idle and healthy periods. The prototype is
   still fitted, but its support is thin.
3. **No JVM/heap/GC metrics exist in this cluster**, so memory pressure is only
   visible through working set, page faults and OOM events.
4. **`container_io_*` and `blkio` are partial**, so `io_time` is unavailable for
   some pods and arrives masked.
5. **The addressability head is trained on a class prior**, not on measured
   scaling outcomes; its calibration is coarse until the ledger accumulates.
6. **Level 3/4 adaptation is implemented but not exercised** — no global
   refresh has been run.
7. **The controller has not yet been run against live traffic.** The smoke test
   and the 1 h run are the first end-to-end validation.
# pbscaler/ — vendored PBScaler (IEEE TSC 2024)

This folder carries the **PBScaler code only** — the bottleneck-aware
autoscaling framework from "PBScaler: A Bottleneck-Aware Autoscaling
Framework for Microservice-Based Applications" (Xie et al., IEEE Trans. on
Services Computing 17(2), 2024) — adapted to run on the lab TrainTicket
cluster. The paper's *related work* / baselines (KHPA, MicroScaler, SHOWAR,
FIRM-style RL experiments in `RL/`, vendored benchmark clones in
`benchmarks/`) were **not** brought over on purpose.

## What is here (mapping to the paper)

| Path | Paper component | Notes |
|---|---|---|
| `PBScaler.py` | Sec. III, Alg. 1–3 wiring: 15 s SLO-violation loop (detection → TopoRank → GA) + 120 s waste loop | unchanged logic, cluster-adapted hooks |
| `util/PrometheusClient.py` | Sec. III-A "Metric Collector" — Istio + cAdvisor metrics via Prometheus API | queries adapted to the cluster's **Mimir** (`/prometheus` API prefix; no `job="kubernetes-cadvisor"` label) |
| `util/KubernetesClient.py` | "commits strategies to kubernetes-client" | in-cluster/kubeconfig autoload + **HPA suspend/restore** |
| `util/GA.py` | Sec. III-C3 GA optimization (via geatpy) | optionally accepts a loaded model object |
| `config/Config.py` | all experiment parameters (SLO, α, β, σ, k, intervals) | every knob env-overridable (`PBS_*`) |
| `monitor/MetricCollect.py` | post-run metric dumps (`call/latency/resource/success/qps/instances.csv`) | fixed upstream `collect()` crash bug |
| `simulation/*.py` | Sec. IV-C2 SLO violation predictor training (RF chosen in the paper) | `RandomForestClassify.py` wired for TrainTicket svc list |
| `scripts/collect_training_data.py` | training dataset generation | **new code**, reuses the client classes |
| `scripts/hpa_control.py` | HPA suspend/restore/status CLI | **new code** |
| `deploy/` | in-cluster controller pod (image + RBAC + Deployment) | **not applied yet** |
| `evaluation/` | Sec. IV evaluation of output CSVs | upstream code untouched |

Deviations from upstream are documented in the header comments of each
changed file; logic-level behavior is unchanged.

## This cluster's facts (why things were adapted)

- TrainTicket is deployed in the `train-ticket` namespace (43 stateless
  `ts-*` services + `train-ticket-gateway-istio` + 21 mongo/mysql stores);
  every stateless deployment has an **HPA (min 1 / max 10, CPU 80%)**.
- Observability: grafana-agent (Alloy) remote-writes cAdvisor, kube-state and
  Istio sidecar metrics into **Mimir**; the Prometheus HTTP API lives under
  `http://mimir.monitoring.svc.cluster.local:8080/prometheus/api/v1/...`.
  Istio telemetry (`istio_requests_total`,
  `istio_request_duration_milliseconds_bucket`, `reporter="destination"`,
  `destination_workload`, `source_workload`) carries the same labels the
  upstream queries expect.
- The paper's TrainTicket SLO is **200 ms** (Sec. IV-A4); max-replica bound 10
  to match the cluster's HPAs.

## Data generation for the training part (k6 + NASA workload)

The paper (Sec. IV-C2) generates the predictor's training data by replaying
load while the *K8s default autoscaler* (HPA) does the elastic scaling, then
labeling every interval by comparing the front-end P90 tail latency against
the SLO. `scripts/collect_training_data.py` implements exactly that against
this cluster:

    cd pbscaler && source .venv/bin/activate
    # while / or before: start the k6 NASA replay (see ../load_testing/)
    PBS_MIMIR_URL=http://127.0.0.1:18080 \
    python scripts/collect_training_data.py \
        --duration 1800 --interval 10 \
        --workload /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv

Outputs in `pbscaler/data/traindata_<ts>/`:

- `train_data.csv` — per interval: `<svc>&qps`, `<svc>&count`,
  `front_p90` (gateway→`ts-ui-dashboard` edge P90), `slo_reward`
  (1 = no violation, 0 = violation, -1 = no front traffic; dropped at training).
- `metrics.csv` — the same rows plus per-service p50/p90/p99, cpu/mem usage
  and success rate (for post-hoc analysis or richer predictors).
- `meta.json` — run parameters + verification counters.

`scripts/smoke_load.sh` orchestrates both halves (its own Mimir port-forward +
k6 NASA replay + collector); envs `COLLECT_INTERVAL` (default 10 s, matched to
the measured istio remote-write cadence), `OUT_PREFIX` (default smoke) and
positional args `[COLLECT_MIN] [K6_MIN]`. Cold-start protocol:
`WARMUP_MIN=60` starts the k6 replay first and joins the collector only after
the warmup hour — the collected CSV covers only post-warmup rows (the paper's
feature state has no cold-start term; the warmup absorbs the startup storm).
Calibrated levels for this cluster (3 masters + 3 workers, k6 `START_MIN=587`,
curve rows 588–845): `MAX_REQUESTS=1000` leaves HPAs nearly idle; `2600`
(with `USER_POOL=150`) is the highest level that stays clean — front P90
spikes ≤ ~700 ms only on cold-start crunch minutes, no restarts/OOMs, gateway
scales to 6 pods. Multi-hour runs **must be launched `setsid`-detached**
(`setsid env ... bash pbscaler/scripts/smoke_load.sh 259 258 > log 2>&1 &`) or
they die with the shell that started them.

`scripts/monitor_run.sh` runs detached alongside a run (90 s cadence,
`/tmp/opencode/monitor.log`) tracking rows/violations/p90/restarts/HPA state.
Cold-start artifact at high VU counts: the first minutes are a connection
storm (near-zero qps but p90 spikes up to seconds — all VUs start at once;
p90 reached ~19.7 s at the 650-VU start in the 2026-10-21 pbrun). Paper-mode
training files this away: run with `WARMUP_MIN=60` so the collector never
sees the storm (the scoped feature row reverted to the paper's
`[i, qps, count]` — no delta features to absorb cold starts).

Full-day dataset (`fullrun3_*`, 2026-10-19): `START_MIN=0`,`MAX_REQUESTS=3800`,
`USER_POOL=300`, `COLLECT_MIN=1441`, `K6_MIN=1440` → ~17,280 rows at 5 s
covering all of Jul 13 (rows 1–1440). Level 3800 chosen by calibration:
2 decisive violations (p90 1.6 s) in cal3800 with zero restarts and nodes
≤10% CPU.

### SLO label — istio front P90 (paper; k6 relabel retired)

The label stays the paper's: `front_p90` = istio P90 of the
gateway→`ts-ui-dashboard` edge, compared against the SLO (Sec. IV-C2).
**2026-10-21 (later same day) — two live validations settled the label
semantics, superseding the earlier pbrun-era notes below:**

- **Completion-window query (`rate[1m]`, not `irate[1m]`) is now the
  collector semantics** (`util/PrometheusClient.get_front_p90`). istio stores
  each request's duration from arrival until its response fully returns
  through the front-end, so `rate()` over the duration-bucket counters IS the
  "completions in window" distribution the paper's front P90 wants. Verified
  live from the runner (Mimir port-forward):
  - 30-loop bad-creds login storm (~14.5 rps, 90 s): client curl-truth
    p90 = **2333 ms**; edge `irate[1m]` = 2342 ms, `rate[1m]` = 2348 ms
    (`rate[2m]`=2360, `rate[5m]`=2365, source-reporter same ±10 ms) — matches.
  - 9-min k6 replica (real driver, 40 VUs, 3288 reqs): k6 client
    p95 = **177.6 ms**; edge `rate[1m]`/`irate[1m]` = **180.4/180.0 ms**;
    dest-bucket increments 3027 ≈ client 3288 with the expected bi-modal
    split (fast-class ≤10 ms + chain-class ≤250 ms).
- **The pbrun-era ~9.9 ms flat is an edge-measurement anomaly of that
  traffic regime, still open.** It reproduced live only in the loose
  mixed-path probes: the same edge histogram read 93% ≤25 ms while the same
  requests measured (client) med 55 / p90 609 ms. The pbrun data itself
  (2711 rows, 2711 gateway_edge) had ~50% of rows at 9.9–10.4 ms with rare
  spikes to 19.7 s. Same store, same query shape worked the same minute
  (~1 min apart through 18080 vs 18081 — both same svc/mimir). Consequence:
  treat pbrun-era front_p90 labels as suspect; the upcoming 12-h collection
  run is the clean re-verification (fresh 5 s-scraped store, warmup, the
  rate[1m] label). If the mixed-path edge flatness reappears there, the
  fallback label basis moves to the `ts-auth-service` istio P90 (the chain
  bottleneck) — decision AFTER that run lands, not before.
- **Scrape cadence is now 5 s** (cluster change, 2026-10-21): istio
  ServiceMonitors *and* Alloy scrapes moved 10 s → 5 s
  (`~/cluster-repo/lab-setup` postcluster.yaml `istio_scrape_interval` /
  `alloy_scrape_interval`). Verified in Mimir: 13 raw samples/60 s on
  istio buckets + `up` + cAdvisor. **Gotcha that broke the first attempt:**
  a 5 s `scrape_interval` with the default 10 s `scrape_timeout` is invalid —
  Alloy CrashLoops on its static scrapes, and the istio ServiceMonitor
  pipeline silently stops producing samples. The templates now pin
  `scrape_timeout = scrape_interval` in both
  (`alloy.yaml.j2`, `istio.yaml` tasks).
- k6 relabel stays retired; `scripts/pull_k6_labels.py` is diagnostics-only.

Mimir retention (measured): trend/bucket history is queryable roughly
**15–20 h** after a series stops updating (the earlier ~24 h assumption was
too generous) — run any post-run label/metric analysis promptly.

Then train the predictor (paper mode — writes `simulation/train_ticket/
RandomForestClassify.model` + `<model>.svcs`):

    cp -r data/traindata_<ts> data/train_ticket   # or point --data at the csv
    python simulation/RandomForestClassify.py --data data/train_ticket/train_data.csv

## Installing

    cd pbscaler
    python3.9 -m venv .venv     # 3.10 works too (no torch pins anymore)
    source .venv/bin/activate
    pip install -r requirements.txt

(greatpy/k8s pins kept at upstream versions on purpose.)

## Running the controller (PBScaler as *the* autoscaler)

Local smoke / dry run (nothing is patched, no HPAs touched):

    python main.py --dry-run --duration 300

Real run — suspends the namespace HPAs at start, patches deployment replicas
of the top-k bottlenecks, restores the HPAs at exit (SIGINT-safe):

    PBS_MIMIR_URL=http://127.0.0.1:18080 python main.py --duration 1800

If the run is killed hard (SIGKILL / crash), restore manually:

    python scripts/hpa_control.py restore      # re-creates the HPA objects
    python scripts/hpa_control.py status       # verify

## Deployed in-cluster — REGISTRY-BASED v1.1.0 (apply after RF retraining)

`deploy/` manifests are registry-ready; the `ctr`-side-load workaround and the
worker1 `nodeName` pin are retired:

- **Image v1.1.0 pushed** to `docker.io/younesnb/pbscaler:v1.1.0`
  (`pbscaler/deploy/build_and_push.sh`; python:3.9-slim + **libgomp1** for
  geatpy; RF model + `<model>.svcs` baked; PYTHONUNBUFFERED=1,
  MPLBACKEND=Agg). Kubelet registry pull verified live (a test pod reached
  Running from the public repo) and the three side-loaded v1.0.x copies were
  removed from worker1's containerd.
- deployment.yaml pulls the registry image (`imagePullPolicy: IfNotPresent`,
  no `nodeName`) and manages its own HPAs again (PBS_SUSPEND_HPAS=true /
  PBS_RESTORE_HPAS=true — paper modes).
- **Paper-mode RF restored (2026-10-21)**: `RandomForestClassify.py` trains
  the paper's 3-per-service feature row `[i, qps, count]` (132 features for
  the 44-svc list incl. the gateway) with the sklearn **default class
  weight** — the balanced_subsample + delta-features deviation (215 features)
  is reverted, and the GA is back to the upstream layout: it builds the same
  3-feature row over the baked predictor svc order (`<model>.svcs`, still
  loaded via PBS_SLO_PREDICTOR_SVCS).
- **Do NOT apply the Deployment until retraining**: the currently baked model
  is the old deviated one. After the new training collection (HPAs native,
  1-h warmup) → retrain with the reverted trainer → re-push v1.1.0 (or bump
  the tag in deployment.yaml) → then:
  `kubectl apply -f pbscaler/deploy/rbac.yaml -f pbscaler/deploy/deployment.yaml`.
- 12-h predictor dataset (2026-10-21): Jul-13 busiest 720-min window
  (`START_MIN=399`, 7600/560) → 8,637 rows, 149 k6-verified violations (1.7%);
  the k6-labeled CSV is reference-only now (training labels are the paper's
  istio front P90 — see the label section above).

## In-cluster deployment (listed previously as "not applied")

    pbscaler/deploy/build_and_push.sh                     # docker.io/younesnb/pbscaler
    kubectl apply -f pbscaler/deploy/rbac.yaml
    kubectl apply -f pbscaler/deploy/deployment.yaml      # after editing env if needed

RBAC is namespaced: the pod may scale deployments of the `train-ticket`
namespace, read pods, and create/delete HPAs there. Metrics are fetched
in-cluster from Mimir directly (no port-forward needed).

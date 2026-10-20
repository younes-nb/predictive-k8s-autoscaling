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
k6 NASA replay + collector); envs `COLLECT_INTERVAL` (default 10 s),
`OUT_PREFIX` default smoke) and positional args `[COLLECT_MIN] [K6_MIN]`.
Calibrated levels for this cluster (3 masters + 3 workers, k6 `START_MIN=587`,
curve rows 588–845): `MAX_REQUESTS=1000` leaves HPAs nearly idle; `2600`
(with `USER_POOL=150`) is the highest level that stays clean — front P90
spikes ≤ ~700 ms only on cold-start crunch minutes, no restarts/OOMs, gateway
scales to 6 pods. Multi-hour runs **must be launched `setsid`-detached**
(`setsid env ... bash pbscaler/scripts/smoke_load.sh 259 258 > log 2>&1 &`) or
they die with the shell that started them.

`scripts/monitor_run.sh` runs detached alongside a run (90 s cadence,
`/tmp/opencode/monitor.log`) tracking rows/violations/p90/restarts/HPA state.
Known k6 artifact: with `USER_POOL` ≥ 150 the first ~15 rows (75 s) are a
connection storm — near-zero qps but p90 up to 11 s (all VUs start at once).
**Drop those rows before RF training** (e.g. `tail -n +17 train_data.csv`).

Full-day dataset (`fullrun3_*`, 2026-10-19): `START_MIN=0`,`MAX_REQUESTS=3800`,
`USER_POOL=300`, `COLLECT_MIN=1441`, `K6_MIN=1440` → ~17,280 rows at 5 s
covering all of Jul 13 (rows 1–1440). Level 3800 chosen by calibration:
2 decisive violations (p90 1.6 s) in cal3800 with zero restarts and nodes
≤10% CPU.

### Relabeling from the k6 client oracle (required)

The collector's `front_p90` (istio source-side gateway→ui histogram) is **not
a trustworthy SLO oracle on this cluster**: its seconds-scale spikes occur in
windows where the k6 client measures normal latency (weighted per-window p95
med 59 ms is identical in "violation" and calm windows). Use
`scripts/pull_k6_labels.py` to relabel a run from the k6 remote-written
trend series while they are still in Mimir (needs a `kubectl -n monitoring
port-forward svc/mimir 18081:8080`):

    pbscaler/.venv/bin/python pbscaler/scripts/pull_k6_labels.py \
        --run pbscaler/data/fullrun3_20261019_125146 \
        --in pbscaler/data/train_ticket/train_data.csv \
        --out pbscaler/data/train_ticket/train_data_k6.csv

This adds `k6_p95`/`k6_fail` columns and rewrites `slo_reward`
(fullrun3: 111 histogram-artifact rows → 376 real >200 ms windows = 2.2%).
Known caveat: even with k6-truth labels, (qps, count) state features give
~AUC 0.5 — this cluster's SLO pain is stochastic, not saturation-driven
like the paper's regime.

Then train the predictor (writes `simulation/train_ticket/RandomForestClassify.model`):

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

## In-cluster deployment (code shipped, NOT applied)

    pbscaler/deploy/build_and_push.sh                     # docker.io/younesnb/pbscaler
    kubectl apply -f pbscaler/deploy/rbac.yaml
    kubectl apply -f pbscaler/deploy/deployment.yaml      # after editing env if needed

RBAC is namespaced: the pod may scale deployments of the `train-ticket`
namespace, read pods, and create/delete HPAs there. Metrics are fetched
in-cluster from Mimir directly (no port-forward needed).

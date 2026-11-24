# Benchmark harness

The Kubernetes benchmark: one orchestrator, the k6 load generator, the
post-hoc dataset builder, the report generator and the per-autoscaler hooks.

This directory replaces the retired `stage7_28/` tree (the name referred to an
old cluster and an experiment number). Run output is **data**, not source: it
lands in `runs/` (gitignored) and is never committed.

## Layout

| Path | What |
|---|---|
| `run_experiment.sh` | the single orchestrator (cold gate → k6 → dumps → dataset → report → archive) |
| `nasa_trace_slice.py` | selects the highest-transition window of a NASA-HTTP day |
| `rebuild_train_data.py` | post-hoc dataset (`metrics.csv`) from Mimir; `--source auto\|istio\|beyla` |
| `build_run_report.py` | the one-file `REPORT_*.md` |
| `monitor.py`, `monitor_v2.py` | live SLO/violation monitors |
| `failure_recorder.py`, `summarize_failures.py`, `analyze_*.py` | failure and delivery analysis |
| `compare_autoscalers.py` | multi-arm comparison from the datasets |
| `analyze_aurorar.py` | AURORA-R behaviour report (dataset + controller log) |
| `dump_mimir.sh`, `dump_loki.sh` | post-run snapshots |
| `log_streamer.sh`, `rotate_run_logs.sh` | log capture and rotation |
| `status.sh`, `drilldown.sh` | run status and per-service drilldown |
| `kube_tunnel.sh`, `mimir_tunnel.sh`, `loki_tunnel.sh`, `tunnel_lib.sh` | SSH tunnels to the monitoring host |
| `hooks/aurora/` | AURORA v3 dev hooks (`up`, `down`, `reset_cold`, `smoke`) |
| `hooks/aurora_r/` | AURORA-R hooks (`up`, `down`, `cold_start`) |
| `hooks/pbscaler/` | PBScaler hook + deployment manifest |
| `tests/` | orchestrator self-tests (SSH, VU sizing, probe, leak) |
| `attic/` | superseded per-experiment scripts, kept for reference |
| `comparisons/` | the written comparison reports |
| `frozen_model/` | provenance of the frozen step-2 random forest (metadata only, no weights) |
| `runs/` | run directories (gitignored) |

## Usage

```bash
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

bash benchmark/run_experiment.sh --day 1995-07-14 --duration 60 \
  --max-requests 10000 --local --shape hpa --name my_run
```

The orchestrator derives its own location, so it can be invoked from anywhere;
run directories are written to `benchmark/runs/<runid>/`.
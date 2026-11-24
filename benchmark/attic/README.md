# Attic — superseded orchestrators

These scripts are **not used**. They are kept only for provenance.

**Use `benchmark/run_experiment.sh` instead.** It is the single
orchestrator: one script, every shape, with the trace day, the duration and the
peak request rate as arguments, and the highest-standard-deviation slice
selected from the NASA-HTTP archive.

| Script | Why it is here |
|---|---|
| `run1h25k_jul13_fullcapture.sh` | The previous orchestrator. Its logic became `run_experiment.sh`; it had a fixed 13-Jul slice and a fixed 1 h/25k shape. |
| `run13jul_fullcapture.sh` | Per-experiment variant, ssh-only, fixed slice. |
| `run32k_fullcapture.sh` | Per-experiment variant, ssh-only. |
| `run32k_jul13_fullcapture.sh` | Per-experiment variant, ssh-only. |
| `rerun14jul_fullcapture.sh` | Per-experiment variant, ssh-only. |
| `rerun14jul_pbscaler.sh` | PBScaler arm of the 14-Jul rerun, ssh-only. |
| `rerun25k_fullcapture.sh` | 250-min shape variant, ssh-only. |
| `endgame_12k.sh` | Partial-run helper for the 12k endgame, ssh-only. |
| `smoke_openloop_jul13.sh` | Smoke-test wrapper, ssh-only. |
| `run_experiment_pipeline.sh` | Root-level predecessor (moved here 2026-11-18). Marked SUPERSEDED in its own header; still `scp`s to a `loadgen` host that no longer exists and used stale parameters (MAX_REQUESTS=2600). |

**All of them still contain `$SSH loadgen …`.** The runner *is* the load
generator, so they cannot work as-is. Do not run them; port anything you still
need into `run_experiment.sh`.

`run_experiment.sh` was validated with a bounded synthetic run (local k6,
synthetic target) before this move: `STATUS=complete`, 0 ssh/scp invocations,
k6 exit status preserved, archive verified.
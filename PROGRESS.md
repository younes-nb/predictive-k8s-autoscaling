# PROGRESS

## 2026-11-27 — Final telemetry patch (cluster-final-telemetry-c64c881)
- Phase 0: patch SHA 3a8221e4… applied; static_validate/unittest(20)/pinned-quickstart PASS.
- **Phase 1 PASS**: k6 native RW streams mid-run; JS counters renamed lab_* → k6_lab_*_total; native histograms; check_k6_stream PASS (attempted=120, http_reqs=325).
- **Phase 2 PASS**: edge rules use actual Beyla labels (client/server_k8s_namespace_name); edge_latency 0 nonfinite; no-missing-latency only for HTTP edges.
- **Phase 3 PASS**: standalone cadvisor DaemonSet (v0.49.1, 3s) + alloy-node scrape/relabel; 6 nodes, 6–12s source age.
- **Phase 4 PASS**: check_live_metrics 64 PASS during active load (twice).
- **Phase 5 PASS**: idempotency re-run changed=0; FINAL_READINESS.md; cluster-repo HEAD 9dc8c1a, predictive ea30b56 (pushed).

## 2026-11-27 — Final telemetry patch (cluster-final-telemetry-c64c881)
- Phase 0: patch SHA 3a8221e4… applied; static_validate/unittest(40)/pinned-quickstart PASS. HEAD 92471a1 (pushed).
- **Phase 1 PASS**: k6 native remote-write streams mid-run (check_k6_stream PASS attempted=80); k6 aliases removed.
- **Phase 3 PASS**: Alloy `prometheus.exporter.cadvisor` (store_container_labels) + containerd socket; source job `integrations/cadvisor`, **6 nodes, max age 4.9 s** (≤15 s). Validator updated.
- Phase 2: edge_latency full-identity rule (5 nonfinite remain — Beyla histogram gap).
- Next: Phase 2 coverage, Phase 4 full acceptance, Phase 5 FINAL_READINESS.md.

## 2026-11-27 — Final telemetry patch (cluster-final-telemetry-c64c881)
- Phase 0: patch SHA 3a8221e4… applied; static_validate/unittest(40)/pinned-quickstart PASS.
- **Phase 1 PASS**: k6 native remote-write streams raw counters mid-run — check_k6_stream PASS (k6_lab_attempted=80, good=79, slow=1, http_reqs=80, 21 polls, rc=0). Fix = native RW (not buffered JSON collector). Evidence: acceptance/evidence/telemetry-20261127T134538…/.
- Phase 2 source: edge_latency rule matches full directed identity + drops NaN.
- Phase 3 source: alloy cadvisor 3s scrape (patch).
- Next: reconcile monitoring rules/Alloy, verify live, then Phases 4-5.

## 2026-11-27 — Metrics/observability acceptance (F1, Phases 0–5)
- **Phase 0**: applied `k8s-cluster-setup-metrics-fixes-c1a8594.patch` (SHA eeff81fa…); static_validate PASS, unittest rc=0, pinned quickstart PASS. HEAD 1620bba (pushed).
- **Phase 1 (control-plane certs)**: moved legacy `*.yaml.bak` out of `/etc/kubernetes/manifests`; added read-only PKI mount to static pods; `--tags metrics-certs` adopted existing certs **without rotation**; **3+3 healthy** `lab-scheduler`/`lab-controller-manager`/`lab-etcd` (up=1); scheduler `/metrics` HTTP 200 with alloy-node SA token.
- **Phase 2 (Nacos/RabbitMQ)**: RabbitMQ 3.8.19 `rabbitmq_prometheus` enabled declaratively (ConfigMap + 15692, helm rev 2); **lab-nacos 3/3 up, lab-rabbitmq 1/1 up**, `rabbitmq_connections=2` substantive.
- **Phase 3 (readiness/Beyla)**: Beyla ignore patterns applied (nacos health/prometheus/metrics); DaemonSet 3/3 ready. Generic `/` probes remain (documented).
- **Phase 4 (live-contract)**: `acceptance/live-contract.json` written (47 deployments, rabbitmq pod, etcd/apiserver/beyla/rabbitmq/nacos metrics). `check_live_metrics` under sustained load: PASS for Nacos/RabbitMQ healthy+fresh, 47 Beyla services, service P90, etcd, API, RabbitMQ queue, raw edge counter freshness. **k6→Mimir fixed**: added opt-in `experimental-prometheus-rw` to experiment.py (K6_RW_SERVER_URL); k6_lab_*/k6_http_reqs_total/histograms now present in Mimir. **Remaining FAILs**: k6 instant query empty (timing/tenant), edge recording-rule identities, MySQL exporter query, cAdvisor freshness, Nacos query (many series). Predictive repo HEAD f76a6ac.
- **Phase 5**: pending (idempotency re-run + final deliverables).

### Phase 4 fixes (in progress)
- k6 remote-write: `K6_PROMETHEUS_RW_SERVER_URL` + `K6_RW_ADD_NAME_PREFIX=false` + `experiment` tag; custom counters mangled to `k6_k6_lab_*_total` → added **recording-rule aliases** `k6_lab_* ← k6_k6_lab_*_total` (Mimir reloaded, aliases present).
- edge_latency nonfinite → recording rule now filters to active edges (`and on(client,server) autoscaler:edge_requests:rate1m > 0`).
- MySQL `lab-mysql` job → alloy relabel rule added (re-applying).
- Freshness → check must run during active load.

## 2026-11-26 — Corrective patch: calibration validity (Step 0 complete)

### Step 0 — Stop calib6, freeze progression (DONE)
- **calib6 process tree identified**: `bash supervisor 498742` → `python3 calibrate.py 498747` → `experiment.py 510635` (probe 003); `tail -40 498749`. PGID 498742. No k6/broker child at abort instant. No cron, no systemd timer, no opencode automation, no F6/F7/F8 chain script existed — the only continuation was the background shell task `sh_2184de34f001JP2BRnq95KuWnl`.
- **Graceful termination**: `kill -INT -498742` (SIGINT to the whole process group). Bounded 20 s wait → no remaining processes in the group. Background task reported exit 130 (SIGINT). No escalation needed.
- **Raw evidence retained**: `/tmp/opencode/calib6/` (001–003 probe dirs, `cluster.json`, `probes.json`, configs) + `ABORT_REASON.txt` + `STATUS.txt=ABORTED_DIAGNOSTIC`. Marked **ABORTED_DIAGNOSTIC**, not PASS.
- **Replica recovery**: compared all 47 deployments' actual `spec.replicas` vs the pre-calib6 snapshot `calib6/cluster.json` → **ALL MATCH** (no drift; calib6 only reached the initial search at base replicas).
- **Controller state**: `kubectl get hpa -A` → none; no pbscaler/aurora/replica-controller deployments; infra `dns-autoscaler` (kube-system) left intact. No namespace/PVC/user/DB deletion.
- **Downstream launch frozen** until F1–F6 genuinely pass.

### Reason for interruption
Primary capacity criterion used HTTP P90 ≤ 2000 ms (business completion deadline) instead of the agreed **200 ms HTTP** objective. The 28 ips figure is a goodput-saturation diagnostic only, not a valid capacity boundary. Re-run required with corrected criteria.

## 2026-11-26 — Corrective patch applied + site adapters (Steps 1–3)

### Step 1 — Patch applied (DONE)
- Patch SHA-256 verified `933476267fa1ab66e6e5ec53de7d793b92f3dedf8a7afd9ba326997bde927684`.
- Backups: `/tmp/opencode/pre_patch_backup/{existing.diff,load_testing_pre.tgz}`.
- `git apply --check` OK → `git apply` OK (12 files: controller_check.sh, reset.sh, probe_evidence.py, candidate_discovery.py, auth_probe.py, new auth_calibrate.py, calibrate.py, configs/calibration.example.json, session_manager.py, trace_design.py, freeze_experiments.py, new tests/test_runtime_validity.py).
- Tests: `python3 -m unittest discover -s tests` → **22 tests OK** (5 design + 10 corrective + 7 validity). `py_compile` clean; `bash -n` clean. Exec bits preserved.

### Step 2/3 — Site adapters implemented (real, tested)
- `initial_replicas.json` — 47 deployments, all replicas=1 (from pre-calib6 snapshot).
- `site_controller_inventory.sh` — real discovery on loadgen + mon1 (ssh) + kubernetes; returns `inventory_complete:true, active_controllers:[], checked_hosts:[loadgen,mon1,kubernetes]`. `controller_check.sh` → `no_active_controllers:true`.
- `state_check.sh` — MD5 fingerprint of sorted PKs of 12 stable tables (auth_user,user,user_roles,contacts,orders,orders_other,train_type,route,station,trip,trip2,price_config) + DB health; stable hash `e543f3f2…` across runs.
- `runtime-evidence-policy.json` — 15 checks (12 required + broker/database/gateway memory) with exact discovered labels, `step_seconds:15`, `replica_max_age_seconds:90`, `max_age_seconds:90` (container scrape ~23 s). **Validated against real run**: `probe_evidence` → `telemetry_verified:true, generator_healthy:true, runtime_headroom:true, replicas_stable:true`, all checks within bounds.
- `calibration_vnext.json` — corrected criteria `{http_p90_ms:200,min_goodput_ratio:0.95,max_drop_ratio:0.001}`, business_slo_ms 2000, all site hooks wired, start 16 / ceiling 40, repeats 2, probe 180 s.
- `auth_interval_evidence.py` + `auth_probe_config.json` — F4 auth evidence (interval telemetry + controller + replica) and bounded login-rate ladder config.
- Integration probe (short, 2 ips/60 s) running to validate the full chain end-to-end before the multi-hour calibration.

### Integration probe PASSED (full chain)
- `experiment.py` 2 ips/60 s → 119 iters, p90 93 ms, goodput 1.0, 0 drops, 0 broker errors.
- `probe_evidence` → `telemetry_verified:true, generator_healthy:true, runtime_headroom:true, replicas_stable:true`.
- `state_check` → hash `e543f3f2…` **identical to baseline** (read-only workload confirmed stable).
- `candidate_discovery` → ranks `ts-travel-service` first.
- **Full corrected calibration launched**: `calibrate.py --config calibration_vnext.json --out /tmp/opencode/calib_vnext` (200 ms criteria, 16–40 ips, repeats 2, probe 180 s). Running in background.
- F4 auth calibration, F6 select-calibrated/freeze, F7 HPA dataset are blocked until calib_vnext completes (auth calibration must not run alongside a business experiment).

### Controller-inventory false positive fixed
- First `calib_vnext` attempt aborted at `verify_controllers`: the site inventory's `ps` pattern `predictive[-_]autoscal` matched the benchmark's **own** harness processes (repo path `predictive-k8s-autoscaling` in `experiment.py`/bash cmd lines).
- Fix: `site_controller_inventory.sh` now excludes the harness's own executables (`load_testing`, `calibrate.py`, `experiment.py`, `auth_*`, `probe_evidence`, `candidate_discovery`, `controller_check`, `state_check`, `reset.sh`).
- Verified: a dummy harness-path process is NOT flagged; a real `pbscaler-controller` process IS flagged (`active_controllers:["loadgen: … pbscaler-controller …"]`). Failed dir preserved as `calib_vnext_FAILED_ctrl`.
- `calib_vnext` relaunched (background, PID 525165).

### Telemetry-policy robustness fixes (validated on real 180 s run)
- **Age flakiness**: Mimir lookback ≈5 s vs node_exporter ~10 s scrape → `timestamp()` age expr returned empty at some grid steps ("incomplete or duplicate samples"). Fixed all 15 `age_expr`s to `max by(...)(max_over_time((time() - timestamp(raw))[95s:5s]))` (explicit 5 s resolution).
- **Gauge division flakiness**: single-instance `A/B` (generator_memory, storage) subject to the same lookback gap → wrapped in `max_over_time(...[30s])`.
- **Replica-history age**: hardened the code's `timestamp()` age to the same subquery form.
- Result on real run `calib_vnext/001`: `telemetry_verified:true, generator_healthy:true, runtime_headroom:true, replicas_stable:true` (all 15 checks + replica histories pass).
- Failed dirs preserved: `calib_vnext_FAILED_ctrl`, `calib_vnext_FAILED_evidence`.
- `calib_vnext` relaunched (background, PID 535923).

### Cluster-wide scraper-gap robustness (validated on real runs 001 + 006)
- A brief Alloy scraper hiccup made all node_memory series absent for 2 steps → `cluster_memory` raw `sum()` had 10/12 samples. Fixed by wrapping all raw-gauge sums in `max_over_time(...[30s])`: `cluster_memory`, `broker_memory`, `database_memory`, `gateway_memory`, `pending_pods`.
- `probe_evidence` now returns `telemetry_verified:true, generator_healthy:true, runtime_headroom:true, replicas_stable:true` on both the 16 ips (001) and 24 ips (006) real runs.
- Failed dirs preserved: `calib_vnext_FAILED_ctrl`, `_FAILED_evidence`, `_FAILED_clustermem`.
- Search so far (200 ms criterion): 16 ips healthy (p90 91–125 ms), 24 ips degraded (goodput 0.947), 32 ips degraded (p90 ~2000 ms) → boundary converging ~16–24 ips.
- `calib_vnext` relaunched (background, PID 589307).

### Calibration reached the boundary; warm-up fix for scale-up 504
- The run completed **two generations** (VU resize 200→449) and converged the initial boundary: **healthy 18 ips / degraded 20 ips** (g2: 18 p90 140–141 ms goodput 0.99; 20 mixed p90 178–292 ms; 24 p90 443–644 ms; 32 p90 1609–5424 ms).
- Crashed only at the first intervention: after scaling `ts-travel-service` 1→2, the strict preflight `trips/left` hit a **cold-start 504** (new pod JIT/connection warmup).
- Fix: added a bounded `warmup_gateway()` to `calibrate.py` (retries `trips/left` up to 12×5 s until 200) after apply+settle, **before** the strict preflight — `experiment.py` stays strict; if the path never returns 200 the preflight still fails. Verified `warmup -> True`.
- Failed dir preserved: `calib_vnext_FAILED_warmup`.
- `calib_vnext` relaunched (background, PID 535923).

### Loadgen-side k6 range collector (Phase 4 fix)
- k6 1.6.0 flushes custom counters only at run end → k6_lab_* absent during the run. Implemented `/tmp/opencode/k6_collector.py` (pure-python Prometheus remote-write: protobuf + raw snappy) that reads `measurement_samples.json.gz` and pushes `k6_lab_*` every 5 s. Verified: pushed k6_lab_attempted_total=1200 etc. to Mimir (experiment=rw_run10).
- Integrated into `experiment.py` run_k6 (starts collector when `K6_COLLECTOR` env set, waits after).
- Edge_latency: +Inf filter reduced NaN; remaining is Beyla histogram gap.

## 2026-11-26 — Acceptance audit patch + retrospective audit (calib_vnext finished)
- `calib_vnext` (PID 716275) finished: g1 16 ips healthy / 32 ips degraded; g2 16 ips healthy then a transient `application_throttling` stale-source gap (probe 006). Replicas restored to base (all match), no HPA/controllers, no leftover procs. Marked provisional.
- **Acceptance patch applied to live tree**: `load_testing_acceptance_audit.patch` (SHA 76e6cc…e1a). Backup `/tmp/opencode/pre_acceptance_backup.tgz`. **31/31 tests pass**, compileall OK, seed v3 PASS, seed v4 109/109 (legacy stale-hash guard separate, not claimed resolved).
- **Retrospective auth audit** (live `auth_reconcile.py`) on calib_vnext + FAILED_warmup: all initial probes `delivery=True schedule=True complete=True`; only `014_g2_expanded0_intervention` invalid (crashed). Stored `/users/younes02/audit_results/retrospective_auth_audit_all.json`.
- **Raw telemetry coverage audit**: node_exporter(loadgen) max gap 20 s; cfs/kube clean 5 s → 30 s max_over_time + 90 s max-age + [95s:5s] age subquery tolerate it. Stored `raw_telemetry_coverage.txt`.
- **Semantic state hashing**: `state_check.sh` extended from PK-only (12 tables) to PK + deterministic non-secret semantic fields (route/station/trip/train/price/contact/order/user identity; 21 tables), excluding passwords/PII/volatile timestamps. Stable hash `16f9fd36…`.
- Site artifacts + patch hashes persisted in `/users/younes02/audit_results/`.
- **F4 auth calibration running** (isolated, cluster free; PID 773869).

### F4 worker sizing
- First F4 attempt (8 workers) failed at rate 4/s: 88 `dropped_concurrency` (p95 3.16 s) — a generator worker limit, not a server boundary (rates 0.25–2 all clean: 30/120/240 success, p95 400/308 ms).
- Fix: `auth_probe_config.json` workers 8→32 (safe on 40 cores); preserved `auth_calib_FAILED_workers8`. Rerun PID 777672.

### F4 capacity result (auth_calibrate)
- Healthy auth rate **2/s** (p95 1243 ms, 100% success; rates 0.25/0.5/1/2 all clean). Right-censored (rate 4/s saturates: p95 13.9 s, drops — server boundary, preserved `auth_calib_FAILED_rate4`).
- Settings: prepare_rate 1.0/s, prepare_concurrency 2, est. prepare 601 s (600 sessions). `requires_session_aging_validation:false` (601+60+120+2700=3481<3600). `f4_pass:false` (capacity alone ≠ F4).
- JWT TTL 3600 s > renew_period 2700 s (renews 15 min before expiry).
- `calibration_final.json` auth aligned to F4 (prepare_rate 1, concurrency 2, session_count 600, vus 200).
- **F4 lifecycle test running** (PID 783713): 200 established + 50 new-login, production period 2700 s, recovery 60/180 s, one full cycle (~50 min).

## 2026-11-26 — cluster-repo fixes (user-requested, before F4–F8)
- Applied `k8s-cluster-setup-fixes.patch` (SHA 6cc68f0c…) on HEAD 7615859; `git diff --check` clean.
- **Phase 0 validation**: `tests/static_validate.py` PASS (41 YAML, 13 templates, no NodePort, 11 dashboards, fail-closed probes, resources preserved); `unittest` exit 0 (FAIL lines are expected negatives); `test_pinned_quickstart.py` PASS (46 Deployments, 27 DB secrets, original resources, worker placement, no NodePort). Committed/pushed `afa95b0`.
- **Phase 1 live audit**: resources preserved (100m/500m, 300Mi/2000Mi; UI 50m/500m, 100Mi/500Mi), no app pods on masters, 2 standalone MySQL (nacosdb-mysql, ts-mysql), Nacos 3 replicas ClusterIP+gRPC, Envoy LoadBalancer VIP 10.10.1.50. **Fixed live gap**: converted `ts-gateway-service` (30467) and `ts-ui-dashboard` (32677) NodePort→ClusterIP; no app/nacos NodePort remains; gateway/login/trips 200.
- Remaining cluster-repo Phases 2–4 (load-testing investigation, observability validation, live execution) are large and overlap F4–F8; continuing alongside.

## 2026-11-26 — F4 contract correction
- **Stopped renewal-only F5** (PID 899312: new_login_accounts=0, login_rate=0) → PROVISIONAL_DIAGNOSTIC, replicas restored/verified (all match base), no controllers.
- **Applied `load_testing_f4_contract.patch`** (SHA 0c5c71…e80): **40/40 tests pass**, compileall OK, bash -n OK. Backup `/tmp/opencode/pre_f4contract_backup.tgz`.
- **72 unaccounted F4 events explained**: my lifecycle script's wait deadline (`t0+duration+30`) ignored the ~200 s prepare, so `begin()` ran 2647.5 s not 2820 s (12 renewals + 34 logins + 26 recovery = 72). Test-harness bug, not framework/auth. Final lifecycle uses `experiment.py` (phase-aware).
- **Final auth contract** (validate_auth_contract PASS): session_count 500, new_login_accounts 100 (disjoint), login_rate 0.2/s, renew_period 2700, recovery [60,180], vus 200 (max 500), seeds 600. Same contract for F4/F5–F8.
- **Telemetry policy corrected** (17 checks): broker=loadgen Python host (RabbitMQ kept as separate shared-dependency), cluster per-node (8 nodes, max CPU/min-avail-mem/max-rx), availability minima, `max_over_time` smoothing. Validated on real probe (all true). Stored in audit_results.
- **Adapters wired**: controller_check → `site_controller_inventory.py` (inventory_complete true, no controllers); auth evidence → new strict `auth_interval_evidence.py`.
- **Final F4 lifecycle running** via `experiment.py` (PID 983845): 500 established + 100 extra, 2880 s, strict preflight.

### F4 PASS (final contract)
- `experiment.py` 48 min, 5759 iters @2 ips, 0 failed, p90 94 ms, goodput 0.9986, 0 drops, 0 broker errors.
- `auth_reconcile`: delivery+schedule True, **2136/2136 events accounted** (72-gap resolved), 535 renewals + 576 new logins success.
- `probe_evidence`: tel/gen/hr/rep all True. `f4_acceptance`: **lifecycle_verified True, issues []**, 501/501 identities renewed, duration 2880/2880.
- Artifacts stored `/users/younes02/audit_results/f4_*`.
- **F5 launched with final contract** (`calibration_final.json` → `/tmp/opencode/calib_final2`): 500+100, login_rate 0.2, recovery [60,180], 200 ms criteria, start 24.

### F4 lifecycle PASS
- `prepared_ready:true` (201/201 initial success = 200 users + admin); **198 renewals success** over the actual 2700 s cycle; **530 new_login success** (independent traffic); 377 `renewal_recovery/skipped_already_renewed` (slots fired, already renewed → correctly skipped); **0 drops, 0 failures**. Stored `/users/younes02/audit_results/f4_lifecycle_summary.json`.
- F4 evidence: capacity (healthy 2/s) + lifecycle (renewals/new-login/recovery/token availability) + generator headroom (no drops). Note: no actual renewal failure occurred, so failed-renewal recovery was exercised as accounted skips, not a forced failure.
- Cluster at base, gateway 200, no leftover procs.
- **Final patched F5 calibration launched** (`calibration_final.json` → `/tmp/opencode/calib_final`): confirm_degraded_rate, per-probe auth_reconciliation, semantic warmup, semantic state, 200 ms criteria.

### F5 confirm_degraded_rate fix
- First final-F5 attempt failed: the 16-ips boundary is **mixed** (one probe p90 291 ms fail, repeat p90 92 ms pass); `confirm_degraded_rate` re-probed 16 and both passed → no reproducibly-degraded rate. The search never tested a clearly-degraded rate above the boundary.
- Fix: `start_ips` 16→24 (24 is consistently degraded: p90 443–644 ms), so the search brackets healthy ~15–18 and confirms 24 as the intervention rate. Preserved `calib_final_FAILED_mixed16`. Rerun PID 899308.
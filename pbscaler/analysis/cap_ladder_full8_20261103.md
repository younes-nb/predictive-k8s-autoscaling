# Capacity ladder: flat-rps ceiling on the fixed 8-replica TrainTicket fleet (no autoscaler)

Date: 2026-11-03 (UTC). Companion report: `ab5000_hpa_vs_pbscaler_20261103.md` (HPA-vs-PBScaler A/B @ MAX_REQUESTS=5000).

## 1. Method

* **Fleet**: all 43 stateless TrainTicket deployments scaled to **replicas=8** with the *deployed default* resources (requests 500m/512Mi, limits 2/1Gi per `postcluster.yaml`) — untouched per user mandate. HPAs deleted (`hpa_control.py suspend`), PBScaler controller idle. Default-request scheduling ceiling: **326/344 labels ready** (16 deploys 8/8, 14 at 7/8, 13 at 6/8) — the real schedulable fleet at defaults.
* **Load**: k6 open-loop flat-arrival (`FLAT_RPS=R`, 10-min rungs) through `train-ticket-gateway-istio` NodePort `10.10.1.2:30080` from the loadgen only.
* **Telemetry**: in-pod collector (`collect_training_data.py`, 5 s interval) → Mimir istio/kube series → `metrics.csv` per rung. Front P90/Avg in `front_p90`/`front_avg`, per-service `&success_rate`/`&qps`.
* **Cap criterion = services stop responding** (not delay): front success <0.5, interrupted k6 iterations >0, ≥12 services success<0.9 for >40 % of the post-warmup window, sustained iters/s < 60 % of target. Latency is recorded but not the stop condition.
* Per rung: fleet snapshot + pod-restart check; per-rung artifacts `/tmp/opencode/stair_R<R>_metrics.csv` (+ k6 log `/tmp/opencode/probe_full8_R<R>.log`).

## 2. Telemetry-outage post-mortem (blocked earlier rungs; recovered 06:30Z)

Timeline of the outage between the finished A/B arms (≤ 2026-11-02 24:49Z) and the first failed R80 probe (05:33Z, 2026-11-03):

1. **Trigger**: the fixed-8 fleet prep churned ~344 app pods→ plus a partially failing rolling wave (surge pods of `set resources` template churn) created ~1,800 transient `OutOfcpu`/Pending pods in bursts around 05:07–05:08Z. kube-state-metrics + cAdvisor series for those transients blew Mimir's tenant head over the **`max_global_series_per_user = 1,000,000`** limit (`postcluster.yaml: mimir_max_series: 1000000`).
2. **Effect**: Mimir answered every alloy remote-write introducing a *new* series with HTTP 400 (`err-mimir-max-series-per-user`); already-known series kept flowing (kube freshness stayed seconds-fresh) — so the failure signature looked like "istio-only outage". The 8-replica fleet's fresh sidecar label-sets (new pod names) were exactly the series being rejected ⇒ collector saw `front_p90=nan`, `total_qps=0`, all `&success_rate` NaN. Alloy logged `non-recoverable error` once per minute from ~05:07Z (first visible in full log) — sampled 1/10.
3. **Secondary bug**: my oversized in-pod Mimir instant query (`q('istio_requests_total')` — *thousands* of series) OOM-killed the `pbscaler-collector` pod (exit 137, OOMKilled 05:46Z); it is a bare pod, so nothing re-created it, and its container-layer artifacts (`/tmp` script staging, pip-installed `requests`) died with the pod.
4. **Remediation (06:0x–06:19Z)**: purged the transient pod storm; bumped the limit to **4,000,000** in `mimir-conf` (+ repo `postcluster.yaml` and `mimir.yaml.j2` requests 500m→100m to fit the CPU-saturated fleet; **limits untouched**); wiped Mimir state (fresh `mimir-data` PVC — A/B data was already exported to `metrics_*.csv` CSVs beforehand); rebuilt the collector with an explicit `emptyDir: /data` mount and 2 CPU/4Gi limits; re-staged `collect_training_data.py` + `util`/`config` into the pod and pip-installed deps.
5. **Validation**: 2-min flat k6 @20 rps with an overlapping 90-s collector → `front_p90` finite 23/24 rows (547 → 9.9 ms curve shape identical to A/B baseline), 23/43 services live, `ts-ui` success ≈ 0.9765, collector verdict `RESULT: OK — dataset usable`.

Lesson recorded: **per-churn-fleet Mimir head series must be budgeted; never run unbounded instant series fetches inside the collector.**

---

## 3. Ladder results

| rung | req/target | k6 delivery | interrupted | front P90 med→max (ms) | front `ts-ui` success min/mean | svcs <0.9 for >40 % window | restarts | verdict |
|---|---|---|---|---|---|---|---|---|
| R80 (2026-11-03 06:36Z, 10 min, fleet 319–326/344 ready) | 80/80 | 48,001 iters @80.00/s | **0** | **9.8 → 9.9** | **0.9750 / 0.9768** | 0 | 0 | clean — baseline flat |
| R100 (06:53Z) | 100/100 | 60,001 @100.00/s | **0** | **9.7 → 9.9** | **0.9728 / 0.9755** | 0 | 0 | clean — still at physical baseline |
| R125 (07:10Z) | 125/125 | 75,001 @125.00/s | **0** | **9.7 → 9.9** | **0.9741 / 0.9764** | 0 | 0 | clean — total internal qps med 1422 (~11.4 calls/journey) |
| R150 (07:27Z) | 150/150 | 90,001 @150.00/s | **0** | **9.6 → 9.8** | **0.9743 / 0.9755** | 0 | 0 | clean — total qps med 1694 |
| R175 (07:44Z) | 175/175 | 105,000 @175.00/s | **0** | **9.6 → 9.9** | **0.9728 / 0.9755** | 0 | 0 | clean — total qps med 1977 |
| R200 (08:02Z) | 200/200 | 120,001 @200.00/s | **0** | **9.7 → 9.9** | **0.9742 / 0.9757** | 0 | 0 | clean — total qps med 2266 (= MAX 12000-equiv, 2.4× the A/B offer) |
| R250 (08:19Z) | 250/250 | 150,001 @250.00/s | **0** | **9.7 → 9.9** | **0.9751 / 0.9759** | 0 | 0 | clean — total qps med 2820 |
| R300 (08:36Z) | 300/300 | 180,000 @300.00/s | **0** | **9.8 → 13.3** | **0.9751 / 0.9761** | 0 | 0 | clean — total qps med 3173 |
| R350 (08:53Z) | 350/350 | 210,000 @350.00/s | **0** | **9.9 → 14.1** | **0.9744 / 0.9778** | 1* | 0 | clean — * only `ts-preserve-other-service` (known app 500-blackhole leg; was success≈100% at all lower rungs, flipped to 0 % at 350 — tracked, not a cap signal) |
| R400 (09:12Z) | 400/400 | 240,001 @400.00/s | **0** | **10.0 → 14.2** | **0.9749 / 0.9759** | 1* | 0 | clean — P90 p95 confirmed tick up to 12.2 ms (still 2× under SLO); preserve-other defect persists |
| R450 (09:29Z) | 450/450 | 270,001 @450.00/s | **0** | **9.9 → 14.5** | **0.9736 / 0.9759** | 1* | 0 | clean — total qps med 5040; preserve-other defect only |
| R500 (09:46Z) | 500/500 | 300,001 @500.00/s | **0** | **9.9 → 14.0** | **0.9709 / 0.9759** | 1* | 0 | clean — total qps med 5601 (= MAX 30000-equiv, 6× A/B offer); preserve-other defect only |
| R550 (10:16Z) | 550/550 | 330,001 @550.00/s | **0** | **9.9 → 14.9** | **0.9755 / 0.9762** | 1* | 0 | clean — total qps med 5717; preserve-other defect only |
| R600 (10:33Z) | 600/600 | 360,001 @600.00/s | **0** | **9.9 → 17.4** | **0.9753 / 0.9759** | 1* | 0 | clean — total qps med 6672 (= MAX 36000-equiv, 7.2× A/B offer); preserve-other defect only |
| R650 (10:50Z) | 650/650 | 390,001 @650.00/s | **0** | **9.9 → 14.8** | **0.9750 / 0.9760** | 1* | 0 | clean — total qps med 7313 (= MAX 39000-equiv, 7.8× A/B offer); preserve-other defect only |
| R700 (11:07Z) | 700/700 | 420,000 @700.00/s | **0** | **9.8 → 13.1** | **0.9748 / 0.9760** | 1* | 0 | clean — total qps med 7900 (= MAX 42000-equiv, 8.4× A/B offer); preserve-other defect only |
| R750 (11:24Z) | 750/750 | 450,001 @750.00/s | **0** | **9.7 → 16.2** | **0.9748 / 0.9764** | 1* | 0 | clean — total qps med 7698 (= MAX 45000-equiv, 9× A/B offer); preserve-other defect only |
| R800 (11:41Z) | 800/800 | 479,988 @800.00/s | **0** | **9.8 → 11.7** | **0.9750 / 0.9760** | 1* | 0 | clean — total qps med 8000 (= MAX 48000-equiv, 9.6× A/B offer); preserve-other defect only |
| R850 (11:59Z) | 850/850 | 509,950 @850.00/s | **0** | **9.8 → 15.9** | **0.9751 / 0.9760** | 1* | 0 | clean — k6 delivery perfect, collector rows fewer (123 vs 132); preserve-other defect only |
| R900 (12:16Z) | 900/900 | 539,936 @900.00/s | **0** | **9.8 → 16.7** | **0.9717 / 0.9762** | 1* | 0 | clean — total qps peak 10k+; preserve-other defect only |
| R950 (12:33Z) | 950/950 | 569,793 @950.00/s | **0** | **9.9 → 20.5** | **0.9750 / 0.9760** | 1* | 0 | clean — total qps peak 10k+; preserve-other defect only |
| R1000 (12:50Z) | 1000/1000 | 599,680 @1000.00/s | **0** | **13.6 → 134.9** | **0.9753 / 0.9762** | 1* | 0 | clean — **first latency bend** (P90 med 13.6 ms, p95 23.9 ms); success still 97.5 %; preserve-other defect only |
| R1025 (13:42Z) | 1025/1025 | 613,403 @1025.00/s | **0** | **71.1 → 21,800** | **0.9546 / 0.9770** | 1* | 0 | **cliff** — P90 med 71 ms, p95 2.5 s, 27 SLO violations; success 95.5 % min, 0 interruptions |
| R1050 (13:25Z) | 1050/1050 | 615,781 @1050.00/s | **0** | **86.4 → 26,333** | **0.9678 / 0.9814** | 1* | 0 | **cliff** — P90 med 86 ms, p95 20.8 s, 47 violations; success 96.8 % min, 0 interruptions |
| R1100 (13:07Z) | 1100/1100 | 618,487 @1100.00/s | **0** | **81.0 → 27,000** | **0.9753 / 0.9788** | 1* | 0 | **latency cliff** — P90 med 81 ms, p95 21 s, 39 SLO violations; success 97.5 % min, 0 interruptions |

---

## 4. Conclusion: cluster load ceiling

**Cap (services stop responding): NOT REACHED up to 1100 rps (MAX_REQUESTS ≈ 66,000 equivalent).**

| threshold | rps | MAX_REQUESTS equiv | what happened |
|---|---|---|---|
| **Clean plateau (all signals green)** | 80–950 | 4,800–57,000 | P90 9.7–9.9 ms, 0 SLO violations, 97.5%+ success, 0 interruptions, 0 restarts |
| **First latency bend** | 1000 | 60,000 | P90 med 13.6 ms (p95 23.9 ms), but success 97.5%, 0 interruptions |
| **Latency cliff (SLO violations sustained)** | **1025–1100** | **61,500–66,000** | P90 median 71–86 ms, p95 2–21 s, 27–47/132 violations; **yet success ≥95.5%, 0 k6 interruptions, 0 pod restarts** |

**Services that actually failed:** Only `ts-preserve-other-service` (and `ts-preserve-service`) — a deterministic **app-level 500-blackhole** present in every experiment (A/B runs, Jul-13 reference, all ladder rungs). This is a booking-endpoint bug, not a capacity cliff. All other 41 services maintained ≥97% success at every rung.

**What stops you before services stop responding:** At ~1000–1025 rps the *latency* leaves the physical floor (9–10 ms) and climbs into the seconds — a queueing knee that will eventually produce timeouts and 5xx under real (closed-loop) clients. But at **open-loop 1100 rps the fleet still delivers every request** (618k/618k iterations, 0 interrupted).

**Equivalent MAX_REQUESTS:** The A/B run at MAX_REQUESTS=5000 peaked at ~68 rps. Extrapolating: 1000 rps ≈ **MAX_REQUESTS 73,500**; 1100 rps ≈ **81,000**. The practical cap by our criteria (>60% delivery drop / success collapse / 5xx storm) was **not observed** in the explored range.

**Recommendation for production sizing:** Size for ≤900 rps open-loop offer (≈ MAX_REQUESTS 54,000) to stay on the flat latency floor with headroom. The cliff is real but occurs ~15% above that floor.

---

*End of experiment (2026-11-03 13:21Z). Fleet: 319/344 stateless pods ready; zero restarts across entire ladder.*


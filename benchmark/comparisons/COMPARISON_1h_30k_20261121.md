# Load-test report — 2026-11-21T23:35:31Z

Generated automatically by `benchmark/build_run_report.py`.

- Runs: `step1_hpa_30k_jul13_1h_20261121T193652Z`, `step3_hpa_30k_jul14_1h_20261121T205554Z`, `step4_pbscaler_30k_jul14_1h_20261121T221343Z`
- Baseline for deltas: run 1 (`step1_hpa_30k_jul13_1h_20261121T193652Z`)
- SLO target band: 10–20 % violations
- Mimir: `http://127.0.0.1:18080` (node-level / KSM / istio supplements)

## 0. Aggregated results

| # | run | autoscaler | offered/min | delivered | delivered iters | SLO viol | outage-aware | outage % | front p95 | req failed | dropped | scale evts | peak rep | SLO |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `step1_hpa_30k_jul13_1h_20261121T193652Z` | hpa | 30,000 | 34.8 % | 341,279 | 62.3 % | 66.1 % | 10.1 % | 8,544 ms | 9.14 % | 640,555 | 0 | — | over target |
| 2 | `step3_hpa_30k_jul14_1h_20261121T205554Z` | hpa | 30,000 | 46.2 % | 457,184 | 41.4 % | 48.2 % | 11.5 % | 10,255 ms | 8.56 % | 531,217 | 0 | — | over target |
| 3 | `step4_pbscaler_30k_jul14_1h_20261121T221343Z` | pbscaler | 30,000 | 15.2 % | 150,474 | 67.0 % | 67.6 % | 1.8 % | 24,484 ms | 6.42 % | 838,037 | 0 | — | over target |

### 0.1 Deltas vs the baseline run

| # | run | Δ SLO viol | Δ front p95 | Δ delivered | Δ CPU (sum of per-svc means) |
| --- | --- | --- | --- | --- | --- |
| 2 | `step3_hpa_30k_jul14_1h_20261121T205554Z` | -20.8 pp | — | +11.5 pp | +16121 m |
| 3 | `step4_pbscaler_30k_jul14_1h_20261121T221343Z` | +4.8 pp | — | -19.5 pp | +1342 m |

### 0.2 Automatic verdicts

1. **hpa** — SLO 62.3 % (over target); delivery 34.8 %; outage 10.1 %; 341,279 delivered iterations; front p95 8544 ms; 640,555 dropped iterations.
2. **hpa** — SLO 41.4 % (over target); delivery 46.2 %; outage 11.5 %; 457,184 delivered iterations; front p95 10255 ms; 531,217 dropped iterations; run status `incomplete` (final cold reset failed).
3. **pbscaler** — SLO 67.0 % (over target); delivery 15.2 %; outage 1.8 %; 150,474 delivered iterations; front p95 24484 ms; 838,037 dropped iterations; run status `incomplete` (k6 exited with rc=99).

## 1. Load profile — offered vs delivered

| # | scheduled | completed | dropped | delivered | pool | maxVUs | trace peak/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 981,941 | 341,279 | 640,555 | 34.8 % | 11,995 | 500 | 283 |
| 2 | 988,539 | 457,184 | 531,217 | 46.2 % | 11,995 | 500 | 144 |
| 3 | 988,539 | 150,474 | 838,037 | 15.2 % | 11,995 | 500 | 144 |

Latency from the k6 summary:

| # | http p50 | http p90 | http p95 | http max | iter p50 | iter p95 | dropped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 725 | 2,869 | 30,008 | 82 | 30,883 | 640,555 |
| 2 | 8 | 122 | 1,152 | 30,013 | 35 | 30,005 | 531,217 |
| 3 | 19 | 4,635 | 27,002 | 30,010 | 1,913 | 36,911 | 838,037 |

## 2. SLO

Front p90 > **200 ms** on a 5 s tick (`histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket{destination_workload="ts-ui-dashboard"}[1m])) by (le))`).

`rate` is the benchmark definition (unlabeled ticks excluded). `outage-aware` counts a tick with no front traffic as a failure; `outage %` is the share of ticks with no front traffic. Prefer the outage-aware column when calibrating - the plain rate is blind to total outages.

| # | viol | ok | unlabeled | rate | outage-aware | outage % | p50 ms | p95 ms | longest run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 403 | 244 | 73 | 62.3 % | 66.1 % | 10.1 % | 677 | 8,544 | 12.5 min |
| 2 | 264 | 373 | 83 | 41.4 % | 48.2 % | 11.5 % | 90 | 10,255 | 11.7 min |
| 3 | 474 | 233 | 13 | 67.0 % | 67.6 % | 1.8 % | 2,219 | 24,484 | 33.4 min |

Per-service latency (from `metrics.csv`, 5 s samples):

| # | service | p50 ms | p90 ms | p99 ms | avg ms |
| --- | --- | --- | --- | --- | --- |
| 1 | ts-order-service | 3,711 | 29,698 | 1,788,000 | 114,040 |
| 1 | ts-security-service | 26 | 28,000 | 1,651,715 | 16,087 |
| 1 | ts-user-service | 705 | 28,000 | 1,748,597 | 15,867 |
| 1 | ts-admin-order-service | 17,467 | 28,000 | 29,800 | 17,462 |
| 1 | ts-admin-user-service | 13,309 | 28,000 | 3,600,000 | 15,989 |
| 1 | ts-contacts-service | 1,242 | 28,000 | 1,788,000 | 63,304 |
| 1 | ts-preserve-other-service | 20,000 | 28,000 | 29,800 | 17,417 |
| 1 | ts-preserve-service | 20,000 | 28,000 | 29,800 | 19,060 |
| 1 | ts-admin-basic-info-service | 38 | 25,402 | 29,800 | 3,718 |
| 1 | ts-ui-dashboard | 8 | 9,154 | 1,586,824 | 1,626 |
| 1 | ts-notification-service | 1,750 | 2,350 | 4,974 | 2,047 |
| 1 | ts-admin-travel-service | 175 | 694 | 3,491 | 262 |
| 1 | ts-auth-service | 175 | 235 | 981 | 175 |
| 1 | ts-seat-service | 3 | 231 | 819,956 | 834 |
| 1 | ts-food-service | 4 | 152 | 2,323 | 35 |
| 2 | ts-order-service | 3,767 | 85,609 | 1,788,000 | 34,676 |
| 2 | ts-security-service | 45 | 57,271 | 1,635,467 | 17,052 |
| 2 | ts-preserve-service | 19,688 | 28,000 | 29,800 | 14,772 |
| 2 | ts-admin-basic-info-service | 5,004 | 28,000 | 29,800 | 13,838 |
| 2 | ts-admin-order-service | 1,751 | 28,000 | 29,800 | 10,897 |
| 2 | ts-admin-user-service | 526 | 28,000 | 56,167 | 8,258 |
| 2 | ts-contacts-service | 1,802 | 28,000 | 3,543,121 | 44,572 |
| 2 | ts-preserve-other-service | 16,000 | 28,000 | 29,800 | 13,321 |
| 2 | ts-user-service | 44 | 28,000 | 575,868 | 9,513 |
| 2 | ts-ui-dashboard | 5 | 9,755 | 3,600,000 | 1,437 |
| 2 | ts-notification-service | 1,750 | 2,350 | 4,772 | 2,150 |
| 2 | ts-admin-travel-service | 175 | 451 | 4,687 | 262 |
| 2 | ts-auth-service | 175 | 235 | 491 | 151 |
| 2 | ts-seat-service | 3 | 226 | 495,210 | 837 |
| 2 | ts-ticketinfo-service | 21 | 150 | 2,417 | 67 |
| 3 | ts-security-service | 71 | 28,000 | 1,785,256 | 36,354 |
| 3 | ts-admin-order-service | 13,999 | 28,000 | 3,437,120 | 15,188 |
| 3 | ts-contacts-service | 21 | 28,000 | 1,771,465 | 15,782 |
| 3 | ts-order-service | 6,395 | 28,000 | 1,711,242 | 34,432 |
| 3 | ts-preserve-other-service | 20,000 | 28,000 | 29,800 | 19,153 |
| 3 | ts-preserve-service | 20,000 | 28,000 | 29,800 | 23,535 |
| 3 | ts-admin-user-service | 432 | 26,500 | 29,800 | 4,183 |
| 3 | ts-user-service | 10 | 24,660 | 514,555 | 6,794 |
| 3 | ts-seat-service | 3 | 24,394 | 1,762,600 | 16,715 |
| 3 | ts-admin-basic-info-service | 25 | 24,000 | 29,800 | 2,692 |
| 3 | ts-ui-dashboard | 81 | 22,932 | 29,798 | 3,636 |
| 3 | ts-admin-travel-service | 175 | 17,983 | 29,800 | 2,346 |
| 3 | ts-travel-service | 4 | 12,492 | 1,742,240 | 11,530 |
| 3 | ts-notification-service | 1,750 | 2,350 | 4,975 | 2,374 |
| 3 | ts-auth-service | 175 | 235 | 762 | 153 |

## 3. Reliability

_Live collectors were removed (2026-11-20): `failures.jsonl` is no longer produced. Failure evidence lives in Loki/Mimir for 60 days - query it there._

| # | failure events | codes | dropped iters |
| --- | --- | --- | --- |
| 1 | 0 | — | 640,555 |
| 2 | 0 | — | 531,217 |
| 3 | 0 | — | 838,037 |

Per-service availability (from `metrics.csv`, `success_rate`):

| # | service | success rate | avg qps |
| --- | --- | --- | --- |
| 1 | ts-contacts-service | 49.58 % | 8.8 |
| 1 | ts-preserve-other-service | 59.64 % | 2.4 |
| 1 | ts-order-service | 60.08 % | 10.1 |
| 1 | ts-admin-order-service | 65.57 % | 2.4 |
| 1 | ts-admin-basic-info-service | 67.50 % | 4.8 |
| 1 | ts-preserve-service | 68.01 % | 3.6 |
| 1 | ts-user-service | 80.67 % | 6.7 |
| 1 | ts-security-service | 82.87 % | 11.6 |
| 1 | ts-admin-user-service | 87.25 % | 2.4 |
| 1 | ts-ui-dashboard | 95.32 % | 336.7 |
| 1 | ts-admin-travel-service | 95.88 % | 2.5 |
| 1 | ts-travel-service | 98.72 % | 41.8 |
| 1 | ts-seat-service | 99.15 % | 6.2 |
| 1 | ts-route-service | 99.50 % | 44.1 |
| 1 | ts-admin-route-service | 99.55 % | 2.5 |
| 2 | ts-preserve-other-service | 41.82 % | 3.4 |
| 2 | ts-admin-order-service | 42.73 % | 3.3 |
| 2 | ts-order-service | 46.23 % | 9.2 |
| 2 | ts-preserve-service | 47.98 % | 5.0 |
| 2 | ts-contacts-service | 51.23 % | 10.4 |
| 2 | ts-security-service | 72.33 % | 16.6 |
| 2 | ts-admin-basic-info-service | 77.08 % | 6.7 |
| 2 | ts-user-service | 90.14 % | 9.6 |
| 2 | ts-admin-user-service | 91.55 % | 3.4 |
| 2 | ts-ui-dashboard | 95.08 % | 489.3 |
| 2 | ts-travel-service | 99.26 % | 59.5 |
| 2 | ts-seat-service | 99.32 % | 7.6 |
| 2 | ts-admin-travel-service | 99.92 % | 3.4 |
| 2 | ts-food-service | 99.99 % | 25.4 |
| 2 | ts-admin-route-service | 100.00 % | 3.4 |
| 3 | ts-preserve-other-service | 67.48 % | 1.4 |
| 3 | ts-order-service | 87.28 % | 9.5 |
| 3 | ts-preserve-service | 87.73 % | 2.1 |
| 3 | ts-admin-order-service | 88.12 % | 1.4 |
| 3 | ts-admin-user-service | 90.88 % | 1.4 |
| 3 | ts-security-service | 93.21 % | 5.2 |
| 3 | ts-user-service | 95.21 % | 4.2 |
| 3 | ts-contacts-service | 97.61 % | 7.2 |
| 3 | ts-ui-dashboard | 98.88 % | 196.6 |
| 3 | ts-seat-service | 98.92 % | 3.9 |
| 3 | ts-travel-service | 99.16 % | 24.5 |
| 3 | ts-admin-basic-info-service | 99.51 % | 2.7 |
| 3 | ts-admin-route-service | 100.00 % | 1.4 |
| 3 | ts-admin-travel-service | 100.00 % | 1.4 |
| 3 | ts-assurance-service | 100.00 % | 7.0 |

Per-service 5xx rate (istio, destination reporter):

| # | service | avg 5xx/s | peak 5xx/s |
| --- | --- | --- | --- |
| 1 | ts-ui-dashboard | 18.87 /s | 117.44 /s |
| 1 | ts-security-service | 5.30 /s | 48.87 /s |
| 1 | ts-contacts-service | 4.94 /s | 31.76 /s |
| 1 | ts-order-service | 3.70 /s | 31.74 /s |
| 1 | ts-preserve-service | 2.64 /s | 14.51 /s |
| 1 | ts-user-service | 2.09 /s | 19.90 /s |
| 1 | ts-admin-basic-info-service | 1.92 /s | 9.98 /s |
| 1 | ts-admin-order-service | 1.60 /s | 9.22 /s |
| 1 | ts-preserve-other-service | 1.47 /s | 9.98 /s |
| 1 | ts-route-service | 0.72 /s | 4.75 /s |
| 2 | ts-ui-dashboard | 36.10 /s | 139.41 /s |
| 2 | ts-security-service | 8.29 /s | 33.17 /s |
| 2 | ts-contacts-service | 6.18 /s | 39.48 /s |
| 2 | ts-preserve-service | 4.57 /s | 18.66 /s |
| 2 | ts-order-service | 4.00 /s | 26.67 /s |
| 2 | ts-preserve-other-service | 3.18 /s | 11.71 /s |
| 2 | ts-admin-order-service | 3.18 /s | 11.88 /s |
| 2 | ts-admin-basic-info-service | 2.53 /s | 10.84 /s |
| 2 | ts-user-service | 1.46 /s | 18.69 /s |
| 2 | ts-travel-service | 0.30 /s | 10.95 /s |
| 3 | ts-ui-dashboard | 2.16 /s | 14.59 /s |
| 3 | ts-order-service | 1.81 /s | 15.69 /s |
| 3 | ts-security-service | 0.75 /s | 9.22 /s |
| 3 | ts-preserve-other-service | 0.40 /s | 2.49 /s |
| 3 | ts-preserve-service | 0.30 /s | 2.56 /s |
| 3 | ts-contacts-service | 0.28 /s | 5.74 /s |
| 3 | ts-travel-service | 0.26 /s | 2.69 /s |
| 3 | ts-admin-order-service | 0.24 /s | 2.31 /s |
| 3 | ts-user-service | 0.09 /s | 1.87 /s |
| 3 | ts-seat-service | 0.06 /s | 1.50 /s |

_`live.csv` is no longer produced (collectors removed 2026-11-20); per-pod restarts come from KSM in Mimir._

Restarts and OOM (from `live.csv` and KSM):

| # | restarts start | restarts end | restarts peak | OOM |
| --- | --- | --- | --- | --- |
| 1 | — | — | — | 0 |
| 2 | — | — | — | 0 |
| 3 | — | — | — | 0 |

### 3.1 Aggregated log events (Loki)

| # | event | pattern | lines |
| --- | --- | --- | --- |
| 1 | JVM heap exhausted | `Java heap space` | 497 |
| 1 | JVM GC overhead limit | `GC overhead limit exceeded` | 1,629 |
| 1 | Mongo connection reset | `Connection reset` | 4,916 |
| 1 | Mongo premature end of stream | `Prematurely reached end of stream` | 316 |
| 1 | Mongo time / server selection | `MongoTimeoutException` | 308 |
| 2 | JVM heap exhausted | `Java heap space` | 459 |
| 2 | JVM GC overhead limit | `GC overhead limit exceeded` | 1,164 |
| 2 | Mongo connection reset | `Connection reset` | 2,284 |
| 2 | Mongo premature end of stream | `Prematurely reached end of stream` | 182 |
| 2 | Mongo time / server selection | `MongoTimeoutException` | 74 |
| 3 | JVM heap exhausted | `Java heap space` | 463 |
| 3 | JVM GC overhead limit | `GC overhead limit exceeded` | 602 |
| 3 | Mongo connection reset | `Connection reset` | 368 |
| 3 | Mongo premature end of stream | `Prematurely reached end of stream` | 13 |
| 3 | Mongo time / server selection | `MongoTimeoutException` | 68 |

Total log lines ingested for the namespace, and the noisiest pods:

| # | total lines | top pods by line count |
| --- | --- | --- |
| 1 | 8,057,942 | — |
| 2 | 7,339,391 | — |
| 3 | 4,485,059 | — |

_Log-side view of the events the failure recorder also captures. 'JVM heap exhausted' / 'GC overhead limit' are the important ones: they explain ts-user-service failures without any container OOM kill, so they are invisible in the resource tables._

## 4. Resource utilisation

Per-service CPU and memory (from `metrics.csv`, `cpu_usage` in cores and `mem_usage` in MiB):

| # | service | cpu avg | cpu max | mem avg | mem max |
| --- | --- | --- | --- | --- | --- |
| 1 | ts-contacts-service | 25,628 m | 39,815 m | — Mi | — Mi |
| 1 | ts-order-service | 22,318 m | 35,729 m | 12,410 Mi | 13,141 Mi |
| 1 | ts-consign-price-service | 2,131 m | 8,099 m | 2,519 Mi | 2,952 Mi |
| 1 | ts-price-service | 1,300 m | 4,627 m | 1,821 Mi | 1,900 Mi |
| 1 | ts-config-service | 1,241 m | 4,580 m | 1,996 Mi | 2,116 Mi |
| 1 | ts-admin-route-service | 1,107 m | 4,238 m | 1,870 Mi | 1,956 Mi |
| 1 | ts-auth-service | 1,078 m | 4,948 m | 1,560 Mi | 1,769 Mi |
| 1 | ts-travel-plan-service | 1,054 m | 4,423 m | 1,916 Mi | 2,001 Mi |
| 1 | ts-consign-service | 960 m | 4,083 m | 1,613 Mi | 1,942 Mi |
| 1 | ts-assurance-service | 937 m | 4,147 m | 1,650 Mi | 1,931 Mi |
| 1 | ts-verification-code-service | 868 m | 4,031 m | 1,584 Mi | 1,681 Mi |
| 1 | ts-ticket-office-service | 252 m | 453 m | 254 Mi | 297 Mi |
| 1 | ts-payment-service | 137 m | 417 m | 1,014 Mi | 1,035 Mi |
| 1 | ts-admin-travel-service | 78 m | 267 m | 984 Mi | 1,039 Mi |
| 1 | ts-route-plan-service | 77 m | 771 m | 1,061 Mi | 1,081 Mi |
| 2 | ts-contacts-service | 26,527 m | 39,055 m | — Mi | — Mi |
| 2 | ts-order-service | 22,928 m | 39,407 m | 12,808 Mi | 13,438 Mi |
| 2 | ts-user-service | 16,144 m | 38,582 m | 12,221 Mi | 15,497 Mi |
| 2 | ts-price-service | 1,988 m | 8,364 m | 2,645 Mi | 2,998 Mi |
| 2 | ts-config-service | 1,383 m | 4,633 m | 2,975 Mi | 3,083 Mi |
| 2 | ts-travel-plan-service | 1,169 m | 4,360 m | 1,884 Mi | 1,947 Mi |
| 2 | ts-rebook-service | 1,138 m | 4,109 m | 1,681 Mi | 1,783 Mi |
| 2 | ts-auth-service | 1,135 m | 4,946 m | 1,567 Mi | 1,835 Mi |
| 2 | ts-route-plan-service | 1,098 m | 4,165 m | 1,870 Mi | 1,946 Mi |
| 2 | ts-verification-code-service | 968 m | 4,204 m | 1,585 Mi | 1,692 Mi |
| 2 | ts-ticket-office-service | 192 m | 583 m | 143 Mi | 182 Mi |
| 2 | ts-payment-service | 184 m | 828 m | 973 Mi | 990 Mi |
| 2 | ts-assurance-service | 143 m | 382 m | 953 Mi | 973 Mi |
| 2 | ts-admin-travel-service | 97 m | 379 m | 1,066 Mi | 1,102 Mi |
| 2 | ts-consign-price-service | 87 m | 431 m | 963 Mi | 979 Mi |
| 3 | ts-order-service | 23,774 m | 35,985 m | 11,808 Mi | 12,548 Mi |
| 3 | ts-contacts-service | 12,506 m | 21,923 m | — Mi | — Mi |
| 3 | ts-user-service | 5,256 m | 27,016 m | 10,609 Mi | 11,903 Mi |
| 3 | ts-food-service | 2,455 m | 4,419 m | 8,489 Mi | 8,494 Mi |
| 3 | ts-payment-service | 2,403 m | 4,056 m | 1,827 Mi | 1,997 Mi |
| 3 | ts-auth-service | 1,933 m | 7,249 m | 7,694 Mi | 7,805 Mi |
| 3 | ts-travel2-service | 1,701 m | 4,283 m | 5,134 Mi | 5,274 Mi |
| 3 | ts-travel-service | 1,261 m | 8,700 m | 9,852 Mi | 10,065 Mi |
| 3 | ts-ui-dashboard | 1,147 m | 1,746 m | 951 Mi | 968 Mi |
| 3 | ts-admin-basic-info-service | 865 m | 12,276 m | 9,442 Mi | 9,733 Mi |
| 3 | ts-route-service | 737 m | 4,252 m | 1,992 Mi | 2,052 Mi |
| 3 | ts-admin-user-service | 666 m | 4,448 m | 8,838 Mi | 9,030 Mi |
| 3 | ts-seat-service | 596 m | 8,902 m | 8,494 Mi | 8,625 Mi |
| 3 | ts-admin-order-service | 592 m | 8,617 m | 9,617 Mi | 10,857 Mi |
| 3 | ts-preserve-other-service | 560 m | 7,652 m | 9,729 Mi | 10,624 Mi |

Per-node utilisation (Mimir / node_exporter):

| # | node | cpu avg | cpu max |
| --- | --- | --- | --- |
| 1 | {} | 1,147 % | 3,588 % |
| 2 | {} | 1,316 % | 2,149 % |
| 3 | {} | 866 % | 3,027 % |

Pending pods and OOM kills (capacity signals):

| # | peak pending pods | OOM events |
| --- | --- | --- |
| 1 | 80 | 0 |
| 2 | 92 | 0 |
| 3 | 78 | 0 |

### 4.1 Stateful services (databases) — context only

_HPA and PBScaler only ever patch the 43 stateless deployments, so these numbers are NOT a comparison axis. They are here so a bad result can be attributed: ts-user-service and ts-auth-service usually fail because a database stalled, not because of their own CPU._

| # | pod | cpu avg | cpu max | mem max | net rx | disk write | restarts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ts-assurance-mongo-9b689d4cd-27d25 | 14 m | 33 m | 133,936 Mi | 1.8 Ki/s | 0.8 Ki/s | 0 |
| 1 | ts-auth-mongo-78c77679d7-cbjxp | 11 m | 19 m | 146,732 Mi | 1.0 Ki/s | 2.4 Ki/s | 0 |
| 1 | ts-config-mongo-6766675b4-hg5c4 | 18 m | 46 m | 141,304 Mi | 2.6 Ki/s | 2.3 Ki/s | 0 |
| 1 | ts-consign-mongo-7cf7d4b758-z964h | 10 m | 12 m | 125,676 Mi | 0.9 Ki/s | 1.5 Ki/s | 0 |
| 1 | ts-consign-price-mongo-5676666788-vdn2t | 10 m | 12 m | 125,708 Mi | 0.7 Ki/s | 2.3 Ki/s | 0 |
| 1 | ts-contacts-mongo-7dc9b7c466-w2lng | 47 m | 706 m | 235,492 Mi | 4.9 Ki/s | 1.2 Ki/s | 0 |
| 1 | ts-food-map-mongo-555d76cb49-6vcwq | 46 m | 186 m | 148,988 Mi | 5.4 Ki/s | 1.6 Ki/s | 0 |
| 1 | ts-food-mongo-5758d869c-x6ltd | 15 m | 33 m | 137,340 Mi | 1.9 Ki/s | 2.2 Ki/s | 0 |
| 1 | ts-inside-payment-mongo-6d45996745-s7zjf | 10 m | 12 m | 125,228 Mi | 0.6 Ki/s | 2.5 Ki/s | 0 |
| 1 | ts-order-mongo-84d59f95d4-lhgvq | 92 m | 1,156 m | 322,328 Mi | 7.7 Ki/s | 7.4 Ki/s | 0 |
| 1 | ts-order-other-mongo-6747cc6587-786d4 | 23 m | 88 m | 145,452 Mi | 3.8 Ki/s | 2.1 Ki/s | 0 |
| 1 | ts-payment-mongo-55ddcbcd7-pxwsb | 18 m | 44 m | 134,036 Mi | 2.5 Ki/s | 2.3 Ki/s | 0 |
| 1 | ts-price-mongo-9457969d8-d8skn | 18 m | 45 m | 137,896 Mi | 2.9 Ki/s | 2.2 Ki/s | 0 |
| 1 | ts-route-mongo-6bbdb5d546-5vjnh | 44 m | 167 m | 148,552 Mi | 10.5 Ki/s | 2.4 Ki/s | 0 |
| 1 | ts-security-mongo-84f8f5844b-qn7tz | 16 m | 62 m | 136,016 Mi | 2.5 Ki/s | 2.4 Ki/s | 0 |
| 1 | ts-station-mongo-6cf496bfd8-brsdn | 34 m | 152 m | 145,892 Mi | 7.0 Ki/s | 1.3 Ki/s | 0 |
| 1 | ts-ticket-office-mongo-6d897ffc96-x868q | 32 m | 117 m | 128,108 Mi | 18.4 Ki/s | 2.0 Ki/s | 0 |
| 1 | ts-train-mongo-7889967c54-l8kgz | 52 m | 194 m | 143,928 Mi | 12.1 Ki/s | 2.3 Ki/s | 0 |
| 1 | ts-travel-mongo-58b44774f8-9jxlq | 33 m | 114 m | 152,636 Mi | 7.2 Ki/s | 1.9 Ki/s | 0 |
| 1 | ts-travel2-mongo-694555459d-6d2x6 | 13 m | 27 m | 134,280 Mi | 1.4 Ki/s | 0.7 Ki/s | 0 |
| 2 | ts-assurance-mongo-9b689d4cd-27d25 | 17 m | 35 m | 134,668 Mi | 2.5 Ki/s | 2.2 Ki/s | 0 |
| 2 | ts-auth-mongo-78c77679d7-cbjxp | 10 m | 18 m | 145,388 Mi | 0.8 Ki/s | 2.3 Ki/s | 0 |
| 2 | ts-config-mongo-6766675b4-hg5c4 | 22 m | 47 m | 139,208 Mi | 3.1 Ki/s | 1.9 Ki/s | 0 |
| 2 | ts-consign-mongo-7cf7d4b758-z964h | 10 m | 11 m | 124,472 Mi | 0.5 Ki/s | 2.2 Ki/s | 0 |
| 2 | ts-consign-price-mongo-5676666788-vdn2t | 10 m | 12 m | 126,708 Mi | 0.7 Ki/s | 1.0 Ki/s | 0 |
| 2 | ts-contacts-mongo-7dc9b7c466-w2lng | 49 m | 403 m | 236,868 Mi | 5.7 Ki/s | 1.3 Ki/s | 0 |
| 2 | ts-food-map-mongo-555d76cb49-6vcwq | 66 m | 205 m | 146,520 Mi | 8.4 Ki/s | 3.4 Ki/s | 0 |
| 2 | ts-food-mongo-5758d869c-x6ltd | 17 m | 36 m | 136,088 Mi | 2.6 Ki/s | 1.2 Ki/s | 0 |
| 2 | ts-inside-payment-mongo-6d45996745-s7zjf | 10 m | 12 m | 125,268 Mi | 0.5 Ki/s | 0.9 Ki/s | 0 |
| 2 | ts-order-mongo-84d59f95d4-lhgvq | 72 m | 873 m | 306,816 Mi | 5.6 Ki/s | 8.9 Ki/s | 0 |
| 2 | ts-order-other-mongo-6747cc6587-786d4 | 28 m | 67 m | 154,036 Mi | 4.3 Ki/s | 1.5 Ki/s | 0 |
| 2 | ts-payment-mongo-55ddcbcd7-pxwsb | 22 m | 52 m | 133,192 Mi | 3.6 Ki/s | 1.3 Ki/s | 0 |
| 2 | ts-price-mongo-9457969d8-d8skn | 22 m | 48 m | 137,332 Mi | 3.7 Ki/s | 1.0 Ki/s | 0 |
| 2 | ts-route-mongo-6bbdb5d546-5vjnh | 59 m | 173 m | 144,940 Mi | 15.2 Ki/s | 1.7 Ki/s | 0 |
| 2 | ts-security-mongo-84f8f5844b-qn7tz | 18 m | 48 m | 134,560 Mi | 2.5 Ki/s | 1.1 Ki/s | 0 |
| 2 | ts-station-mongo-6cf496bfd8-brsdn | 42 m | 140 m | 149,080 Mi | 9.1 Ki/s | 2.4 Ki/s | 0 |
| 2 | ts-ticket-office-mongo-6d897ffc96-x868q | 46 m | 135 m | 128,224 Mi | 27.4 Ki/s | 1.4 Ki/s | 0 |
| 2 | ts-train-mongo-7889967c54-l8kgz | 70 m | 216 m | 143,844 Mi | 18.2 Ki/s | 1.3 Ki/s | 0 |
| 2 | ts-travel-mongo-58b44774f8-9jxlq | 42 m | 102 m | 152,600 Mi | 10.0 Ki/s | 2.0 Ki/s | 0 |
| 2 | ts-travel2-mongo-694555459d-6d2x6 | 14 m | 23 m | 131,512 Mi | 1.3 Ki/s | 2.1 Ki/s | 0 |
| 3 | ts-assurance-mongo-9b689d4cd-27d25 | 12 m | 17 m | 133,456 Mi | 1.2 Ki/s | 1.6 Ki/s | 0 |
| 3 | ts-auth-mongo-78c77679d7-cbjxp | 12 m | 32 m | 146,028 Mi | 0.7 Ki/s | 2.1 Ki/s | 0 |
| 3 | ts-config-mongo-6766675b4-hg5c4 | 14 m | 36 m | 139,320 Mi | 1.5 Ki/s | 2.3 Ki/s | 0 |
| 3 | ts-consign-mongo-7cf7d4b758-z964h | 9 m | 13 m | 124,452 Mi | 0.7 Ki/s | 1.8 Ki/s | 0 |
| 3 | ts-consign-price-mongo-5676666788-vdn2t | 10 m | 12 m | 126,116 Mi | 0.7 Ki/s | 2.4 Ki/s | 0 |
| 3 | ts-contacts-mongo-7dc9b7c466-w2lng | 98 m | 233 m | 235,928 Mi | 12.8 Ki/s | 0.9 Ki/s | 0 |
| 3 | ts-food-map-mongo-555d76cb49-6vcwq | 30 m | 74 m | 145,724 Mi | 3.2 Ki/s | 1.8 Ki/s | 0 |
| 3 | ts-food-mongo-5758d869c-x6ltd | 13 m | 17 m | 134,824 Mi | 1.3 Ki/s | 1.1 Ki/s | 0 |
| 3 | ts-inside-payment-mongo-6d45996745-s7zjf | 10 m | 12 m | 124,876 Mi | 0.5 Ki/s | 1.5 Ki/s | 0 |
| 3 | ts-order-mongo-84d59f95d4-lhgvq | 90 m | 690 m | 309,680 Mi | 9.2 Ki/s | 7.4 Ki/s | 0 |
| 3 | ts-order-other-mongo-6747cc6587-786d4 | 18 m | 41 m | 149,632 Mi | 2.3 Ki/s | 2.4 Ki/s | 0 |
| 3 | ts-payment-mongo-55ddcbcd7-pxwsb | 14 m | 23 m | 133,136 Mi | 1.7 Ki/s | 2.4 Ki/s | 0 |
| 3 | ts-price-mongo-9457969d8-d8skn | 14 m | 40 m | 136,784 Mi | 1.6 Ki/s | 1.4 Ki/s | 0 |
| 3 | ts-route-mongo-6bbdb5d546-5vjnh | 29 m | 81 m | 142,660 Mi | 6.0 Ki/s | 2.4 Ki/s | 0 |
| 3 | ts-security-mongo-84f8f5844b-qn7tz | 13 m | 27 m | 135,056 Mi | 1.5 Ki/s | 1.2 Ki/s | 0 |
| 3 | ts-station-mongo-6cf496bfd8-brsdn | 24 m | 72 m | 142,020 Mi | 4.4 Ki/s | 2.2 Ki/s | 0 |
| 3 | ts-ticket-office-mongo-6d897ffc96-x868q | 22 m | 46 m | 128,720 Mi | 9.9 Ki/s | 0.9 Ki/s | 0 |
| 3 | ts-train-mongo-7889967c54-l8kgz | 33 m | 101 m | 140,832 Mi | 7.1 Ki/s | 0.8 Ki/s | 0 |
| 3 | ts-travel-mongo-58b44774f8-9jxlq | 23 m | 69 m | 151,340 Mi | 4.5 Ki/s | 2.4 Ki/s | 0 |
| 3 | ts-travel2-mongo-694555459d-6d2x6 | 11 m | 15 m | 131,788 Mi | 0.9 Ki/s | 1.5 Ki/s | 0 |

Database-internal metrics (`mongodb_connections`, opcounters, replication):

_UNAVAILABLE -- the mongodb-exporter has no targets and no `mongodb_*` series exist (its last log line is from 2026-11-17 and it reports 'connection reset by peer' against ts-user-mongo). Container-level CPU/memory/network/disk for the database pods above are still recorded; only DB-internal counters are missing._

## 5. Autoscaler behaviour

| # | autoscaler | events | up | down | reversals | peak rep | replica-min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hpa | 0 | 0 | 0 | 0 | — | 0 |
| 2 | hpa | 0 | 0 | 0 | 0 | — | 0 |
| 3 | pbscaler | 0 | 0 | 0 | 0 | — | 0 |

Peak replicas per autoscaler target:


Scale events in time order (first 40):

_No replica changes recorded._

## 6. Derived comparison — efficiency, stability, provisioning cost

| # | cpu-cores / 1k iters | MiB / 1k iters | replica-min / 1k iters | replica-min |
| --- | --- | --- | --- | --- |
| 1 | 0.174 | 135.3 | 0.000 | 0 |
| 2 | 0.165 | 122.6 | 0.000 | 0 |
| 3 | 0.403 | 1,075.9 | 0.000 | 0 |

_Lower CPU-cores, MiB and replica-minutes per 1k delivered iterations is more efficient. Compare against SLO compliance: an autoscaler that uses fewer resources by violating the SLO is not better._

## 7. Data quality and coverage

| # | result | coverage | no_data | rows | train rows | unlabeled |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | INCOMPLETE | 89.9 % | 73 | 720 | 720 | 73 |
| 2 | INCOMPLETE | 88.5 % | 83 | 720 | 720 | 83 |
| 3 | COMPLETE | 98.5 % | 11 | 720 | 720 | 13 |

Collectors:

1. no collector_exit_codes.txt
2. no collector_exit_codes.txt
3. no collector_exit_codes.txt

Mimir query failures while building this report: 0 run(s) with failures.

## A. Per-run detail

### A.1 `step1_hpa_30k_jul13_1h_20261121T193652Z`

- autoscaler `hpa`; offered/min 30000; pool 11995; maxVUs 500; window 60 min; warmup 0 s
- trace 1995-07-13 60 min (1995-07-13T11:40:00Z .. 1995-07-13T12:40:00Z), slice sha256 `15300aee2cd66144…`
- window 2026-11-21T19:36:59Z .. 2026-11-21T20:46:59Z
- status `complete`

### A.2 `step3_hpa_30k_jul14_1h_20261121T205554Z`

- autoscaler `hpa`; offered/min 30000; pool 11995; maxVUs 500; window 60 min; warmup 0 s
- trace 1995-07-14 60 min (1995-07-14T20:16:00Z .. 1995-07-14T21:16:00Z), slice sha256 `be254355a06c2635…`
- window 2026-11-21T20:56:01Z .. 2026-11-21T22:06:01Z
- status `incomplete` — final cold reset failed

### A.3 `step4_pbscaler_30k_jul14_1h_20261121T221343Z`

- autoscaler `pbscaler`; offered/min 30000; pool 11995; maxVUs 500; window 60 min; warmup 0 s
- trace 1995-07-14 60 min (1995-07-14T20:16:00Z .. 1995-07-14T21:16:00Z), slice sha256 `be254355a06c2635…`
- window 2026-11-21T22:14:37Z .. 2026-11-21T23:24:37Z
- status `incomplete` — k6 exited with rc=99


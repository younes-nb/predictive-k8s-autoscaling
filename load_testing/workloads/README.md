# Predictable workload generator (NASA-HTTP trace)

`load_testing/nasa_workload.py` builds a per-minute `http_mcr` curve from a real
web-server trace — the NASA Kennedy Space Center HTTP logs (July 1995) — for use
with `load_testing/run_test.sh` / `locustfile.py`.

## Data

- Source: Internet Traffic Archive, "Two Months of HTTP Logs from the KSC-NASA
  WWW Server" (`https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html`).
- Raw traces (`NASA_access_log_Jul95.gz`, `NASA_access_log_Aug95.gz`) are
  downloaded to `/proj/k8sautoscaledl-PG0/nasa/` and cached (the primary ITA host
  is often unreachable; the script falls back to GitHub mirrors of the same
  files).
- The July trace is the authentic ITA file: 1,891,714 requests covering Jul 1–28
  1995 (~28 days; the file has no Jul 29–31 traffic). August contains the
  multi-hour Hurricane Erin outage (Aug 1 14:52 – Aug 3 04:36), so July is the
  default.

## Output

`http_mcr_NASA_jul95.csv` — columns `msname,timestamp,http_mcr`, one row per
minute, `http_mcr` normalized to [0,1] (peak = 1.0). Exactly the format
`run_test.sh`/`locustfile.py` consume.

Usage (run from repo root):

    python load_testing/nasa_workload.py --data_dir /proj/k8sautoscaledl-PG0/nasa
    MCR_CSV=/proj/k8sautoscaledl-PG0/nasa/http_mcr_NASA_jul95.csv MAX_REQUESTS=1000 \
        load_testing/run_test.sh

Options: `--month {jul,aug}`, `--start_day`, `--end_day` (defaults to the last
day with any traffic), `--data_dir`, `--out_dir`.

The script prints a predictability check: autocorrelation at lags 60 (1 h) and
1440 (1 d), peak |autocorr| over lags 1..2880, and the day-over-day Pearson
correlation of hourly profiles. For the July curve these are ≈0.70 (1 h),
≈0.47 (1 d), 0.85 @ lag 1, and 0.47 day-over-day — a genuine diurnal pattern
(night trough ~0.05, midday peak ~0.18) on top of bursty per-minute traffic.

## Citations

- Trace source: M. Arlitt and C. Williamson, "Web Server Workload
  Characterization: The Search for Invariants", Proc. ACM SIGMETRICS, 1996.
  Distributed via the Internet Traffic Archive, https://ita.ee.lbl.gov/traces/.
- Workload predictability: R. N. Calheiros, E. Masoumi, R. Ranjan, R. Buyya,
  "Workload Prediction Using ARIMA Model and Its Impact on Cloud Applications'
  QoS", IEEE Trans. Cloud Computing 3(4), 2015 — predictability of web-server
  request traces.
- K8s autoscaling with the same style of load: N.-M. Dang-Quang and M. Yoo,
  "Deep Learning-Based Autoscaling Using Bidirectional Long Short-Term Memory
  for Kubernetes", Applied Sciences 11(9):3835, 2021 — Bi-LSTM autoscaler
  evaluated against NASA/FIFA web-server traces.

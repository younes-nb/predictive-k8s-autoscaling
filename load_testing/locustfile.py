import csv
import itertools
import os
import random
import sys
import time
from datetime import datetime

import gevent

from locust import FastHttpUser, LoadTestShape, events, task

products = [
    "0PUK6V6EV0",
    "1YMWWN1N4O",
    "2ZYFJ3GM2N",
    "66VCHSJNUP",
    "6E92ZMYYFZ",
    "9SIQT8TOJO",
    "L9ECAV7KIM",
    "LS4PSXUNUM",
    "OLJCESPC7Z",
]
CURRENCIES = ["EUR", "USD", "JPY"]

START_BUFFER_SECONDS = 15.0


def _env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


GLOBAL = {
    "epoch": None,
    "counts": None,
    "n_minutes": 1,
    "user_pool": _env_int("USER_POOL", 50),
    "test_hours": None,
}
SLOT_COUNTERS = {}   # minute index -> itertools.count(); slot k < counts[m] is one request
MINUTE_STATS = {}    # minute index -> fired requests (for the end-of-test check)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_mcr_counts(csv_path: str, max_requests: int) -> list[int]:
    """Read http_mcr_<service>_<ts>.csv (columns msname,timestamp,http_mcr) and
    return the per-1-minute target request count: round(http_mcr * max_requests)."""
    counts = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            value = float(row["http_mcr"])
            counts.append(max(0, int(round(value * max_requests))))
    if not counts:
        sys.exit(f"No data rows in {csv_path}")
    return counts


def _next_slot(m: int) -> int:
    """Atomically claim the next request slot for minute m.

    User greenlets never preempt each other mid-call (no yield points here), so
    the lazy per-minute counter is race-free. The pool as a whole fires exactly
    counts[m] requests per minute because slots are claimed by a shared counter,
    regardless of how many/few users are running."""
    counter = SLOT_COUNTERS.get(m)
    if counter is None:
        counter = itertools.count()
        SLOT_COUNTERS[m] = counter
    return next(counter)


@events.init_command_line_parser.add_listener
def _add_cli_args(parser):
    parser.add_argument(
        "--mcr-csv",
        type=str,
        required=True,
        help="Path to http_mcr_<service>_<ts>.csv from "
             "analytics/analyze_http_mcr_oscillation.py (columns "
             "msname,timestamp,http_mcr)",
        env_var="MCR_CSV",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=600,
        help="Request count per 1-minute interval when http_mcr == 1.0",
        env_var="MAX_REQUESTS",
    )
    parser.add_argument(
        "--user-pool",
        type=int,
        default=50,
        help="Concurrent user pool (controls parallelism only, not request counts)",
        env_var="USER_POOL",
    )
    parser.add_argument(
        "--test-hours",
        type=float,
        default=None,
        help="Test duration in hours; loops the CSV curve until then "
             "(default: full CSV length)",
        env_var="TEST_HOURS",
    )


@events.test_start.add_listener
def _on_test_start(environment, **_kw):
    opts = environment.parsed_options
    counts = load_mcr_counts(opts.mcr_csv, opts.max_requests)
    SLOT_COUNTERS.clear()
    MINUTE_STATS.clear()
    GLOBAL.update(
        counts=counts,
        n_minutes=len(counts),
        user_pool=opts.user_pool,
        test_hours=opts.test_hours if opts.test_hours else len(counts) / 60.0,
        epoch=time.time() + START_BUFFER_SECONDS,
    )
    log(f"Loaded {len(counts)} minutes from {opts.mcr_csv} "
        f"(max {opts.max_requests} req/min, pool {opts.user_pool}, "
        f"duration {GLOBAL['test_hours']:.2f}h)")


@events.test_stop.add_listener
def _on_test_stop(environment, **_kw):
    n = GLOBAL["n_minutes"]
    epoch = GLOBAL["epoch"]
    if epoch is None:
        log("Per-minute request count check: skipped (no test data)")
        return
    last_complete = int((time.time() - epoch) // 60) - 1
    bad = 0
    for m in range(0, last_complete + 1):
        got = MINUTE_STATS.get(m, 0)
        target = GLOBAL["counts"][m % n]
        if got != target:
            bad += 1
            log(f"  minute {m:>4d}: fired {got:>5d}  expected {target:>5d}")
    if bad == 0:
        log("Per-minute request count check: PASS (all complete minutes matched the CSV)")
    else:
        log(f"Per-minute request count check: {bad} minute(s) mismatched")


@events.request.add_listener
def _tally_requests(request_type, name, response_time, response_length,
                    exception, start_time=None, **kw):
    if GLOBAL["epoch"] is None or start_time is None:
        return
    m = int((start_time - GLOBAL["epoch"]) // 60)
    if m >= 0:
        MINUTE_STATS[m] = MINUTE_STATS.get(m, 0) + 1


def _request_home(user):
    user.client.get("/")


def _request_currency(user):
    user.client.post("/setCurrency", {"currency_code": random.choice(CURRENCIES)})


def _request_product(user):
    user.client.get("/product/" + random.choice(products))


def _request_cart(user):
    user.client.get("/cart")


ENDPOINTS = [
    (_request_home, 1),
    (_request_currency, 2),
    (_request_product, 10),
    (_request_cart, 3),
]
ENDPOINT_FUNCS = [e[0] for e in ENDPOINTS]
ENDPOINT_WEIGHTS = [e[1] for e in ENDPOINTS]


class DriverUser(FastHttpUser):
    wait_time = lambda self: 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cur_m = -1
        self._next_target = 0.0
        self._done = False

    @task
    def drive(self):
        now = time.time()
        epoch = GLOBAL["epoch"]
        if epoch is None or now < epoch:
            gevent.sleep(1.0)
            return

        m = int((now - epoch) // 60)
        if m != self._cur_m:
            self._cur_m = m
            self._next_target = epoch + m * 60
            self._done = False

        if self._done:
            gevent.sleep(1.0)
            return

        count = GLOBAL["counts"][m % GLOBAL["n_minutes"]]
        if count == 0:
            self._done = True
            gevent.sleep(1.0)
            return

        if now < self._next_target:
            gevent.sleep(max(0.05, self._next_target - now))
            return

        slot = _next_slot(m)
        if slot < count:
            gevent.spawn(self._hit)
            step = 60.0 / count
            self._next_target = epoch + m * 60 + (slot + 1) * step
        else:
            self._done = True
            gevent.sleep(1.0)

    def _hit(self):
        func = random.choices(ENDPOINT_FUNCS, weights=ENDPOINT_WEIGHTS, k=1)[0]
        func(self)


class MCRLoadShape(LoadTestShape):
    def tick(self):
        test_hours = GLOBAL["test_hours"]
        if test_hours is not None and self.get_run_time() > (
            test_hours * 3600 + START_BUFFER_SECONDS + 5
        ):
            return None
        pool = GLOBAL["user_pool"]
        return (pool, max(5, pool // 2))

import csv
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
MINUTE_STATS = {}    # minute index -> fired requests (for the end-of-test check)
_NEXT_USER_ID = 0    # per-user id generator (single-process runs only)


def _next_user_id() -> int:
    global _NEXT_USER_ID
    user_id = _NEXT_USER_ID
    _NEXT_USER_ID += 1
    return user_id


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


def _hit(user):
    func = random.choices(ENDPOINT_FUNCS, weights=ENDPOINT_WEIGHTS, k=1)[0]
    func(user)


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
        default=10000,
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


def _request_add_to_cart(user):
    data = {
        "product_id": random.choice(products),
        "quantity": str(random.randint(1, 3)),
    }
    resp = user.client.post("/cart", data=data, allow_redirects=False)
    if resp.status_code < 400:
        user._cart_items += 1


def _request_checkout(user):
    data = {
        "email": "user@example.com",
        "street_address": "123 Main St",
        "zip_code": "12345",
        "city": "Anytown",
        "state": "CA",
        "country": "US",
        "credit_card_number": "4432-8015-6152-0454",
        "credit_card_expiration_month": "12",
        "credit_card_expiration_year": "2030",
        "credit_card_cvv": "672",
    }
    resp = user.client.post("/cart/checkout", data=data, allow_redirects=False)
    if resp.status_code < 400:
        user._cart_items = 0


ENDPOINTS = [
    (_request_home, 1),
    (_request_currency, 2),
    (_request_product, 10),
    (_request_cart, 3),
    (_request_add_to_cart, 4),
    (_request_checkout, 1),
]
ENDPOINT_FUNCS = [e[0] for e in ENDPOINTS]
ENDPOINT_WEIGHTS = [e[1] for e in ENDPOINTS]


class DriverUser(FastHttpUser):
    wait_time = lambda self: 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_id = _next_user_id()
        self._cur_m = -1
        self._slot = self._user_id - GLOBAL["user_pool"]
        self._pending = None
        self._cart_items = 0

    @task
    def drive(self):
        now = time.time()
        epoch = GLOBAL["epoch"]
        if epoch is None or now < epoch:
            gevent.sleep(1.0)
            return

        pool = GLOBAL["user_pool"]
        m = int((now - epoch) // 60)
        if m != self._cur_m:
            self._cur_m = m
            self._slot = self._user_id - pool
            self._pending = None

        if self._pending is not None:
            slot, fire_at = self._pending
            if now < fire_at:
                gevent.sleep(fire_at - now)
                return
            self._pending = None
            gevent.spawn(self._hit)
            return

        count = GLOBAL["counts"][m % GLOBAL["n_minutes"]]
        if count == 0:
            gevent.sleep(1.0)
            return

        # Each user owns slots u, u+pool, u+2*pool, ... so the minute's
        # requests are paced one every 60/count seconds (fire_at is an absolute
        # time), instead of all firing in a burst at the top of the minute.
        slot = self._slot + pool
        if slot >= count:
            gevent.sleep(1.0)
            return
        self._slot = slot

        fire_at = epoch + m * 60 + slot * (60.0 / count)
        if now < fire_at:
            self._pending = (slot, fire_at)
            gevent.sleep(fire_at - now)
            return
        gevent.spawn(self._hit)

    def _hit(self):
        func = random.choices(ENDPOINT_FUNCS, weights=ENDPOINT_WEIGHTS, k=1)[0]
        if func is _request_checkout and self._cart_items <= 0:
            func = _request_add_to_cart
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

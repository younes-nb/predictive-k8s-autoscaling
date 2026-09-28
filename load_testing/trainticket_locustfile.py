"""Locust load for FudanSELab train-ticket (in-cluster, MCR-paced).

MCR pacing mirrors load_testing/locustfile.py: per-1-minute request counts
from an http_mcr CSV (--mcr-csv), paced evenly within each minute across a
fixed user pool (--user-pool). Each fired request runs one action
(gevent-spawned, so slow sagas like preserve don't block pacing).

Coverage: register -> login -> travel query -> contacts -> preserve -> pay
-> collect/enter -> cancel/rebook, plus broad read coverage of every
service. Per-user state (token/account/contacts/orderIds) chains the saga;
actions missing prerequisites fall back to reads. Coverage verified via
istio_requests_total by destination_workload,response_code.
"""

import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import gevent

from locust import FastHttpUser, LoadTestShape, events, task

NS = os.environ.get("TT_NAMESPACE", "train-ticket")
START_BUFFER_SECONDS = 15.0

SEED_FILE = os.environ.get("SEED_FILE", "/mnt/users.json")
SEED_POOL = []
if os.path.exists(SEED_FILE):
    try:
        with open(SEED_FILE) as f:
            SEED_POOL = json.load(f)
        print(f"[seed] loaded {len(SEED_POOL)} pre-seeded users from {SEED_FILE}",
              flush=True)
    except Exception as e:
        print(f"[seed] failed to load {SEED_FILE}: {e}", flush=True)

TRIP_IDS = ["G1234", "G1235", "G1236", "D1345"]
OD_PAIRS = [("nanjing", "shanghai"), ("nanjing", "suzhou"),
            ("suzhou", "shanghai"), ("wuxi", "shanghai")]
ROUTE_IDS = ["0b23bd3e-876a-4af3-b920-c50a90c90b04",
             "92708982-77af-4318-be25-57ccb0ff69ad"]


def _env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


GLOBAL = {
    "epoch": None,
    "counts": None,
    "n_minutes": 1,
    "user_pool": _env_int("USER_POOL", 100),
    "start_hours": 0.0,
    "test_hours": None,
}
MINUTE_STATS = {}
_NEXT_USER_ID = 0


def _next_user_id():
    global _NEXT_USER_ID
    v = _NEXT_USER_ID
    _NEXT_USER_ID += 1
    return v


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def url(svc, port, path):
    # Frontend-only mode: the ingress `/` catch-all sends everything to
    # ts-ui-dashboard:8080, which proxies /api/* to the backends. Clients use
    # relative paths and locust prepends --host
    # (https://train-ticket.younesnb.linkpc.net).
    # svc/port are kept as documentation of the backend mapping.
    return path


def load_mcr_counts(csv_path, max_requests, n_minutes=None, start_minutes=0):
    mcr = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mcr.append(float(row["http_mcr"]))
    if not mcr:
        sys.exit(f"No data rows in {csv_path}")
    if start_minutes:
        mcr = mcr[start_minutes:]
        if not mcr:
            sys.exit(f"No data rows after minute {start_minutes} of {csv_path}")
    if n_minutes is not None:
        mcr = mcr[:n_minutes]
        if not mcr:
            sys.exit(f"No data rows in the first {n_minutes} minutes of {csv_path}")
    peak = max(mcr)
    if peak <= 0:
        sys.exit(f"Workload window has peak http_mcr <= 0 in {csv_path}")
    return [max(0, int(round(v / peak * max_requests))) for v in mcr]


@events.init_command_line_parser.add_listener
def _add_cli_args(parser):
    parser.add_argument("--mcr-csv", type=str, required=True, env_var="MCR_CSV")
    parser.add_argument("--max-requests", type=int, default=300,
                        env_var="MAX_REQUESTS")
    parser.add_argument("--user-pool", type=int, default=100, env_var="USER_POOL")
    parser.add_argument("--test-hours", type=float, default=None, env_var="TEST_HOURS")
    parser.add_argument("--start-hours", type=float, default=0.0, env_var="START_HOURS")


@events.test_start.add_listener
def _on_test_start(environment, **_kw):
    opts = environment.parsed_options
    start_minutes = int(round(opts.start_hours * 60))
    n_minutes = int(round(opts.test_hours * 60)) if opts.test_hours else None
    counts = load_mcr_counts(opts.mcr_csv, opts.max_requests, n_minutes,
                             start_minutes)
    MINUTE_STATS.clear()
    GLOBAL.update(counts=counts, n_minutes=len(counts),
                  user_pool=opts.user_pool, start_hours=opts.start_hours,
                  test_hours=opts.test_hours if opts.test_hours else len(counts) / 60.0,
                  epoch=time.time() + START_BUFFER_SECONDS)
    log(f"Loaded {len(counts)} minutes from {opts.mcr_csv} "
        f"(max {opts.max_requests} req/min, pool {opts.user_pool}, "
        f"duration {GLOBAL['test_hours']:.2f}h)")


@events.test_stop.add_listener
def _on_test_stop(environment, **_kw):
    n = GLOBAL["n_minutes"]
    epoch = GLOBAL["epoch"]
    if epoch is None:
        return
    last_complete = int((time.time() - epoch) // 60) - 1
    bad = sum(1 for m in range(0, last_complete + 1)
              if MINUTE_STATS.get(m, 0) != GLOBAL["counts"][m % n])
    log(f"Per-minute check: {'PASS' if bad == 0 else f'{bad} mismatched'}")


@events.request.add_listener
def _tally(request_type, name, response_time, response_length,
           exception, start_time=None, **kw):
    if GLOBAL["epoch"] is None or start_time is None:
        return
    m = int((start_time - GLOBAL["epoch"]) // 60)
    if m >= 0:
        MINUTE_STATS[m] = MINUTE_STATS.get(m, 0) + 1


def _H(user):
    return {"Authorization": f"Bearer {user._token}"} if user._token else {}


def _ensure_login(user, force=False):
    if user._token and not force:
        return True
    try:
        r = user.client.post(
            url("ts-auth-service", 12340, "/api/v1/users/login"),
            json={"username": user._username, "password": "loadtest123"})
        if r.status_code == 200:
            d = r.json()["data"]
            user._token = d["token"]
            user._account_id = d["userId"]
            return True
    except Exception:
        pass
    return False


def _a_register(user):
    uname = f"ltuser{_next_user_id()}_{random.randint(0, 999999)}"
    try:
        r = user.client.post(
            url("ts-user-service", 12342, "/api/v1/userservice/users/register"),
            json={"userName": uname, "password": "loadtest123", "gender": 1,
                  "documentType": 1,
                  "documentNum": f"{random.randint(10**9, 10**10 - 1)}",
                  "email": f"{uname}@test.com"})
        if r.status_code in (200, 201):
            user._username = uname
            if _ensure_login(user) and user._account_id:
                c = user.client.post(
                    url("ts-contacts-service", 12347,
                        "/api/v1/contactservice/contacts"),
                    headers=_H(user),
                    json={"accountId": user._account_id, "name": "Tester",
                          "documentType": 1,
                          "documentNumber": f"{random.randint(10**9, 10**10 - 1)}",
                          "phoneNumber": f"138{random.randint(10**7, 10**8 - 1)}"})
                if c.status_code in (200, 201):
                    try:
                        user._contacts_id = c.json()["data"]["id"]
                    except Exception:
                        pass
    except Exception:
        pass


def _a_travel_query(user):
    o, d = random.choice(OD_PAIRS)
    day = random.randint(27, 29)
    user.client.post(
        url("ts-travel-service", 12346, "/api/v1/travelservice/trips/left"),
        json={"startPlace": o, "endPlace": d,
              "departureTime": f"2026-09-{day:02d}"})
    user.client.post(
        url("ts-travel2-service", 16346, "/api/v1/travel2service/trips/left"),
        json={"startPlace": o, "endPlace": d,
              "departureTime": f"2026-09-{day:02d}"})
    user.client.get(url("ts-travel-service", 12346,
                        f"/api/v1/travelservice/trips/{random.choice(TRIP_IDS)}"))
    user.client.post(
        url("ts-travel2-service", 16346, "/api/v1/travel2service/trips/left"),
        json={"startPlace": o, "endPlace": d,
              "departureTime": f"2026-09-{day:02d}"})


def _a_static_reads(user):
    u = url
    user.client.get(u("ts-route-service", 11178,
                      f"/api/v1/routeservice/routes/{random.choice(ROUTE_IDS)}"))
    user.client.get(u("ts-train-service", 14567, "/api/v1/trainservice/trains"))
    user.client.get(u("ts-station-service", 12345,
                      "/api/v1/stationservice/stations/name/shanghai"))
    user.client.get(u("ts-price-service", 16579, "/api/v1/priceservice/prices"))
    user.client.get(u("ts-config-service", 15679, "/api/v1/configservice/configs"))
    user.client.get(u("ts-basic-service", 15680, "/api/v1/basicservice/welcome"))
    user.client.get(u("ts-route-plan-service", 14578,
                      "/api/v1/routeplanservice/welcome"))
    user.client.get(u("ts-travel-plan-service", 14322,
                      "/api/v1/travelplanservice/welcome"))


def _a_menus(user):
    if not _ensure_login(user):
        return
    u, H = url, _H(user)
    user.client.get(u("ts-assurance-service", 18888,
                      "/api/v1/assuranceservice/assurances"), headers=H)
    user.client.get(u("ts-assurance-service", 18888,
                      "/api/v1/assuranceservice/assurances/types"), headers=H)
    user.client.get(u("ts-food-service", 18856, "/api/v1/foodservice/welcome"))
    user.client.get(u("ts-food-service", 18856, "/api/v1/foodservice/orders"))
    user.client.get(u("ts-food-service", 18856,
                      "/api/v1/foodservice/foods/2026-09-27/shanghai/taiyuan/G1234"))
    user.client.get(u("ts-food-map-service", 18855, "/welcome"), headers=H)
    user.client.get(u("ts-consign-service", 16111,
                      "/api/v1/consignservice/welcome"), headers=H)
    user.client.get(u("ts-consign-price-service", 16110,
                      "/api/v1/consignpriceservice/welcome"), headers=H)
    user.client.get(u("ts-security-service", 11188,
                      "/api/v1/securityservice/securityConfigs"), headers=H)
    user.client.get(u("ts-contacts-service", 12347,
                      f"/api/v1/contactservice/contacts/account/{user._account_id or 'x'}"),
                    headers=H)


def _a_leaf_probes(user):
    if not _ensure_login(user):
        return
    u = url
    H = _H(user)
    user.client.get(u("ts-seat-service", 18898, "/api/v1/seatservice/welcome"))
    user.client.get(u("ts-payment-service", 19001, "/api/v1/paymentservice/welcome"), headers=H)
    user.client.get(u("ts-payment-service", 19001, "/api/v1/paymentservice/payment"),
                    headers=H)
    try:
        user.client.post(u("ts-payment-service", 19001,
                           "/api/v1/paymentservice/payment"),
                         headers=H,
                         json={"userId": user._account_id or "x",
                               "orderId": random.choice(user._order_ids)
                               if user._order_ids else "x",
                               "tripId": "G1234", "price": "100.0"})
    except Exception:
        pass
    user.client.get(u("ts-execute-service", 12386, "/api/v1/executeservice/welcome"), headers=H)
    user.client.get(u("ts-rebook-service", 18886, "/api/v1/rebookservice/welcome"), headers=H)
    user.client.get(u("ts-ticket-office-service", 16108, "/office/getAll"))
    user.client.get(u("ts-ticket-office-service", 16108, "/office/getRegionList"))
    user.client.get(u("ts-news-service", 12862, "/api/v1/newsservice/welcome"))
    try:
        user.client.post(
            u("ts-avatar-service", 17001, "/api/v1/avatar"),
            json={"img": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="})
    except Exception:
        pass
    user.client.get(u("ts-verification-code-service", 15678,
                      "/api/v1/verifycode/generate/"))
    user.client.get(u("ts-ui-dashboard", 8080, "/"), headers=H)


def _a_admin_reads(user):
    if not _ensure_login(user):
        return
    H, u = _H(user), url
    user.client.get(u("ts-admin-travel-service", 16114,
                      "/api/v1/admintravelservice/admintravel"), headers=H)
    user.client.get(u("ts-admin-route-service", 16113,
                      "/api/v1/adminrouteservice/adminroute"), headers=H)
    user.client.get(u("ts-admin-user-service", 16115,
                      "/api/v1/adminuserservice/users"), headers=H)
    user.client.get(u("ts-admin-order-service", 16112,
                      "/api/v1/adminorderservice/adminorder"), headers=H)
    user.client.get(u("ts-admin-basic-info-service", 18767,
                      "/api/v1/adminbasicservice/adminbasic/contacts"), headers=H)
    user.client.get(u("ts-admin-basic-info-service", 18767,
                      "/api/v1/adminbasicservice/adminbasic/stations"), headers=H)


def _a_orders_query(user):
    if not _ensure_login(user):
        return
    H = _H(user)
    user.client.post(
        url("ts-order-service", 12031, "/api/v1/orderservice/order/query"),
        headers=H, json={"loginId": user._account_id})
    try:
        for path in ("/api/v1/orderservice/order/query",
                     "/api/v1/orderOtherService/orderOther/query"):
            r = user.client.post(
                url("ts-order-service" if "orderservice/" in path
                    else "ts-order-other-service",
                    12031 if "orderservice/" in path else 12032, path),
                headers=H, json={"loginId": user._account_id})
            if r.status_code == 200:
                data = r.json().get("data") or []
                for o in data[:3]:
                    if isinstance(o, dict) and o.get("id"):
                        user._order_ids.append(o["id"])
    except Exception:
        pass


def _a_preserve(user):
    if not _ensure_login(user) or not user._contacts_id:
        _a_travel_query(user)
        return
    o, d = random.choice(OD_PAIRS[:3])
    try:
        r = user.client.post(
            url("ts-preserve-service", 14568, "/api/v1/preserveservice/preserve"),
            headers=_H(user),
            json={"accountId": user._account_id, "contactsId": user._contacts_id,
                  "tripId": random.choice(TRIP_IDS[:3]),
                  "seatType": random.choice([2, 3]),
                  "date": f"2026-09-{random.randint(27, 29):02d}",
                  "from": o, "to": d, "assurance": 0})
        if r.status_code == 200:
            _a_orders_query(user)
    except Exception:
        pass


def _a_pay_execute(user):
    if not _ensure_login(user) or not user._order_ids:
        _a_orders_query(user)
        return
    oid = random.choice(user._order_ids)
    H = _H(user)
    try:
        price = "100.0"
        try:
            pr = user.client.get(
                url("ts-order-service", 12031,
                    f"/api/v1/orderservice/order/price/{oid}"), headers=H)
            if pr.status_code == 200:
                price = str(pr.json().get("data", price))
        except Exception:
            pass
        user.client.post(
            url("ts-inside-payment-service", 18673,
                "/api/v1/inside_pay_service/inside_payment"),
            headers=H,
            json={"userId": user._account_id, "orderId": oid,
                  "tripId": random.choice(TRIP_IDS[:3]), "price": price})
        user.client.get(url("ts-execute-service", 12386,
                            f"/api/v1/executeservice/execute/collected/{oid}"),
                        headers=H)
        user.client.get(url("ts-execute-service", 12386,
                            f"/api/v1/executeservice/execute/execute/{oid}"),
                        headers=H)
        if random.random() < 0.25:
            user.client.get(url("ts-cancel-service", 18885,
                                f"/api/v1/cancelservice/cancel/{oid}/{user._account_id}"),
                            headers=H)
        if random.random() < 0.15 and len(user._order_ids) > 0:
            oid2 = random.choice(user._order_ids)
            try:
                user.client.post(
                    url("ts-rebook-service", 18886, "/api/v1/rebookservice/rebook"),
                    headers=H,
                    json={"orderId": oid2, "oldTripId": "G1234",
                          "tripId": "G1235", "seatType": 2,
                          "date": f"2026-09-{random.randint(27, 29):02d}"})
            except Exception:
                pass
    except Exception:
        pass


ENDPOINTS = [
    (_a_travel_query, 10),
    (_a_static_reads, 6),
    (_a_menus, 5),
    (_a_leaf_probes, 4),
    (_a_orders_query, 5),
    (_a_preserve, 2),
    (_a_pay_execute, 3),
    (_a_admin_reads, 1),
    (_a_register, 1),
]
ENDPOINT_FUNCS = [e[0] for e in ENDPOINTS if e[1] > 0]
ENDPOINT_WEIGHTS = [e[1] for e in ENDPOINTS if e[1] > 0]


class DriverUser(FastHttpUser):
    wait_time = lambda self: 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_id = _next_user_id()
        self._cur_m = -1
        self._slot = self._user_id - GLOBAL["user_pool"]
        self._pending = None
        self._token = None
        self._account_id = None
        self._contacts_id = None
        self._order_ids = []
        self._username = f"ltpool{self._user_id}"

    def on_start(self):
        gevent.sleep(random.uniform(0, 20))
        if SEED_POOL:
            s = random.choice(SEED_POOL)
            self._username = s["username"]
            self._account_id = s.get("accountId")
            self._contacts_id = s.get("contactsId")
            self._token = None
            _ensure_login(self)
        else:
            _a_register(self)

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
        if random.random() < 0.03:
            _ensure_login(self, force=True)
        func = random.choices(ENDPOINT_FUNCS, weights=ENDPOINT_WEIGHTS, k=1)[0]
        try:
            func(self)
        except Exception:
            pass


class MCRLoadShape(LoadTestShape):
    def tick(self):
        test_hours = GLOBAL["test_hours"]
        if test_hours is not None and self.get_run_time() > (
            test_hours * 3600 + START_BUFFER_SECONDS + 5
        ):
            return None
        pool = GLOBAL["user_pool"]
        return (pool, max(5, pool // 2))

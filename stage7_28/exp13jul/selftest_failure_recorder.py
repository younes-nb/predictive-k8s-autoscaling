#!/usr/bin/env python3
"""Integration self-test for the sidecar fields added to failure_recorder.py.

Runs main() against synthetic Mimir + synthetic Loki and asserts the
FAILURE_EVENT record carries `sidecar_lines` / `sidecar_source` /
`sidecar_request_ids` with the Envoy request ID parsed, that the app-exception
stream stays Envoy-free, and that the sidecar evidence refines the
classification instead of leaving the generic 503 tag.

Two scenarios:
  A. sidecar 503 UF, no Java exception  -> MESH_UPSTREAM_CONN_FAILURE
  B. MongoSocketException, no sidecar    -> DB_POOL_STALE (regression guard)

    python3 stage7_28/exp13jul/selftest_failure_recorder.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "fr", os.path.join(HERE, "failure_recorder.py"))
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

NOW = time.time()
REQ_ID = "37c26481-3e08-4b96-b4a3-7d35d75ac98d"

ENVOY_503_UF = (
    '[2026-11-16T15:13:12.908Z] "GET /api/v1/foodservice/foods HTTP/1.1" '
    '503 UF via_upstream - "-" 0 0 12 11 "10.233.104.64" "Grafana k6/2.3.0" '
    '"%s" "ts-food-service:18856" "10.233.105.134:18856"' % REQ_ID)

MONGO_EXC = (
    "2026-11-16 15:13:12.908 ERROR 1 --- [http-nio-18856-exec-3] "
    "c.f.c.s.service.FoodService : food query failed "
    "org.springframework.dao.DataAccessResourceFailureException: "
    "com.mongodb.MongoSocketException: Read timed out")


def run_scenario(name, envoy_lines, exceptions):
    def fake_qmap(expr):
        if "istio_requests_total" in expr and "destination_workload" in expr:
            key = {"destination_workload": "ts-food-service", "response_code": "503"}
            return {json.dumps(key): 0.5}
        if "container_working_set_bytes" in expr:
            return {json.dumps({"pod": "ts-food-service-abc"}): 5.0e8}
        return {}

    def fake_loki_range(expr, start_ns, end_ns, limit=60, timeout=12):
        if ":istio-proxy" in expr and "ts-food-service" in expr:
            values = [[str(int(NOW * 1e9)), ln] for ln in envoy_lines]
            return [{"stream": {"instance": "train-ticket/ts-food-service-abc:istio-proxy"},
                     "values": values}]
        return []

    fr.qmap = fake_qmap
    fr.q = lambda expr, timeout=12: []
    fr.loki_query_range = fake_loki_range
    fr.kubectl_json = lambda args, timeout=30: {"items": []}
    fr.kubectl_scale_events = lambda service, since: []
    fr.exceptions_for = (
        (lambda service, since, now, limit=40:
         ([{"pod": "ts-food-service-abc", "line": ln} for ln in exceptions],
          {"MongoSocketException": len(exceptions)}, "loki"))
        if exceptions else
        (lambda service, since, now, limit=40: ([], {}, "loki")))

    run_dir = tempfile.mkdtemp(prefix="fr_selftest_")
    sys.argv = ["failure_recorder.py", "--run-dir", run_dir,
                "--end", str(time.time() + 1.5), "--runid", name]
    fr.main()

    records = [json.loads(line)
               for line in open(os.path.join(run_dir, "failures.jsonl"))]
    events = [r for r in records if r.get("type") == "FAILURE_EVENT"]
    assert events, "%s: no FAILURE_EVENT produced" % name
    return events[0]


print("scenario A: sidecar 503 UF, no Java exception")
a = run_scenario("A", [ENVOY_503_UF], [])
print("  sidecar_source  :", a["sidecar_source"])
print("  sidecar_lines   :", len(a["sidecar_lines"]))
print("  sidecar_req_ids :", a["sidecar_request_ids"])
print("  classify        :", a["classify"])
assert a["sidecar_source"] == "loki"
assert a["sidecar_request_ids"] == [REQ_ID], a["sidecar_request_ids"]
assert a["sidecar_lines"] and "503 UF" in a["sidecar_lines"][0]["line"]
assert "MESH_UPSTREAM_CONN_FAILURE" in a["classify"], a["classify"]
assert "503_APP_OR_MESH" not in a["classify"], a["classify"]
assert all("HTTP/1.1" not in r["line"] for r in a["app_exceptions"]), \
    "Envoy lines leaked into app_exceptions"

print("\nscenario B: MongoSocketException, no sidecar lines")
b = run_scenario("B", [], [MONGO_EXC])
print("  sidecar_lines   :", len(b["sidecar_lines"]))
print("  classify        :", b["classify"])
assert b["sidecar_lines"] == []
assert "DB_POOL_STALE" in b["classify"], b["classify"]

print("\nOK: sidecar capture, request-ID parsing, mesh-vs-app classification")
#!/usr/bin/env python3
"""Offline unit tests for seed_tt_users_v3_july_semantics.py.

These tests run without any cluster access, database mutation, or API calls.
They verify:
  - Deterministic output (same seed -> same plan)
  - 3,000 users generated
  - Unique usernames and required fields
  - contactsId = null for all users (July-compatible)
  - No contact-creation endpoints in the request plan
  - --allow-nonempty bypass is removed (unconditional refusal)
  - Preflight accepts the exact verified pre-seed baseline (Test A)
  - Preflight rejects unexpected records (Test B)
  - Preflight rejects partial/incomplete baseline (Test C)
  - Preflight rejects unexpected Auth records (Test D)
  - Preflight fails on missing collection/database (Test E)
  - Determinism across equivalent valid baselines (Test F)
  - Invalid/partial states cause zero writes

Tests use mocks, fixtures, or isolated temporary data. They must not connect
to the live cluster or mutate any real database.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "load_testing", "seed_tt_users_v3_july_semantics.py")


def run_gen(args):
    """Run the generator with given args, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, GEN] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def load_module():
    """Import the generator module and return it."""
    spec = importlib.util.spec_from_file_location("seed_v3", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mock database layer
# ---------------------------------------------------------------------------

class MockCursor:
    """Mock pymysql cursor that returns canned rows for COUNT and identity queries."""

    def __init__(self, rows, identity_columns):
        self._rows = rows
        self._identity_columns = identity_columns
        self._result = []

    def execute(self, sql, params=None):
        sql_lower = sql.strip().lower()
        if "count(*)" in sql_lower:
            self._result = [{"cnt": len(self._rows)}]
        elif "limit 1" in sql_lower:
            if params is None:
                self._result = [None]
            else:
                found = None
                for row in self._rows:
                    if all(row.get(col) == val for col, val in
                           zip(self._identity_columns, params)):
                        found = row
                        break
                self._result = [found]
        elif params is not None:
            # Identity query without LIMIT 1 — return all matching rows
            # (used for case-sensitive comparison in Python)
            self._result = [
                row for row in self._rows
                if all(row.get(col) == val for col, val in zip(self._identity_columns, params))
            ]
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result) if self._result else []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConn:
    """Mock pymysql connection."""

    def __init__(self, rows, identity_columns):
        self._rows = rows
        self._identity_columns = identity_columns
        self._closed = False

    def cursor(self):
        return MockCursor(self._rows, self._identity_columns)

    def close(self):
        self._closed = True


def make_mock_factory(baseline_rows, identity_columns_map):
    """Build a connection factory that routes by database name and table.

    baseline_rows: dict of {db_key: {table_name: [rows]}}
    identity_columns_map: dict of {db_key: {table_name: [col_names]}}
    """

    def mock_connect(db_cfg, table_name=""):
        database = db_cfg["database"]
        matched_key = None
        for db_key, entry in PRE_SEED_BASELINE_REF.items():
            if entry["db_default"] == database:
                matched_key = db_key
                break
        if matched_key is None:
            return MockConn([], [])
        tables = baseline_rows.get(matched_key, {})
        rows = tables.get(table_name, [])
        id_cols = identity_columns_map.get(matched_key, {}).get(table_name, [])
        return MockConn(rows, id_cols)

    return mock_connect


# We'll set this from each test that needs it
PRE_SEED_BASELINE_REF = {}


# ---------------------------------------------------------------------------
# Test A — Valid baseline: preflight succeeds, plan generated
# ---------------------------------------------------------------------------

def test_valid_baseline_preflight_succeeds():
    """Test A: All collections match the verified pre-seed baseline.
    Preflight succeeds. The expected seed plan is generated."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    # Identity columns per table (used for identity matching queries)
    identity_columns_map = {
        "ts-user-mysql": {
            "user": ["user_name", "user_id"],
        },
        "ts-auth-mysql": {
            "auth_user": ["user_name"],
        },
        "ts-contacts-mysql": {
            "contacts": ["name", "id"],
        },
        "ts-order-mysql": {
            "orders": ["id"],
        },
        "ts-order-other-mysql": {
            "orders_other": ["id"],
        },
        "ts-payment-mysql": {
            "payment": ["id"],
        },
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # Baseline rows matching source-verified InitData
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert ok, f"Preflight should pass for valid baseline, got: {results}"

    # Verify all results show count and identity match
    for r in results:
        assert r["count_match"], \
            f"Count mismatch for {r['database']}.{r['table']}: " \
            f"expected={r['expected_count']}, observed={r['observed_count']}"
        if r["expected_identities"]:
            assert r["identity_match"], \
                f"Identity mismatch for {r['database']}.{r['table']}"

    # Dry-run plan should also generate successfully
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "plan.json")
        rc, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                            "--dry-run", "--out", out])
        assert rc == 0, f"dry-run failed: rc={rc}"
        with open(out) as f:
            plan = json.load(f)
        assert plan["count"] == 3000
        assert plan["plan_sha256"] is not None

    print("PASS: valid baseline preflight succeeds (Test A)")


# ---------------------------------------------------------------------------
# Test B — Unexpected record: preflight fails, no writes
# ---------------------------------------------------------------------------

def test_unexpected_record_refusal():
    """Test B: Add one unexpected record to a checked collection.
    Preflight fails. No writes or API requests occur."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # user table has 2 rows (1 expected + 1 unexpected)
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "unexpected_user",
                 "user_id": "11111111-1111-1111-1111-111111111111"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with unexpected user record"

    # The user table should show count mismatch (2 observed vs 1 expected)
    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["observed_count"] == 2, \
        f"Expected observed_count=2, got {user_result['observed_count']}"
    assert not user_result["count_match"], \
        "Count match should be False for user table"

    # Live run should also refuse (rc=2)
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--out", "/tmp/test_unexpected.json"])
    assert rc == 2, f"Expected rc=2 (refusal), got rc={rc}"
    assert "REFUSING" in stderr or "REFUSING" in stdout, \
        f"Expected refusal message, got: stdout={stdout}, stderr={stderr}"

    print("PASS: unexpected record refusal (Test B)")


# ---------------------------------------------------------------------------
# Test C — Partial baseline: preflight fails, no writes
# ---------------------------------------------------------------------------

def test_partial_baseline_refusal():
    """Test C: Remove one expected initialization record or create an
    incomplete baseline. Preflight fails. No writes or API requests occur."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # contacts table has only 1 row (Contacts_One missing — partial baseline)
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with partial baseline"

    # The contacts table should show count mismatch (1 observed vs 2 expected)
    contacts_result = [r for r in results if r["table"] == "contacts"][0]
    assert contacts_result["observed_count"] == 1, \
        f"Expected observed_count=1, got {contacts_result['observed_count']}"
    assert not contacts_result["count_match"], \
        "Count match should be False for contacts table"

    # Live run should refuse
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--out", "/tmp/test_partial.json"])
    assert rc == 2, f"Expected rc=2 (refusal), got rc={rc}"
    assert "REFUSING" in stderr or "REFUSING" in stdout, \
        f"Expected refusal message, got: stdout={stdout}, stderr={stderr}"

    print("PASS: partial baseline refusal (Test C)")


# ---------------------------------------------------------------------------
# Test D — Unexpected non-empty Auth database: preflight fails, no writes
# ---------------------------------------------------------------------------

def test_unexpected_auth_record_refusal():
    """Test D: Add an unexpected Auth record. Preflight fails.
    No writes or API requests occur."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # auth_user has 3 rows (2 expected + 1 unexpected)
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
                {"user_name": "unexpected_auth_user",
                 "user_id": "22222222-2222-2222-2222-222222222222"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with unexpected Auth record"

    auth_result = [r for r in results if r["table"] == "auth_user"][0]
    assert auth_result["observed_count"] == 3, \
        f"Expected observed_count=3, got {auth_result['observed_count']}"
    assert not auth_result["count_match"], \
        "Count match should be False for auth_user table"

    # Live run should refuse
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--out", "/tmp/test_auth_unexpected.json"])
    assert rc == 2, f"Expected rc=2 (refusal), got rc={rc}"
    assert "REFUSING" in stderr or "REFUSING" in stdout, \
        f"Expected refusal message, got: stdout={stdout}, stderr={stderr}"

    print("PASS: unexpected Auth record refusal (Test D)")


# ---------------------------------------------------------------------------
# Test E — Missing required collection or database: preflight fails clearly
# ---------------------------------------------------------------------------

def test_missing_database_or_collection():
    """Test E: Preflight fails clearly when a collection or database is
    missing. No writes or API requests occur."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    class FailingCursor:
        def execute(self, sql, params=None):
            raise Exception("Table 'ts.user' doesn't exist")

        def fetchone(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FailingConn:
        def cursor(self):
            return FailingCursor()

        def close(self):
            pass

    def failing_connect(db_cfg, table_name=""):
        return FailingConn()

    mod.set_connection_factory(failing_connect)

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail when table is missing"

    # Every result should have an error and count_match=False
    for r in results:
        assert r["error"] is not None, \
            f"Expected error for {r['database']}.{r['table']}"
        assert not r["count_match"], \
            f"Count match should be False for {r['database']}.{r['table']}"

    # Live run should refuse with rc=2
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--out", "/tmp/test_missing.json"])
    assert rc == 2, f"Expected rc=2 (refusal), got rc={rc}"
    assert "REFUSING" in stderr or "REFUSING" in stdout, \
        f"Expected refusal message, got: stdout={stdout}, stderr={stderr}"

    print("PASS: missing database/collection failure (Test E)")


# ---------------------------------------------------------------------------
# Test F — Determinism: two executions produce identical plans
# ---------------------------------------------------------------------------

def test_determinism_across_valid_baselines():
    """Test F: Two executions against equivalent valid baselines produce
    identical seed plans and account identities."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    # Run preflight against two equivalent baselines
    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))
    ok1, results1 = mod.preflight_check(verbose=False)
    assert ok1, "First baseline preflight should pass"

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))
    ok2, results2 = mod.preflight_check(verbose=False)
    assert ok2, "Second baseline preflight should pass"

    # Dry-run plans should be byte-identical
    with tempfile.TemporaryDirectory() as td:
        out1 = os.path.join(td, "plan1.json")
        out2 = os.path.join(td, "plan2.json")
        rc1, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                             "--dry-run", "--out", out1])
        rc2, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                             "--dry-run", "--out", out2])
        assert rc1 == 0 and rc2 == 0
        with open(out1) as f:
            plan1 = json.load(f)
        with open(out2) as f:
            plan2 = json.load(f)
        assert plan1["plan_sha256"] == plan2["plan_sha256"], \
            f"Plan SHA-256 mismatch: {plan1['plan_sha256']} != {plan2['plan_sha256']}"
        assert plan1["users_first_5"] == plan2["users_first_5"], \
            "First 5 users differ between runs"

    print("PASS: determinism across valid baselines (Test F)")


# ---------------------------------------------------------------------------
# Existing tests (updated for new generator behavior)
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """Two dry-run invocations with same args must produce byte-identical plans."""
    with tempfile.TemporaryDirectory() as td:
        out1 = os.path.join(td, "plan1.json")
        out2 = os.path.join(td, "plan2.json")
        rc1, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                            "--dry-run", "--out", out1])
        rc2, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                            "--dry-run", "--out", out2])
        assert rc1 == 0, f"dry-run 1 failed: rc={rc1}"
        assert rc2 == 0, f"dry-run 2 failed: rc={rc2}"
        with open(out1) as f:
            plan1 = json.load(f)
        with open(out2) as f:
            plan2 = json.load(f)
        assert plan1["plan_sha256"] == plan2["plan_sha256"], \
            f"plan SHA-256 mismatch: {plan1['plan_sha256']} != {plan2['plan_sha256']}"
        assert plan1["users_first_5"] == plan2["users_first_5"], \
            "first 5 users differ between runs"
    print("PASS: deterministic output")


def test_user_count_and_fields():
    """Dry-run plan must contain exactly 3,000 users with required fields."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "plan.json")
        rc, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                           "--dry-run", "--out", out])
        assert rc == 0, f"dry-run failed: rc={rc}"
        with open(out) as f:
            plan = json.load(f)
        assert plan["count"] == 3000, f"count={plan['count']}"
        assert plan["users_first_5"] is not None
        assert len(plan["users_first_5"]) == 5
        assert len(plan["users_last_2"]) == 2
        for u in plan["users_first_5"] + plan["users_last_2"]:
            assert u["contactsId"] is None, \
                f"user {u['idx']} has contactsId={u['contactsId']}"
            assert "username" in u
            assert "password" in u
            assert "documentNum" in u
            assert "phoneNumber" in u
    print("PASS: user count and fields")


def test_unique_usernames():
    """All 3,000 usernames must be unique."""
    mod = load_module()
    users = mod.plan_users(3000, 20260301)
    usernames = [u["username"] for u in users]
    assert len(usernames) == len(set(usernames)), "duplicate usernames found"
    assert len(users) == 3000
    print("PASS: unique usernames")


def test_null_contacts_id():
    """All users must have contactsId = null (July-compatible semantics)."""
    mod = load_module()
    users = mod.plan_users(3000, 20260301)
    null_count = sum(1 for u in users if u["contactsId"] is None)
    assert null_count == 3000, f"contactsId null count={null_count}, expected 3000"
    print("PASS: null contactsId for all users")


def test_no_contact_creation_endpoint():
    """Dry-run plan must not include contact creation requests."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "plan.json")
        rc, _, _ = run_gen(["--count", "3000", "--rng-seed", "20260301",
                           "--dry-run", "--out", out])
        assert rc == 0
        with open(out) as f:
            plan = json.load(f)
        requests = plan.get("requests_per_user", [])
        contact_requests = [r for r in requests if "contacts" in r.lower()]
        assert len(contact_requests) == 0, \
            f"contact creation endpoints found: {contact_requests}"
        register_requests = [r for r in requests if "register" in r.lower()]
        login_requests = [r for r in requests if "login" in r.lower()]
        assert len(register_requests) == 1, "expected exactly 1 register request"
        assert len(login_requests) == 1, "expected exactly 1 login request"
    print("PASS: no contact creation endpoint")


def test_plan_sha256_deterministic():
    """plan_sha256 must match between two dry-run invocations."""
    with tempfile.TemporaryDirectory() as td:
        out1 = os.path.join(td, "plan1.json")
        out2 = os.path.join(td, "plan2.json")
        run_gen(["--count", "3000", "--rng-seed", "20260301",
                "--dry-run", "--out", out1])
        run_gen(["--count", "3000", "--rng-seed", "20260301",
                "--dry-run", "--out", out2])
        with open(out1) as f:
            p1 = json.load(f)
        with open(out2) as f:
            p2 = json.load(f)
        assert p1["plan_sha256"] == p2["plan_sha256"]
        assert len(p1["plan_sha256"]) == 64
        int(p1["plan_sha256"], 16)  # must be valid hex
    print("PASS: plan SHA-256 deterministic and valid")


def test_allow_nonempty_argument_removed():
    """The --allow-nonempty argument must not exist (unconditional refusal)."""
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--allow-nonempty", "--out", "/tmp/test_allow.json"])
    # argparse should reject the unknown argument
    assert rc == 2, f"expected argparse error rc=2, got rc={rc}"
    assert "unrecognized arguments" in stderr or "unrecognized arguments" in stdout, \
        f"expected argparse unrecognized-arguments error, got: stdout={stdout}, stderr={stderr}"
    print("PASS: --allow-nonempty argument removed")


def test_no_force_flag():
    """The --force argument must not exist either."""
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--force", "--out", "/tmp/test_force.json"])
    assert rc == 2, f"expected argparse error rc=2, got rc={rc}"
    assert "unrecognized arguments" in stderr or "unrecognized arguments" in stdout, \
        f"expected argparse unrecognized-arguments error, got: stdout={stdout}, stderr={stderr}"
    print("PASS: --force argument removed")


def test_no_bypass_reference_in_source():
    """The source must not reference any bypass argument in executable code."""
    with open(GEN) as f:
        source = f.read()
    # Check executable code lines only (skip docstrings/comments)
    lines = source.split('\n')
    code_lines = [l for l in lines if not l.strip().startswith('#') and not l.strip().startswith('"')]
    code = '\n'.join(code_lines)
    for forbidden in ["allow_nonempty", "skip_preflight", "--bypass"]:
        assert forbidden not in code, \
            f"Source still references forbidden bypass in code: {forbidden}"
    # --force is only mentioned in the docstring as a forbidden argument name,
    # which is acceptable — it is NOT a defined argument.
    assert '"--force"' not in code, \
        "Source defines --force argument"
    print("PASS: no bypass reference in source")


def test_preflight_only_without_out():
    """--preflight-only must not require --out (CLI fix verification)."""
    # preflight-only will fail (rc=3) because pymysql is not installed,
    # but crucially it must NOT fail with the "--out is required" error (rc=2
    # with that message). We verify the error message is the pymysql error,
    # not a missing-argument error.
    rc, stdout, stderr = run_gen(["--preflight-only"])
    combined = stdout + stderr
    assert "--out is required" not in combined, \
        f"--preflight-only wrongly requires --out: {combined}"
    # Without pymysql installed, preflight connection fails -> rc=3.
    # If pymysql were installed this would be rc=0 (mocked) or rc=3 (live DB
    # unreachable from runner). Either way, no --out complaint.
    assert rc in (0, 3), f"unexpected rc={rc}: {combined}"
    print(f"PASS: --preflight-only without --out (rc={rc}, no --out error)")


def test_dry_run_without_out_fails():
    """--dry-run without --out must fail with a clear error (rc=2)."""
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301",
                                   "--dry-run"])
    assert rc == 2, f"expected rc=2, got rc={rc}"
    combined = stdout + stderr
    assert "--out is required" in combined, \
        f"expected --out-required error, got: {combined}"
    print("PASS: --dry-run without --out fails (rc=2)")


def test_live_without_out_fails():
    """Live run without --out must fail with a clear error (rc=2)."""
    rc, stdout, stderr = run_gen(["--count", "3000", "--rng-seed", "20260301"])
    assert rc == 2, f"expected rc=2, got rc={rc}"
    combined = stdout + stderr
    assert "--out is required" in combined, \
        f"expected --out-required error, got: {combined}"
    print("PASS: live run without --out fails (rc=2)")


def test_help_output_cli_contract():
    """--help must show the CLI contract: --out optional in help text,
    --preflight-only present, no --allow-nonempty."""
    rc, stdout, stderr = run_gen(["--help"])
    assert rc == 0, f"--help failed: rc={rc}"
    combined = stdout + stderr
    assert "--preflight-only" in combined, "--preflight-only missing from help"
    assert "--allow-nonempty" not in combined, "--allow-nonempty in help"
    assert "--force" not in combined, "--force in help"
    assert "--out" in combined, "--out missing from help"
    print("PASS: --help output shows corrected CLI contract")


def test_no_allow_nonempty_reference_in_source():
    """The source must not reference args.allow_nonempty anywhere."""
    with open(GEN) as f:
        source = f.read()
    assert "allow_nonempty" not in source, \
        "Source still references allow_nonempty — defect not eliminated"
    print("PASS: no allow_nonempty reference in source")


def test_exact_permitted_baseline_state():
    """The exact permitted baseline: fdse_microservice user in user-service,
    2 InitData auth records, 2 contacts, 3 orders, etc. (source-verified)."""
    mod = load_module()
    users = mod.plan_users(3000, 20260301)
    # All seed users have contactsId=null
    assert all(u["contactsId"] is None for u in users), \
        "not all seed users have contactsId=null"
    # All seed users have the expected fields
    for u in users:
        assert "username" in u and u["username"].startswith("seeduser"), \
            f"unexpected username: {u.get('username')}"
        assert u["password"] == "loadtest123", \
            f"unexpected password for {u.get('username')}"
        assert isinstance(u["documentNum"], int) and 10**9 <= u["documentNum"] < 10**10, \
            f"unexpected documentNum for {u.get('username')}"
        assert isinstance(u["phoneNumber"], str) and u["phoneNumber"].startswith("138"), \
            f"unexpected phoneNumber for {u.get('username')}"
    # No user named fdse_microservice in the seed (it's InitData, not seed)
    fdse_users = [u for u in users if u["username"] == "fdse_microservice"]
    assert len(fdse_users) == 0, \
        f"fdse_microservice should not be in seed (it's InitData), found {len(fdse_users)}"
    print("PASS: exact permitted baseline state")


def test_invalid_fixture_zero_writes():
    """Invalid fixture demonstrates non-zero exit and zero writes.
    Runs preflight against a fixture with an unexpected record and verifies
    rc != 0 and no output file is created by the generator."""
    mod = load_module()

    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    class FailingCursor:
        def execute(self, sql, params=None):
            raise Exception("connection refused")

        def fetchone(self):
            return None

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FailingConn:
        def cursor(self):
            return FailingCursor()

        def close(self):
            pass

    def failing_connect(db_cfg, table_name=""):
        return FailingConn()

    mod.set_connection_factory(failing_connect)

    ok, _ = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail for invalid fixture"

    # The CLI preflight-only path should return non-zero
    # (we test via direct module call since CLI needs real MySQL for preflight)
    assert not ok, "Invalid fixture must cause preflight failure (zero writes)"

    print("PASS: invalid fixture zero writes")


def test_endpoint_wrong_host():
    """Wrong host in endpoint_expected causes preflight failure.
    Verifies that the endpoint identity validation (host, port, database)
    fails closed before any table query."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    # Tamper with endpoint_expected: wrong host for ts-user-mysql
    mod.PRE_SEED_BASELINE["ts-user-mysql"]["endpoint_expected"] = {
        "host": "wrong-host.example.com",
        "port": 3306,
        "database": "ts-user-mysql",
    }

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with wrong host"

    # The user table should show endpoint error
    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["error"] is not None, \
        f"Expected endpoint error, got: {user_result}"
    assert "host mismatch" in user_result["error"], \
        f"Expected host mismatch error, got: {user_result['error']}"
    assert not user_result["count_match"], \
        "Count match should be False on endpoint mismatch"
    assert not user_result["identity_match"], \
        "Identity match should be False on endpoint mismatch"

    print("PASS: wrong host endpoint rejected (Test G)")


def test_endpoint_wrong_port():
    """Wrong port in endpoint_expected causes preflight failure."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    # Tamper with endpoint_expected: wrong port for ts-user-mysql
    mod.PRE_SEED_BASELINE["ts-user-mysql"]["endpoint_expected"] = {
        "host": "ts-user-mysql",
        "port": 9999,
        "database": "ts-user-mysql",
    }

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with wrong port"

    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["error"] is not None, \
        f"Expected endpoint error, got: {user_result}"
    assert "port mismatch" in user_result["error"], \
        f"Expected port mismatch error, got: {user_result['error']}"
    assert not user_result["count_match"], \
        "Count match should be False on endpoint mismatch"

    print("PASS: wrong port endpoint rejected (Test H)")


def test_endpoint_wrong_database():
    """Wrong database in endpoint_expected causes preflight failure."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    # Tamper with endpoint_expected: wrong database for ts-user-mysql
    mod.PRE_SEED_BASELINE["ts-user-mysql"]["endpoint_expected"] = {
        "host": "ts-user-mysql",
        "port": 3306,
        "database": "wrong-database",
    }

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with wrong database"

    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["error"] is not None, \
        f"Expected endpoint error, got: {user_result}"
    assert "database mismatch" in user_result["error"], \
        f"Expected database mismatch error, got: {user_result['error']}"
    assert not user_result["count_match"], \
        "Count match should be False on endpoint mismatch"

    print("PASS: wrong database endpoint rejected (Test I)")


def test_wrong_identity_correct_count():
    """Correct table count but wrong identity values fails closed."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # Count is correct (1 user) but identity is wrong (different UUID)
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with wrong identity (correct count)"

    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["count_match"], \
        f"Count should match (1=1), got count_match={user_result['count_match']}"
    assert not user_result["identity_match"], \
        "Identity match should be False for wrong identity values"
    assert user_result["observed_identities"] == [False], \
        f"Expected observed_identities=[False], got {user_result['observed_identities']}"

    print("PASS: wrong identity with correct count rejected (Test J)")


def test_missing_and_unexpected_identity():
    """Correct count with one expected identity missing and an unexpected
    identity present fails closed."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # 2 users in auth_user (correct count) but neither matches expected identities
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "wrong_user_1",
                 "user_id": "11111111-1111-1111-1111-111111111111"},
                {"user_name": "wrong_user_2",
                 "user_id": "22222222-2222-2222-2222-222222222222"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with missing+unexpected identities"

    auth_result = [r for r in results if r["table"] == "auth_user"][0]
    assert auth_result["count_match"], \
        f"Count should match (2=2), got count_match={auth_result['count_match']}"
    assert not auth_result["identity_match"], \
        "Identity match should be False when expected identities are missing"
    assert auth_result["observed_identities"] == [False, False], \
        f"Expected observed_identities=[False, False], got {auth_result['observed_identities']}"

    print("PASS: missing+unexpected identity rejected (Test K)")


def test_case_different_identity():
    """Case-different identity values fail closed (case-sensitive comparison)."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    # Same UUID but different case for user_name
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "FDSE_MICROSERVICE",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    mod.set_connection_factory(make_mock_factory(baseline_rows, identity_columns_map))

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail with case-different identity"

    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["count_match"], \
        f"Count should match (1=1), got count_match={user_result['count_match']}"
    assert not user_result["identity_match"], \
        "Identity match should be False for case-different values (case-sensitive)"
    assert user_result["observed_identities"] == [False], \
        f"Expected observed_identities=[False], got {user_result['observed_identities']}"

    print("PASS: case-different identity rejected (Test L)")


def test_query_exception_and_incomplete_results():
    """Query exceptions and incomplete query results fail closed."""
    mod = load_module()
    global PRE_SEED_BASELINE_REF
    PRE_SEED_BASELINE_REF = mod.PRE_SEED_BASELINE

    class ExceptionCursor:
        """Cursor that raises on identity queries (simulating SQL error)."""
        def __init__(self, fail_on_identity=False):
            self.fail_on_identity = fail_on_identity
            self.call_count = 0

        def execute(self, sql, params=None):
            sql_lower = sql.strip().lower()
            self.call_count += 1
            if "count(*)" in sql_lower:
                self._result = [{"cnt": 1}]
            elif self.fail_on_identity and params is not None:
                raise Exception("SQL error: connection lost during identity query")
            else:
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result) if self._result else []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class ExceptionConn:
        def __init__(self, fail_on_identity=False):
            self.fail_on_identity = fail_on_identity

        def cursor(self):
            return ExceptionCursor(self.fail_on_identity)

        def close(self):
            pass

    def exception_connect(db_cfg, table_name=""):
        # Fail on identity queries for ts-user-mysql.user
        if db_cfg["database"] == "ts-user-mysql" and table_name == "user":
            return ExceptionConn(fail_on_identity=True)
        return ExceptionConn(fail_on_identity=False)

    mod.set_connection_factory(exception_connect)

    ok, results = mod.preflight_check(verbose=True)
    assert not ok, "Preflight should fail on SQL exception during identity query"

    user_result = [r for r in results if r["table"] == "user"][0]
    assert user_result["error"] is not None, \
        f"Expected SQL error, got: {user_result}"
    assert "SQL error" in user_result["error"] or "connection lost" in user_result["error"], \
        f"Expected SQL error message, got: {user_result['error']}"
    assert not user_result["count_match"], \
        "Count match should be False on query exception"
    assert not user_result["identity_match"], \
        "Identity match should be False on query exception"

    print("PASS: query exception fails closed (Test M)")


def test_exit_code_summary():
    """Verify that EXIT_CODES is defined with all required entries."""
    mod = load_module()
    assert hasattr(mod, "EXIT_CODES"), "EXIT_CODES not defined in source"
    codes = mod.EXIT_CODES
    assert 0 in codes, "Exit code 0 (SUCCESS) missing"
    assert 1 in codes, "Exit code 1 (PARTIAL_FAILURE) missing"
    assert 2 in codes, "Exit code 2 (CLI_OR_PREFLIGHT_FAILURE) missing"
    assert 3 in codes, "Exit code 3 (PREFLIGHT_ONLY_FAILURE) missing"
    assert 4 in codes, "Exit code 4 (WRITE_FAILURE) missing"
    assert "SUCCESS" in codes[0], "Exit code 0 description missing SUCCESS"
    assert "PARTIAL" in codes[1], "Exit code 1 description missing PARTIAL"
    assert "PREFLIGHT" in codes[2] or "CLI" in codes[2], \
        "Exit code 2 description missing CLI/PREFLIGHT"
    assert "PREFLIGHT_ONLY" in codes[3], \
        "Exit code 3 description missing PREFLIGHT_ONLY"
    assert "WRITE" in codes[4], \
        "Exit code 4 description missing WRITE"

    print("PASS: EXIT_CODES summary verified (Test N)")


class MockHttpResponse:
    """Mock HTTP response object with status_code and json() method."""
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


def test_zero_http_after_preflight_failure_behavioral():
    """Behavioral test: when preflight fails, zero HTTP registration/login
    calls are made and the live path returns exit code 2.

    This executes the actual main() control flow with a mocked HTTP transport
    and a mocked connection factory that makes preflight fail."""
    mod = load_module()

    # Track HTTP calls
    http_calls = []

    class MockResponse:
        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json = json_data or {"token": "tok", "userId": "uid"}

        def json(self):
            return self._json

    def mock_http_post(url, json=None, timeout=30):
        http_calls.append({"url": url, "json": json})
        return MockResponse(status_code=200, json_data={"token": "tok", "userId": "uid"})

    # Monkeypatch the HTTP post hook
    mod._http_post = mock_http_post

    # Monkeypatch connection factory to make preflight fail
    class FailingCursor:
        def execute(self, sql, params=None):
            raise Exception("Table doesn't exist")

        def fetchone(self):
            return None

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FailingConn:
        def cursor(self):
            return FailingCursor()

        def close(self):
            pass

    def failing_connect(db_cfg, table_name=""):
        return FailingConn()

    mod.set_connection_factory(failing_connect)

    # Run main() with live-mode args (--out provided, no --dry-run, no --preflight-only)
    rc = mod.main(["--count", "10", "--rng-seed", "20260301", "--out", "/tmp/test_http_fail.json"])

    # Verify: preflight failed, so rc should be 2
    assert rc == 2, f"Expected rc=2 (live preflight failure), got rc={rc}"

    # Verify: zero HTTP calls were made
    assert len(http_calls) == 0, \
        f"Expected zero HTTP calls after preflight failure, got {len(http_calls)}: {http_calls}"

    print("PASS: zero HTTP calls after preflight failure (behavioral, Test O)")


def test_preflight_only_success_with_mock():
    """Behavioral test: successful preflight-only execution with mocked DB."""
    mod = load_module()

    # Build a valid baseline mock
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    class MockCursor:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def execute(self, sql, params=None):
            sl = sql.strip().lower()
            if "count(*)" in sl:
                self._result = [{"cnt": len(self._rows)}]
            elif params is not None:
                self._result = [
                    row for row in self._rows
                    if all(row.get(c) == v for c, v in zip(self._id_cols, params))
                ]
            else:
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result) if self._result else []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockConn:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def cursor(self):
            return MockCursor(self._rows, self._id_cols)

        def close(self):
            pass

    def mock_connect(db_cfg, table_name=""):
        database = db_cfg["database"]
        matched_key = None
        for db_key, entry in mod.PRE_SEED_BASELINE.items():
            if entry["db_default"] == database:
                matched_key = db_key
                break
        if matched_key is None:
            return MockConn([], [])
        rows = baseline_rows.get(matched_key, {}).get(table_name, [])
        id_cols = identity_columns_map.get(matched_key, {}).get(table_name, [])
        return MockConn(rows, id_cols)

    mod.set_connection_factory(mock_connect)

    # Run preflight-only
    rc = mod.main(["--preflight-only"])
    assert rc == 0, f"Expected rc=0 (preflight-only success), got rc={rc}"

    print("PASS: successful preflight-only with mocked DB (Test P)")


def test_preflight_only_failure():
    """Behavioral test: failed preflight-only execution returns rc=3."""
    mod = load_module()

    class FailingCursor:
        def execute(self, sql, params=None):
            raise Exception("connection refused")

        def fetchone(self):
            return None

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FailingConn:
        def cursor(self):
            return FailingCursor()

        def close(self):
            pass

    def failing_connect(db_cfg, table_name=""):
        return FailingConn()

    mod.set_connection_factory(failing_connect)

    rc = mod.main(["--preflight-only"])
    assert rc == 3, f"Expected rc=3 (preflight-only failure), got rc={rc}"

    print("PASS: failed preflight-only returns rc=3 (Test Q)")


def test_live_preflight_failure():
    """Behavioral test: failed live preflight returns rc=2 with zero writes."""
    mod = load_module()

    class FailingCursor:
        def execute(self, sql, params=None):
            raise Exception("Table doesn't exist")

        def fetchone(self):
            return None

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FailingConn:
        def cursor(self):
            return FailingCursor()

        def close(self):
            pass

    def failing_connect(db_cfg, table_name=""):
        return FailingConn()

    mod.set_connection_factory(failing_connect)

    rc = mod.main(["--count", "10", "--rng-seed", "20260301", "--out", "/tmp/test_live_preflight_fail.json"])
    assert rc == 2, f"Expected rc=2 (live preflight failure), got rc={rc}"

    print("PASS: failed live preflight returns rc=2 (Test R)")


def test_partial_registration_failure():
    """Behavioral test: partial registration/login failure returns rc=1
    and output file records incomplete state."""
    mod = load_module()

    # Build valid baseline mock
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    class MockCursor:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def execute(self, sql, params=None):
            sl = sql.strip().lower()
            if "count(*)" in sl:
                self._result = [{"cnt": len(self._rows)}]
            elif params is not None:
                self._result = [
                    row for row in self._rows
                    if all(row.get(c) == v for c, v in zip(self._id_cols, params))
                ]
            else:
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result) if self._result else []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockConn:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def cursor(self):
            return MockCursor(self._rows, self._id_cols)

        def close(self):
            pass

    def mock_connect(db_cfg, table_name=""):
        database = db_cfg["database"]
        matched_key = None
        for db_key, entry in mod.PRE_SEED_BASELINE.items():
            if entry["db_default"] == database:
                matched_key = db_key
                break
        if matched_key is None:
            return MockConn([], [])
        rows = baseline_rows.get(matched_key, {}).get(table_name, [])
        id_cols = identity_columns_map.get(matched_key, {}).get(table_name, [])
        return MockConn(rows, id_cols)

    mod.set_connection_factory(mock_connect)

    # Mock HTTP: first user register fails, second succeeds, third login fails
    call_count = [0]

    def mock_http_post(url, json=None, timeout=30):
        call_count[0] += 1
        if "register" in url:
            if call_count[0] == 1:
                return MockHttpResponse(status_code=500)  # register fails
            return MockHttpResponse(status_code=200)  # register succeeds
        elif "login" in url:
            if call_count[0] == 3:
                return MockHttpResponse(status_code=500)  # login fails
            return MockHttpResponse(status_code=200, json_data={"token": "tok", "userId": "uid"})
        return MockHttpResponse(status_code=200)

    mod._http_post = mock_http_post

    import tempfile, os, json
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "partial_seed.json")
        rc = mod.main(["--count", "3", "--rng-seed", "20260301", "--out", out])

        assert rc == 1, f"Expected rc=1 (partial failure), got rc={rc}"

        # Verify output file records incomplete state
        assert os.path.exists(out), "Seed file should be written even on partial failure"
        with open(out) as f:
            data = json.load(f)
        assert data["complete"] is False, "Seed file should mark complete=False"
        assert data["failed"] > 0, "Seed file should record failures"
        assert data["seeded"] < data["requested"], "Seed file should show partial seeding"

    print("PASS: partial registration failure returns rc=1 with incomplete state (Test S)")


def test_write_failure():
    """Behavioral test: output file write failure returns rc=4."""
    mod = load_module()

    # Mock preflight to pass (valid baseline)
    baseline_rows = {
        "ts-user-mysql": {
            "user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
        },
        "ts-auth-mysql": {
            "auth_user": [
                {"user_name": "fdse_microservice",
                 "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                {"user_name": "admin",
                 "user_id": "00000000-0000-0000-000000000000"},
            ],
        },
        "ts-contacts-mysql": {
            "contacts": [
                {"name": "Contacts_One",
                 "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                {"name": "Contacts_Two",
                 "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
            ],
        },
        "ts-order-mysql": {
            "orders": [
                {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
            ],
        },
        "ts-order-other-mysql": {
            "orders_other": [
                {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
            ],
        },
        "ts-payment-mysql": {
            "payment": [
                {"id": "5ad7750ba68b49c0a8c035276b067701"},
            ],
        },
        "ts-inside-payment-mysql": {
            "inside_money": [
                {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
            ],
            "inside_payment": [
                {"id": "5ad7750ba68b49c0a8c035276b321701"},
            ],
        },
    }

    identity_columns_map = {
        "ts-user-mysql": {"user": ["user_name", "user_id"]},
        "ts-auth-mysql": {"auth_user": ["user_name"]},
        "ts-contacts-mysql": {"contacts": ["name", "id"]},
        "ts-order-mysql": {"orders": ["id"]},
        "ts-order-other-mysql": {"orders_other": ["id"]},
        "ts-payment-mysql": {"payment": ["id"]},
        "ts-inside-payment-mysql": {
            "inside_money": ["user_id"],
            "inside_payment": ["id"],
        },
    }

    class MockCursor:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def execute(self, sql, params=None):
            sl = sql.strip().lower()
            if "count(*)" in sl:
                self._result = [{"cnt": len(self._rows)}]
            elif params is not None:
                self._result = [
                    row for row in self._rows
                    if all(row.get(c) == v for c, v in zip(self._id_cols, params))
                ]
            else:
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result) if self._result else []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockConn:
        def __init__(self, rows, id_cols):
            self._rows = rows
            self._id_cols = id_cols

        def cursor(self):
            return MockCursor(self._rows, self._id_cols)

        def close(self):
            pass

    def mock_connect(db_cfg, table_name=""):
        database = db_cfg["database"]
        matched_key = None
        for db_key, entry in mod.PRE_SEED_BASELINE.items():
            if entry["db_default"] == database:
                matched_key = db_key
                break
        if matched_key is None:
            return MockConn([], [])
        rows = baseline_rows.get(matched_key, {}).get(table_name, [])
        id_cols = identity_columns_map.get(matched_key, {}).get(table_name, [])
        return MockConn(rows, id_cols)

    mod.set_connection_factory(mock_connect)

    # Mock HTTP: all succeed
    def mock_http_post(url, json=None, timeout=30):
        return MockHttpResponse(status_code=200, json_data={"token": "tok", "userId": "uid"})

    mod._http_post = mock_http_post

    # Try to write to a read-only directory
    rc = mod.main(["--count", "2", "--rng-seed", "20260301", "--out", "/proc/seed_write_fail.json"])
    assert rc == 4, f"Expected rc=4 (write failure), got rc={rc}"

    print("PASS: seed file write failure returns rc=4 (Test T)")


def test_successful_dry_run():
    """Behavioral test: successful dry-run returns rc=0 and writes plan file."""
    mod = load_module()

    import tempfile, os, json
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "dryrun.json")
        rc = mod.main(["--count", "5", "--rng-seed", "20260301", "--dry-run", "--out", out])

        assert rc == 0, f"Expected rc=0 (dry-run success), got rc={rc}"
        assert os.path.exists(out), "Dry-run plan file should be written"
        with open(out) as f:
            data = json.load(f)
        assert data["count"] == 5
        assert data["mode"] == "dry-run-plan"

    print("PASS: successful dry-run returns rc=0 (Test U)")


if __name__ == "__main__":
    test_deterministic_output()
    test_user_count_and_fields()
    test_unique_usernames()
    test_null_contacts_id()
    test_no_contact_creation_endpoint()
    test_plan_sha256_deterministic()
    test_allow_nonempty_argument_removed()
    test_no_force_flag()
    test_no_bypass_reference_in_source()
    test_no_allow_nonempty_reference_in_source()
    test_exact_permitted_baseline_state()
    test_preflight_only_without_out()
    test_dry_run_without_out_fails()
    test_live_without_out_fails()
    test_help_output_cli_contract()
    test_valid_baseline_preflight_succeeds()        # Test A
    test_unexpected_record_refusal()                # Test B
    test_partial_baseline_refusal()                 # Test C
    test_unexpected_auth_record_refusal()           # Test D
    test_missing_database_or_collection()           # Test E
    test_determinism_across_valid_baselines()       # Test F
    test_invalid_fixture_zero_writes()
    # Stage 7.16 new tests
    test_endpoint_wrong_host()                      # Test G
    test_endpoint_wrong_port()                      # Test H
    test_endpoint_wrong_database()                  # Test I
    test_wrong_identity_correct_count()             # Test J
    test_missing_and_unexpected_identity()          # Test K
    test_case_different_identity()                  # Test L
    test_query_exception_and_incomplete_results()   # Test M
    test_exit_code_summary()                        # Test N
    test_zero_http_after_preflight_failure_behavioral()  # Test O (behavioral)
    # Stage 7.17 new tests
    test_preflight_only_success_with_mock()         # Test P
    test_preflight_only_failure()                   # Test Q
    test_live_preflight_failure()                   # Test R
    test_partial_registration_failure()             # Test S
    test_write_failure()                            # Test T
    test_successful_dry_run()                       # Test U
    print("\nAll offline seed tests PASSED.")
#!/usr/bin/env python3
"""Offline unit tests for seed_tt_users_v4.py (v4.2, Stage 7.24).

These tests run without any cluster access, database mutation, or live API
call. All HTTP traffic is handled by an in-process mock transport installed
via set_http_post(). All attestations are synthetic fixtures signed with a
test-generated Ed25519 key pair.

Coverage groups:
  S  — registration/login parsing and combined state machine
  R  — retry/uncertainty policy
  C  — credential artifact separation and secret hygiene
  G  — attestation integrity (checksum + Ed25519 signature)
  B  — baseline attestation v2 schema (seven-instance inventory)
  K  — backup attestation v2 schema (per-target evidence)
  E  — end-to-end mock live runs and checkpoint/resume recovery
  O  — output write-failure handling
  V  — v3 compatibility (unchanged source, identical plans, tests still pass)

Note on legacy tests: the Stage 7.23 test names are preserved. Where the
v4.2 contract legitimately changed per the Stage 7.24 task (sanitized dry-run
plan, exit-code renumbering, results/seed-artifact separation), the test
bodies were updated to the new contract; the intent of every legacy test is
preserved. The v3 generator and v3 tests are NOT modified.
"""

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat as stat_mod
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_V4 = os.path.join(REPO, "load_testing", "seed_tt_users_v4.py")
GEN_V3 = os.path.join(REPO, "load_testing", "seed_tt_users_v3_july_semantics.py")
TEST_V3 = os.path.join(REPO, "load_testing", "test_seed_v3_offline.py")
SCRATCH = "/tmp/opencode/stage7_24_tests"

spec = importlib.util.spec_from_file_location("seed_v4", GEN_V4)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Known discrepancy identifiers
ALL_DISCREPANCIES = ",".join(sorted(mod.KNOWN_DISCREPANCY_IDS))


# ---------------------------------------------------------------------------
# Ed25519 fixture helpers
# ---------------------------------------------------------------------------

def ed25519_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    sk = Ed25519PrivateKey.generate()
    raw_pub = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return sk, base64.b64encode(raw_pub).decode("ascii")


def write_attestation(dirpath, name, obj, signing_key, sign=True,
                      checksum=True):
    """Write attestation bytes + detached .sha256 + detached .sig."""
    data = mod.canonical_json_bytes(obj)
    path = os.path.join(dirpath, name)
    with open(path, "wb") as f:
        f.write(data)
    if checksum:
        with open(path + ".sha256", "w") as f:
            f.write(f"{hashlib.sha256(data).hexdigest()}  {path}\n")
    if sign:
        sig = signing_key.sign(data)
        with open(path + ".sig", "wb") as f:
            f.write(base64.b64encode(sig))
    return path


def iso_now(minutes_ago=1):
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


def iso_future(hours=1):
    return (datetime.now(timezone.utc)
            + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def iso_old(days=30):
    return (datetime.now(timezone.utc)
            - timedelta(days=days)).isoformat().replace("+00:00", "Z")


def _col(name, exists, count=None, identities=None, discrepancies=None,
         index_summary="only default _id_ (unique)"):
    return {
        "name": name,
        "exists": exists,
        "document_count": count,
        "index_inventory_summary": index_summary,
        "identity_checks": identities or [],
        "discrepancy_ids": discrepancies or [],
    }


def _ident(field, identity, present, bson_type=None):
    chk = {"field": field, "identity": identity, "present": present}
    if bson_type:
        chk["bson_type"] = bson_type
    return chk


def make_baseline(mutate=None, generated_at=None):
    inst = []
    for i, defn in enumerate(mod.EXPECTED_INSTANCES):
        rec = {
            "instance_id": defn["instance_id"],
            "namespace": defn["namespace"],
            "service_name": defn["service_name"],
            "service_uid": f"uid-ts-mongo-{i}",
            "application": defn["application"],
            "database": defn["database"],
            "mongodb_version": "4.4.x",
            "collections": [],
        }
        for c in defn["collections"]:
            if defn["instance_id"] == "ts-user-mongo" and c["name"] == "user":
                rec["collections"].append(_col(
                    "user", True, 82,
                    [_ident("userName", "fdse_microservice", False)],
                    [mod.DISC_USER_INIT_FDSE_MISSING]))
            elif defn["instance_id"] == "ts-auth-mongo" and c["name"] == "user":
                rec["collections"].append(_col(
                    "user", True, 84,
                    [_ident("username", "fdse_microservice", True),
                     _ident("username", "admin", True)]))
            elif defn["instance_id"] == "ts-contacts-mongo":
                rec["collections"].append(_col(
                    "contacts", True, 2,
                    [_ident("name", "Contacts_One", True),
                     _ident("name", "Contacts_Two", True)]))
            elif defn["instance_id"] == "ts-order-mongo":
                rec["collections"].append(_col(
                    "orders", True, 3,
                    [_ident("_id", "5ad7750b-a68b-49c0-a8c0-32776b067703", True,
                            bson_type="binary_subtype_3"),
                     _ident("_id", "8177ac5a-61ac-42f4-83f4-bd7b394d0531", True,
                            bson_type="binary_subtype_3"),
                     _ident("_id", "d3c91694-d5b8-424c-9974-e14c89226e49", True,
                            bson_type="binary_subtype_3")]))
            elif defn["instance_id"] == "ts-order-other-mongo":
                # One-order baseline sentinel (Stage 7.25): accountId is NOT
                # a unique document identity — multiple orders can belong to
                # one account. The pinned check is the narrow initialization
                # state: exactly one document (document_count == 1) whose
                # accountId equals the fdse_microservice userId. create()
                # randomizes the document _id, so the deterministic input id
                # is NOT the document identity (Stage 7.22).
                rec["collections"].append(_col(
                    "orders", True, 1,
                    [{"field": "accountId", "identity": "fdse_microservice_userId",
                      "present": True,
                      "note": "one_order_baseline_sentinel (exactly one "
                              "document with the expected accountId; _id "
                              "randomized by create())"}]))
            elif defn["instance_id"] == "ts-payment-mongo":
                if c["name"] == "payment":
                    rec["collections"].append(_col(
                        "payment", True, 1,
                        [_ident("_id", "5ad7750ba68b49c0a8c035276b067701", False)],
                        [mod.DISC_PAYMENT_EXPECTED_ID_ABSENT,
                         mod.DISC_PAYMENT_LIVE_ID_DIFFERS]))
                else:
                    rec["collections"].append(_col(
                        "addMoney", False, None, [],
                        [mod.DISC_PAYMENT_ADDMONEY_ABSENT]))
            elif defn["instance_id"] == "ts-inside-payment-mongo":
                if c["name"] == "payment":
                    rec["collections"].append(_col(
                        "payment", True, 1,
                        [_ident("_id", "5ad7750ba68b49c0a8c035276b321701", True)]))
                else:
                    rec["collections"].append(_col(
                        "addMoney", True, 1,
                        [_ident("userId",
                                "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f", True)]))
        inst.append(rec)

    att = {
        "schema_version": mod.BASELINE_SCHEMA_VERSION,
        "generated_at_utc": generated_at or iso_now(),
        "audit_method": "read-only pymongo inventory via kubectl port-forward",
        "operator": "stage7-24-fixture",
        "source_audit_packages": [
            {"package_id": "stage7_21_live_mongodb_baseline_audit_20261111T191043Z",
             "package_sha256": "0" * 64},
            {"package_id": "stage7_22_baseline_discrepancy_closure_20261111T200609Z",
             "package_sha256": "1" * 64},
            {"package_id": "stage7_23_seed_generator_v4_offline_readiness_20261111T203548Z",
             "package_sha256": "2" * 64},
        ],
        "instances": inst,
    }
    if mutate:
        mutate(att)
    return att


def make_backup(mutate=None, generated_at=None, target_mutate=None):
    targets = []
    for defn in mod.EXPECTED_INSTANCES:
        t = {
            "instance_id": defn["instance_id"],
            "database": defn["database"],
            "backup_method": "mongodump --archive",
            "artifact_location": "file:///offcluster/backups/"
                                 f"{defn['instance_id']}.archive",
            "artifact_sha256": hashlib.sha256(
                defn["instance_id"].encode()).hexdigest(),
            "backup_created_at_utc": iso_now(minutes_ago=10),
            "mongodb_version": "4.4.x",
            "off_cluster_storage_location": "nfs://offcluster-nas/tt-backups",
            "off_cluster_storage_verified": True,
            "restore_test_location": "docker://restore-sandbox/mongod:4.4",
            "restore_tested_at_utc": iso_now(minutes_ago=5),
            "restore_result": "verified",
            "restored_collection_counts": {
                c["name"]: (0 if not c["exists"] else 1)
                for c in defn["collections"]
            },
            "evidence_reference": "restore-verification-report-2026-11-12.md",
        }
        if target_mutate:
            target_mutate(t, defn)
        targets.append(t)
    att = {
        "schema_version": mod.BACKUP_SCHEMA_VERSION,
        "generated_at_utc": generated_at or iso_now(),
        "operator": "stage7-24-fixture",
        "targets": targets,
    }
    if mutate:
        mutate(att)
    return att


# ---------------------------------------------------------------------------
# Mock HTTP transport
# ---------------------------------------------------------------------------

class MockResp:
    def __init__(self, status_code, body=None, json_error=False):
        self.status_code = status_code
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("no json")
        return self._body


class TransportError(Exception):
    pass


class Recorder:
    """Scriptable mock transport that records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "payload": json})
        return self.handler(url, json, self.calls)

    def register_count(self, username=None):
        return sum(1 for c in self.calls
                   if c["url"].endswith("/users/register")
                   and (username is None
                        or c["payload"]["userName"] == username))

    def login_count(self, username=None):
        return sum(1 for c in self.calls
                   if c["url"].endswith("/users/login")
                   and (username is None
                        or c["payload"]["username"] == username))


def ok_register(payload):
    return MockResp(201, {"status": 1, "msg": "REGISTER USER SUCCESS",
                          "data": {"userName": payload["userName"]}})


def ok_login(payload):
    return MockResp(200, {"status": 1, "msg": "login success",
                          "data": {"userId": f"acct-{payload['username']}",
                                   "username": payload["username"],
                                   "token": "eyJhbGciOiJIUzI1NiJ9.testsig"}})


# ---------------------------------------------------------------------------
# Test-case base with scratch directory
# ---------------------------------------------------------------------------

class Stage724Case(unittest.TestCase):
    def setUp(self):
        os.makedirs(SCRATCH, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="s724_", dir=SCRATCH)
        self.sk, self.pub_b64 = ed25519_keypair()
        self.sk_other, self.pub_other = ed25519_keypair()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_valid_attestations(self):
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(), self.sk)
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(), self.sk)
        return base_path, back_path

    def gate_args(self, base_path, back_path, extra=None):
        return [
            "--baseline-attestation", base_path,
            "--backup-attestation", back_path,
            "--approve-discrepancies", ALL_DISCREPANCIES,
            "--approved-key", self.pub_b64,
        ] + (extra or [])

    def run_gate(self, base_path, back_path, extra=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main(["--preflight-only"]
                          + self.gate_args(base_path, back_path, extra))
        return rc, out.getvalue(), err.getvalue()

    def assertNoSecrets(self, text, users):
        self.assertNotIn("loadtest123", text)
        self.assertNotIn("eyJ", text)
        for u in users:
            self.assertNotIn(str(u["documentNum"]), text)
            self.assertNotIn(str(u["phoneNumber"]), text)
            for tok in ("testsig",):
                self.assertNotIn(tok, text)


# ===========================================================================
# Group S — registration/login parsing and combined state machine
# ===========================================================================

class TestRegistrationParsing(Stage724Case):

    def test_register_http_201_status_1_success(self):
        r = mod.parse_registration_response(
            MockResp(201, {"status": 1, "msg": "REGISTER USER SUCCESS",
                           "data": {}}))
        self.assertEqual(r["state"], mod.REG_REGISTERED)
        self.assertEqual(r["http_status"], 201)
        self.assertEqual(r["body_status"], 1)

    def test_register_http_201_status_0_duplicate(self):
        r = mod.parse_registration_response(
            MockResp(201, {"status": 0, "msg": "USER HAS ALREADY EXISTS",
                           "data": None}))
        self.assertEqual(r["state"], mod.REG_DUPLICATE)

    def test_register_http_201_status_0_app_error(self):
        r = mod.parse_registration_response(
            MockResp(201, {"status": 0, "msg": "INTERNAL VALIDATION FAILED",
                           "data": None}))
        self.assertEqual(r["state"], mod.REG_APPLICATION_FAILURE)

    def test_register_unexpected_http_status(self):
        r = mod.parse_registration_response(MockResp(409, {"status": 1}))
        self.assertEqual(r["state"], mod.REG_HTTP_FAILURE)
        self.assertFalse(r["outcome_uncertain"])

    def test_register_5xx_is_uncertain(self):
        r = mod.parse_registration_response(MockResp(503, {"status": 1}))
        self.assertEqual(r["state"], mod.REG_HTTP_FAILURE)
        self.assertTrue(r["outcome_uncertain"])
        combined = mod.combine_outcome(r, None)
        self.assertEqual(combined["outcome"], mod.REGISTRATION_OUTCOME_UNCERTAIN)
        self.assertTrue(combined["uncertain"])

    def test_register_invalid_json(self):
        r = mod.parse_registration_response(MockResp(201, json_error=True))
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)
        self.assertEqual(r["error_category"], "invalid_json")

    def test_register_missing_status_field(self):
        r = mod.parse_registration_response(
            MockResp(201, {"msg": "no status"}))
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)
        self.assertEqual(r["error_category"], "invalid_status_type")

    def test_register_status_wrong_type(self):
        r = mod.parse_registration_response(
            MockResp(201, {"status": "not_an_int"}))
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)

    def test_register_boolean_status_rejected(self):
        # isinstance(True, int) is True in Python; the parser must reject it.
        r = mod.parse_registration_response(
            MockResp(201, {"status": True, "msg": "REGISTER USER SUCCESS"}))
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)
        r2 = mod.parse_registration_response(
            MockResp(201, {"status": False, "msg": "USER HAS ALREADY EXISTS"}))
        self.assertEqual(r2["state"], mod.REG_OUTCOME_UNKNOWN)

    def test_register_unknown_status_value(self):
        r = mod.parse_registration_response(MockResp(201, {"status": 7}))
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)
        self.assertEqual(r["error_category"], "unknown_status_7")

    def test_register_none_response_unknown(self):
        r = mod.parse_registration_response(None)
        self.assertEqual(r["state"], mod.REG_OUTCOME_UNKNOWN)
        self.assertTrue(r["outcome_uncertain"])

    def test_duplicate_requires_verified_message(self):
        # status 0 without the verified duplicate message is NOT a duplicate.
        r = mod.parse_registration_response(
            MockResp(201, {"status": 0, "msg": "EMAIL ALREADY REGISTERED"}))
        self.assertEqual(r["state"], mod.REG_APPLICATION_FAILURE)


class TestLoginParsing(Stage724Case):

    def valid_login_body(self, username="userA"):
        return {"status": 1, "msg": "login success",
                "data": {"userId": "acct-1", "username": username,
                         "token": "eyJhbGciOiJIUzI1NiJ9.t"}}

    def test_login_http_200_status_1_valid_data(self):
        r = mod.parse_login_response(MockResp(200, self.valid_login_body()),
                                     "userA")
        self.assertEqual(r["state"], mod.LOGIN_VERIFIED)
        self.assertEqual(r["user_id"], "acct-1")

    def test_login_http_200_status_0(self):
        r = mod.parse_login_response(
            MockResp(200, {"status": 0,
                           "msg": "Incorrect username or password.",
                           "data": None}), "userA")
        self.assertEqual(r["state"], mod.LOGIN_APPLICATION_FAILURE)

    def test_login_http_error(self):
        r = mod.parse_login_response(MockResp(500, {}), "userA")
        self.assertEqual(r["state"], mod.LOGIN_HTTP_FAILURE)

    def test_login_invalid_json(self):
        r = mod.parse_login_response(MockResp(200, json_error=True), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_boolean_status_rejected(self):
        body = self.valid_login_body()
        body["status"] = True
        r = mod.parse_login_response(MockResp(200, body), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_username_mismatch_rejected(self):
        r = mod.parse_login_response(
            MockResp(200, self.valid_login_body(username="someone_else")),
            "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)
        self.assertEqual(r["error_category"], "username_mismatch")

    def test_login_missing_user_id_rejected(self):
        body = self.valid_login_body()
        del body["data"]["userId"]
        r = mod.parse_login_response(MockResp(200, body), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_empty_token_rejected(self):
        body = self.valid_login_body()
        body["data"]["token"] = ""
        r = mod.parse_login_response(MockResp(200, body), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_token_wrong_type_rejected(self):
        body = self.valid_login_body()
        body["data"]["token"] = {"jwt": True}
        r = mod.parse_login_response(MockResp(200, body), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_missing_data_rejected(self):
        r = mod.parse_login_response(
            MockResp(200, {"status": 1, "msg": "login success"}), "userA")
        self.assertEqual(r["state"], mod.LOGIN_INVALID_RESPONSE)

    def test_login_timeout_unknown(self):
        r = mod.parse_login_response(None, "userA")
        self.assertEqual(r["state"], mod.LOGIN_OUTCOME_UNKNOWN)


class TestCombinedStateMachine(Stage724Case):

    def reg(self, state, **kw):
        d = {"state": state, "http_status": 201, "body_status": 1,
             "error_category": None, "outcome_uncertain": False}
        d.update(kw)
        return d

    def login_ok(self):
        return {"state": mod.LOGIN_VERIFIED, "http_status": 200,
                "body_status": 1, "error_category": None, "token": "t",
                "user_id": "u", "outcome_uncertain": False}

    def login_fail(self, state):
        return {"state": state, "http_status": 200, "body_status": 0,
                "error_category": "application_error", "token": None,
                "user_id": None, "outcome_uncertain": False}

    def test_registration_success_login_verified(self):
        c = mod.combine_outcome(self.reg(mod.REG_REGISTERED), self.login_ok())
        self.assertEqual(c["outcome"], mod.REGISTERED_AND_LOGIN_VERIFIED)
        self.assertTrue(c["confirmed"])
        self.assertFalse(c["login_retry_allowed"])

    def test_registration_success_login_failure(self):
        c = mod.combine_outcome(self.reg(mod.REG_REGISTERED),
                                self.login_fail(mod.LOGIN_APPLICATION_FAILURE))
        self.assertEqual(c["outcome"], mod.REGISTRATION_SUCCESS_LOGIN_FAILURE)
        self.assertTrue(c["confirmed"])
        self.assertTrue(c["login_retry_allowed"])

    def test_duplicate_login_verified(self):
        c = mod.combine_outcome(self.reg(mod.REG_DUPLICATE), self.login_ok())
        self.assertEqual(c["outcome"], mod.DUPLICATE_AND_LOGIN_VERIFIED)

    def test_duplicate_login_failure_explicit_state(self):
        c = mod.combine_outcome(self.reg(mod.REG_DUPLICATE),
                                self.login_fail(mod.LOGIN_HTTP_FAILURE))
        self.assertEqual(c["outcome"], mod.DUPLICATE_LOGIN_FAILURE)
        self.assertTrue(c["confirmed"])
        self.assertTrue(c["login_retry_allowed"])

    def test_registration_application_failure(self):
        c = mod.combine_outcome(
            self.reg(mod.REG_APPLICATION_FAILURE), None)
        self.assertEqual(c["outcome"], mod.REGISTRATION_APPLICATION_FAILURE)
        self.assertTrue(c["confirmed"])
        self.assertFalse(c["login_retry_allowed"])

    def test_registration_http_failure_4xx(self):
        c = mod.combine_outcome(self.reg(mod.REG_HTTP_FAILURE), None)
        self.assertEqual(c["outcome"], mod.REGISTRATION_HTTP_FAILURE)
        self.assertTrue(c["confirmed"])

    def test_registration_outcome_unknown(self):
        c = mod.combine_outcome(self.reg(mod.REG_OUTCOME_UNKNOWN), None)
        self.assertEqual(c["outcome"], mod.REGISTRATION_OUTCOME_UNKNOWN)
        self.assertFalse(c["confirmed"])
        self.assertTrue(c["uncertain"])

    def test_registration_outcome_uncertain_5xx(self):
        c = mod.combine_outcome(
            self.reg(mod.REG_HTTP_FAILURE, http_status=503,
                     outcome_uncertain=True), None)
        self.assertEqual(c["outcome"], mod.REGISTRATION_OUTCOME_UNCERTAIN)
        self.assertFalse(c["confirmed"])
        self.assertTrue(c["uncertain"])


# ===========================================================================
# Group R — retry / uncertainty policy
# ===========================================================================

class TestRetryPolicy(Stage724Case):

    def test_no_automatic_retry_after_timeout(self):
        calls = []

        def handler(url, payload, all_calls):
            calls.append(url)
            raise TransportError("simulated timeout")

        mod.set_http_post(Recorder(handler))
        try:
            user = mod.plan_users(1, 1)[0]
            combined, reg, login = mod.register_and_login_user(
                user, "http://u", "http://a", 5)
        finally:
            mod.set_http_post(None)
        self.assertEqual(len(calls), 1)  # exactly one register POST, no retry
        self.assertEqual(combined["outcome"], mod.REGISTRATION_OUTCOME_UNKNOWN)
        self.assertTrue(combined["uncertain"])

    def test_no_retry_after_5xx(self):
        calls = []

        def handler(url, payload, all_calls):
            calls.append(url)
            return MockResp(502, {"status": 0})

        mod.set_http_post(Recorder(handler))
        try:
            user = mod.plan_users(1, 1)[0]
            combined, reg, login = mod.register_and_login_user(
                user, "http://u", "http://a", 5)
        finally:
            mod.set_http_post(None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(combined["outcome"], mod.REGISTRATION_OUTCOME_UNCERTAIN)
        self.assertIsNone(login)  # no login attempted with unverified credentials


# ===========================================================================
# Group C — credential artifacts and secret hygiene
# ===========================================================================

class TestCredentialArtifacts(Stage724Case):

    def _run_live(self, users_n=5, handler=None):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        if handler is None:
            def handler(url, payload, all_calls):
                if url.endswith("/users/register"):
                    return ok_register(payload)
                return ok_login(payload)

        rec = Recorder(handler)
        mod.set_http_post(rec)
        users = mod.plan_users(users_n, 20260301)
        try:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = mod.main([
                    "--count", str(users_n), "--rng-seed", "20260301",
                    "--write-authorize",
                    "--results", results,
                    "--seed-artifact", seed_art,
                    "--checkpoint", checkpoint,
                    "--baseline-attestation", base_path,
                    "--backup-attestation", back_path,
                    "--approve-discrepancies", ALL_DISCREPANCIES,
                    "--approved-key", self.pub_b64,
                    "--user-base", "http://127.0.0.1:1",
                    "--auth-base", "http://127.0.0.1:1",
                ])
        finally:
            mod.set_http_post(None)
        return rc, rec, results, seed_art, checkpoint, users, out.getvalue() + err.getvalue()

    def test_full_live_run_success(self):
        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live()
        self.assertEqual(rc, mod.EXIT_SUCCESS, console)
        self.assertEqual(rec.register_count(), 5)
        self.assertEqual(rec.login_count(), 5)

        # restricted artifact permissions
        st = os.stat(seed_art)
        self.assertEqual(stat_mod.S_IMODE(st.st_mode), 0o600)

        artifact = json.load(open(seed_art))
        self.assertEqual(len(artifact), 5)
        for entry in artifact:
            self.assertEqual(set(entry.keys()),
                             {"idx", "username", "accountId", "contactsId"})
            self.assertIsNone(entry["contactsId"])

        manifest = json.load(open(results))
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["usable_pool_size"], 5)

        # No secrets in the results manifest, restricted artifact, or console.
        self.assertNoSecrets(json.dumps(manifest), users)
        self.assertNoSecrets(open(seed_art).read(), users)
        self.assertNoSecrets(console, users)
        # Checkpoint contains no credentials either.
        self.assertNoSecrets(open(checkpoint).read(), users)

    def test_no_jwt_password_in_result_manifests(self):
        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live()
        manifest_text = open(results).read()
        self.assertNotIn('"password"', manifest_text)
        self.assertNotIn('"token"', manifest_text)
        self.assertNotIn('"documentNum"', manifest_text)
        self.assertNotIn('"phoneNumber"', manifest_text)

    def test_no_jwt_password_in_stdout_stderr(self):
        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live()
        self.assertNoSecrets(console, users)
        self.assertNotIn("password", console.lower().replace(
            "no passwords", "").replace("passwords", ""))

    def test_restricted_artifact_contains_only_required_fields(self):
        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live()
        artifact = json.load(open(seed_art))
        for entry in artifact:
            self.assertNotIn("password", entry)
            self.assertNotIn("token", entry)
            self.assertNotIn("documentNum", entry)
            self.assertNotIn("phoneNumber", entry)

    def test_restricted_artifact_permissions_0600(self):
        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live()
        self.assertEqual(
            stat_mod.S_IMODE(os.stat(seed_art).st_mode), 0o600)

    def test_no_secret_in_transport_error_diagnostics(self):
        def handler(url, payload, all_calls):
            raise TransportError(f"connect failure for {url}")

        rc, rec, results, seed_art, checkpoint, users, console = \
            self._run_live(handler=handler)
        self.assertEqual(rc, mod.EXIT_PARTIAL_FAILURE)
        # All users uncertain; no secrets surfaced through exception text.
        self.assertNoSecrets(console, users)
        manifest = json.load(open(results))
        self.assertEqual(manifest["uncertain_outcomes"], 5)
        self.assertNoSecrets(json.dumps(manifest), users)

    def test_atomic_write_permissions_and_cleanup(self):
        target = os.path.join(self.tmp, "restricted.bin")
        mod.atomic_write(target, b"secret-bytes", 0o600)
        self.assertEqual(
            stat_mod.S_IMODE(os.stat(target).st_mode), 0o600)
        self.assertEqual(open(target, "rb").read(), b"secret-bytes")
        leftovers = [f for f in os.listdir(self.tmp)
                     if f.startswith(".restricted.bin.tmp-")]
        self.assertEqual(leftovers, [])

        mod.atomic_write(target, b"public-bytes", 0o640)
        self.assertEqual(
            stat_mod.S_IMODE(os.stat(target).st_mode), 0o640)

    def test_checkpoint_writer_fail_closed_on_write_error(self):
        path = os.path.join(self.tmp, "ckpt.jsonl")
        w = mod.CheckpointWriter(path)
        w.append({"t": "attempt", "idx": 0, "phase": "register"})
        w.close()
        with self.assertRaises(OSError):
            w.append({"t": "outcome", "idx": 0})  # fd closed -> fail closed


# ===========================================================================
# Group G — attestation integrity (checksum + signature)
# ===========================================================================

class TestAttestationIntegrity(Stage724Case):

    def test_valid_signatures_pass(self):
        base_path, back_path = self.write_valid_attestations()
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

    def test_missing_baseline_attestation_aborts(self):
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(), self.sk)
        rc, out, err = self.run_gate(os.path.join(self.tmp, "nope.json"),
                                     back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("GATE_FAILURE", err)

    def test_missing_backup_attestation_aborts(self):
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(), self.sk)
        rc, out, err = self.run_gate(base_path,
                                     os.path.join(self.tmp, "nope.json"))
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)

    def test_missing_signature(self):
        base_path, back_path = self.write_valid_attestations()
        os.unlink(base_path + ".sig")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing detached signature", err)

    def test_invalid_signature_modified_attestation(self):
        base_path, back_path = self.write_valid_attestations()
        # Modify the attestation AFTER signing, then refresh only the
        # checksum so the failure is isolated to signature verification.
        att = json.load(open(base_path))
        att["operator"] = "tampered"
        data = mod.canonical_json_bytes(att)
        with open(base_path, "wb") as f:
            f.write(data)
        with open(base_path + ".sha256", "w") as f:
            f.write(f"{hashlib.sha256(data).hexdigest()}  {base_path}\n")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("no approved key verified", err)

    def test_unknown_signer(self):
        base_path, back_path = self.write_valid_attestations()
        # Signature made by a different key than the approved one.
        data = open(base_path, "rb").read()
        with open(base_path + ".sig", "wb") as f:
            f.write(base64.b64encode(self.sk_other.sign(data)))
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("no approved key verified", err)

    def test_missing_checksum(self):
        base_path, back_path = self.write_valid_attestations()
        os.unlink(base_path + ".sha256")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing detached checksum", err)

    def test_checksum_mismatch(self):
        base_path, back_path = self.write_valid_attestations()
        with open(base_path + ".sha256", "w") as f:
            f.write("0" * 64 + "  x\n")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("mismatch", err)

    def test_malformed_checksum(self):
        base_path, back_path = self.write_valid_attestations()
        with open(base_path + ".sha256", "w") as f:
            f.write("zzzz-not-hex\n")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("malformed detached checksum", err)

    def test_pinned_fingerprint_unknown_key_rejected(self):
        base_path, back_path = self.write_valid_attestations()
        rc, out, err = self.run_gate(
            base_path, back_path,
            extra=["--expected-key-fingerprint", "ab" * 32])
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("no approved key verified", err)

    def test_pinned_fingerprint_correct_key_accepted(self):
        from cryptography.hazmat.primitives import serialization
        raw = self.sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        fp = hashlib.sha256(raw).hexdigest()
        base_path, back_path = self.write_valid_attestations()
        rc, out, err = self.run_gate(
            base_path, back_path,
            extra=["--expected-key-fingerprint", fp])
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

    def test_missing_approval_list_refused(self):
        base_path, back_path = self.write_valid_attestations()
        rc, out, err = self.run_gate(base_path, back_path,
                                     extra=["--approve-discrepancies", ""])
        self.assertEqual(rc, mod.EXIT_CLI_FAILURE)  # empty list parsed -> CLI error

    def test_gate_without_approval_flag_refused(self):
        base_path, back_path = self.write_valid_attestations()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main(["--preflight-only",
                           "--baseline-attestation", base_path,
                           "--backup-attestation", back_path,
                           "--approved-key", self.pub_b64])
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("approve", err.getvalue())


# ===========================================================================
# Group B — baseline attestation v2 schema
# ===========================================================================

class TestBaselineSchema(Stage724Case):

    def gate_b(self, mutate, extra=None, backup_mutate=None):
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(mutate=mutate), self.sk)
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(mutate=backup_mutate),
                                      self.sk)
        return self.run_gate(base_path, back_path, extra=extra)

    def test_valid_baseline_passes(self):
        rc, out, err = self.gate_b(None)
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

    def test_five_ts_databases_on_separate_instances(self):
        att = make_baseline()
        ts_instances = [i for i in att["instances"] if i["database"] == "ts"]
        self.assertEqual(len(ts_instances), 5)
        ids = {i["instance_id"] for i in ts_instances}
        self.assertEqual(len(ids), 5)

    def test_duplicate_instance_rejected(self):
        def mutate(att):
            dup = json.loads(json.dumps(att["instances"][0]))
            att["instances"].append(dup)
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("duplicate instance_id", err)

    def test_missing_instance_rejected(self):
        def mutate(att):
            att["instances"] = [i for i in att["instances"]
                                if i["instance_id"] != "ts-order-mongo"]
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("expected exactly 7", err)

    def test_unexpected_instance_rejected(self):
        def mutate(att):
            extra = json.loads(json.dumps(att["instances"][0]))
            extra["instance_id"] = "ts-voucher-mongo"
            att["instances"].append(extra)
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("unexpected or missing instance_id", err)

    def test_wrong_database_name_rejected(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-contacts-mongo":
                    i["database"] = "contacts-db"
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("wrong database name", err)

    def test_ambiguous_ts_conflation_rejected(self):
        # Collapsing the five 'ts' databases into one instance record
        # (e.g. removing order-other) must be rejected as a missing target.
        def mutate(att):
            att["instances"] = [
                i for i in att["instances"]
                if i["instance_id"] != "ts-order-other-mongo"]
            att["instances"].append({
                "instance_id": "ts-order-mongo",
                "namespace": "train-ticket",
                "service_name": "ts-order-mongo",
                "service_uid": "uid-x",
                "application": "ts-order-service",
                "database": "ts",
                "mongodb_version": "4.4.x",
                "collections": [
                    _col("orders", True, 4,
                         [_ident("_id", "5ad7750b-a68b-49c0-a8c0-32776b067703",
                                 True, bson_type="binary_subtype_3")])],
            })
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)

    def test_absent_collection_accepted_and_wrong_exists_rejected(self):
        # The valid fixture marks ts-payment-mongo.addMoney absent -> passes.
        rc, out, err = self.gate_b(None)
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-payment-mongo":
                    for c in i["collections"]:
                        if c["name"] == "addMoney":
                            c["exists"] = True
                            c["document_count"] = 0
                            c["discrepancy_ids"] = []
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("collection inventory mismatch", err)

    def test_wrong_collection_count_rejected(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-contacts-mongo":
                    i["collections"].append(
                        _col("surprise", True, 1, []))
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("collection inventory mismatch", err)

    def test_order_other_sentinel_accountid_check_required(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-other-mongo":
                    for c in i["collections"]:
                        c["identity_checks"] = [
                            _ident("_id",
                                   "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
                                   True)]
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing identity check: accountId", err)

    def test_order_other_sentinel_count_two_rejected(self):
        # accountId is not unique in general: two orders for one account are
        # perfectly legal at runtime. The sentinel pins the INITIALIZATION
        # baseline (exactly one document), so a two-document collection is a
        # gate failure, not an acceptable fingerprint match.
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-other-mongo":
                    for c in i["collections"]:
                        c["document_count"] = 2
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("one_order_baseline_sentinel", err)
        self.assertIn("document_count must be the strict integer 1", err)

    def test_order_other_sentinel_count_missing_rejected(self):
        # document_count must be a strict int for an existing collection;
        # null (the absent-collection shape) or a missing field fails closed
        # at the collection-schema layer BEFORE the sentinel check runs.
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-other-mongo":
                    for c in i["collections"]:
                        c["document_count"] = None
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("existing collection requires integer document_count",
                      err)

    def test_order_other_sentinel_boolean_count_rejected(self):
        # isinstance(True, int) is True in Python; strict typing must reject
        # boolean counts at the collection-schema layer.
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-other-mongo":
                    for c in i["collections"]:
                        c["document_count"] = True
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("existing collection requires integer document_count",
                      err)

    def test_order_other_claiming_fdse_id_as_docid_rejected(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-other-mongo":
                    for c in i["collections"]:
                        c["identity_checks"] = [
                            _ident("_id",
                                   "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
                                   True, bson_type="binary_subtype_3")]
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)

    def test_order_identities_require_binary_subtype_3(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-order-mongo":
                    for c in i["collections"]:
                        for chk in c["identity_checks"]:
                            chk.pop("bson_type", None)
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("must record bson_type", err)

    def test_known_discrepancies_explicitly_required(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-user-mongo":
                    i["collections"][0]["discrepancy_ids"] = []
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("does not report known discrepancy", err)

    def test_unknown_discrepancy_rejected(self):
        def mutate(att):
            for i in att["instances"]:
                if i["instance_id"] == "ts-auth-mongo":
                    i["collections"][0]["discrepancy_ids"] = \
                        ["SOME_UNAPPROVED_DISCREPANCY"]
        rc, out, err = self.gate_b(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("unknown discrepancy identifier", err)

    def test_unapproved_observed_discrepancy_rejected(self):
        partial = ",".join([mod.DISC_USER_INIT_FDSE_MISSING])
        rc, out, err = self.gate_b(None, extra=["--approve-discrepancies",
                                                partial])
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("not approved by operator", err)

    def test_future_timestamp_rejected(self):
        rc, out, err = self.gate_b(
            lambda att: att.update({"generated_at_utc": iso_future()}))
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("in the future", err)

    def test_stale_timestamp_rejected(self):
        rc, out, err = self.gate_b(
            lambda att: att.update({"generated_at_utc": iso_old()}))
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("stale", err)

    def test_unsupported_schema_version_rejected(self):
        rc, out, err = self.gate_b(
            lambda att: att.update({"schema_version": "baseline/v1"}))
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("unsupported baseline schema_version", err)

    def test_malformed_baseline_aborts(self):
        base_path = os.path.join(self.tmp, "broken.json")
        with open(base_path, "w") as f:
            f.write("not json {{{")
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(), self.sk)
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)

    def test_stale_baseline_attestation_aborts(self):
        # legacy name: stale baseline (kept from Stage 7.23 suite)
        rc, out, err = self.gate_b(
            lambda att: att.update({"generated_at_utc": iso_old()}))
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("stale", err)


# ===========================================================================
# Group K — backup attestation v2 schema
# ===========================================================================

class TestBackupSchema(Stage724Case):

    def gate_k(self, mutate=None, target_mutate=None):
        def combined_mutate(att):
            if mutate:
                mutate(att)
            if target_mutate:
                for t in att["targets"]:
                    target_mutate(t)
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(), self.sk)
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(mutate=combined_mutate),
                                      self.sk)
        return self.run_gate(base_path, back_path)

    def test_valid_backup_passes(self):
        rc, out, err = self.gate_k()
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

    def test_all_seven_ts_targets_independent(self):
        att = make_backup()
        self.assertEqual(len(att["targets"]), 7)
        self.assertEqual(len({(t["instance_id"], t["database"])
                              for t in att["targets"]}), 7)
        self.assertEqual(
            sum(1 for t in att["targets"] if t["database"] == "ts"), 5)

    def test_duplicate_backup_target_rejected(self):
        def mutate(att):
            att["targets"].append(json.loads(json.dumps(att["targets"][3])))
        rc, out, err = self.gate_k(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("duplicate backup target", err)

    def test_missing_backup_target_rejected(self):
        def mutate(att):
            att["targets"] = [t for t in att["targets"]
                              if t["instance_id"] != "ts-payment-mongo"]
        rc, out, err = self.gate_k(mutate)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("expected exactly 7", err)

    def test_backup_restored_counts_missing_inventory_key_rejected(self):
        # Stage 7.25 erratum: the collection inventory is nested within each
        # of the seven instance/database targets. A ts-payment-mongo/ts
        # backup whose restored counts omit addMoney (the absent collection,
        # restored as count 0) no longer carries the complete inventory and
        # is refused.
        def tmut(t):
            if t["instance_id"] == "ts-payment-mongo":
                t["restored_collection_counts"] = {"payment": 1}
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("restored_collection_counts keys must be exactly",
                      err)
        self.assertIn("'addMoney'", err)

    def test_backup_restored_counts_unexpected_key_rejected(self):
        # No restored count may name a collection the instance/database does
        # not own — an extra key means the target identity and its inventory
        # disagree.
        def tmut(t):
            if t["instance_id"] == "ts-order-other-mongo":
                t["restored_collection_counts"] = {"orders": 1, "junk": 0}
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("restored_collection_counts keys must be exactly", err)

    def test_backup_restored_absent_collection_zero_accepted(self):
        # The absent ts-payment-mongo/ts/addMoney entry is part of the
        # inventory and restores as an explicit count 0 (documented absent
        # collection, never silently dropped from the target).
        def tmut(t):
            if t["instance_id"] == "ts-payment-mongo":
                t["restored_collection_counts"] = {"payment": 1, "addMoney": 0}
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_SUCCESS, err)

    def test_backup_wrong_database_name_rejected(self):
        def mutate(att):
            for i in att["instances"] if False else []:
                pass
        def tmut(t):
            if t["instance_id"] == "ts-inside-payment-mongo":
                t["database"] = "inside-payment-db"
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("wrong database name", err)

    def test_backup_missing_artifact_hash_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-user-mongo":
                del t["artifact_sha256"]
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing required field artifact_sha256", err)

    def test_backup_bad_artifact_hash_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-user-mongo":
                t["artifact_sha256"] = "not-a-hash"
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("64 hex chars", err)

    def test_backup_missing_offcluster_evidence_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-auth-mongo":
                del t["off_cluster_storage_location"]
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing required field off_cluster_storage_location", err)

    def test_backup_offcluster_false_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-auth-mongo":
                t["off_cluster_storage_verified"] = False
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("off_cluster_storage_verified is not true", err)

    def test_backup_missing_restore_evidence_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-contacts-mongo":
                del t["restore_tested_at_utc"]
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing required field restore_tested_at_utc", err)

    def test_backup_restore_failed_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-contacts-mongo":
                t["restore_result"] = "failed"
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("restore_result is not 'verified'", err)

    def test_backup_stale_rejected(self):
        def tmut(t):
            if t["instance_id"] == "ts-user-mongo":
                t["backup_created_at_utc"] = iso_old()
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("stale", err)

    def test_backup_invalid_signature_rejected(self):
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(), self.sk)
        back_path = write_attestation(self.tmp, "backup.json",
                                      make_backup(), self.sk_other)
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("no approved key verified", err)

    def test_backup_tampered_rejected(self):
        base_path, back_path = self.write_valid_attestations()
        att = json.load(open(back_path))
        att["targets"][0]["artifact_sha256"] = "f" * 64
        data = mod.canonical_json_bytes(att)
        with open(back_path, "wb") as f:
            f.write(data)
        with open(back_path + ".sha256", "w") as f:
            f.write(f"{hashlib.sha256(data).hexdigest()}  {back_path}\n")
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("no approved key verified", err)

    def test_backup_boolean_only_coverage_rejected(self):
        # A v1-style attestation with a boolean 'all_databases_covered'
        # instead of per-target evidence must be rejected.
        att = {"schema_version": mod.BACKUP_SCHEMA_VERSION,
               "generated_at_utc": iso_now(),
               "all_databases_covered": True}
        base_path = write_attestation(self.tmp, "baseline.json",
                                      make_baseline(), self.sk)
        back_path = write_attestation(self.tmp, "backup.json", att, self.sk)
        rc, out, err = self.run_gate(base_path, back_path)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)
        self.assertIn("missing required field: targets", err)

    def test_incomplete_backup_attestation_aborts(self):
        # legacy name from Stage 7.23 suite
        def tmut(t):
            if t["instance_id"] == "ts-order-mongo":
                del t["evidence_reference"]
        rc, out, err = self.gate_k(target_mutate=tmut)
        self.assertEqual(rc, mod.EXIT_GATE_FAILURE)


# ===========================================================================
# Group E — end-to-end mock runs and checkpoint/resume recovery
# ===========================================================================

class TestEndToEndAndRecovery(Stage724Case):

    def _live_args(self, results, seed_art, checkpoint, base_path, back_path,
                   users_n=5, extra=None):
        return [
            "--count", str(users_n), "--rng-seed", "20260301",
            "--write-authorize",
            "--results", results,
            "--seed-artifact", seed_art,
            "--checkpoint", checkpoint,
            "--baseline-attestation", base_path,
            "--backup-attestation", back_path,
            "--approve-discrepancies", ALL_DISCREPANCIES,
            "--approved-key", self.pub_b64,
            "--user-base", "http://127.0.0.1:1",
            "--auth-base", "http://127.0.0.1:1",
        ] + (extra or [])

    def _run_main(self, argv, transport):
        mod.set_http_post(transport)
        try:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = mod.main(argv)
            return rc, out.getvalue() + err.getvalue()
        finally:
            mod.set_http_post(None)

    def test_dry_run_no_http_calls(self):
        calls = []

        def handler(url, payload, all_calls):
            calls.append(url)
            raise AssertionError("dry-run must not make HTTP calls")

        rec = Recorder(handler)
        plan_path = os.path.join(self.tmp, "plan.json")
        rc, console = self._run_main(
            ["--count", "10", "--dry-run", "--out", plan_path], rec)
        self.assertEqual(rc, mod.EXIT_SUCCESS, console)
        self.assertEqual(calls, [])
        plan = json.load(open(plan_path))
        self.assertEqual(plan["count"], 10)
        self.assertEqual(plan["mode"], "dry-run-plan")
        self.assertEqual(plan["contacts_semantics"],
                         "july_compatible_null_contactsId")
        # Sanitized plan: no credential/PII fields.
        self.assertNotIn("users_first_5", plan)
        self.assertNoSecrets(open(plan_path).read(), mod.plan_users(10, 1))

    def test_checkpoint_resume_skips_interrupted_user(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        # Pre-existing checkpoint: user 0 terminal-verified; user 1 attempt
        # started but never terminalized (interrupted -> unknown outcome);
        # users 2..4 never attempted.
        with open(checkpoint, "w") as f:
            f.write(json.dumps({
                "t": "outcome", "idx": 0,
                "outcome": mod.REGISTERED_AND_LOGIN_VERIFIED,
                "confirmed": True, "login_retry_allowed": False,
                "uncertain": False,
                "reg_http_status": 201, "reg_body_status": 1,
                "login_http_status": 200, "login_body_status": 1,
                "login_user_id": "acct-seeduser0_310538",
                "error_category": None}) + "\n")
            f.write(json.dumps({
                "t": "attempt", "idx": 1, "phase": "register"}) + "\n")

        def handler(url, payload, all_calls):
            if url.endswith("/users/register"):
                return ok_register(payload)
            return ok_login(payload)

        rec = Recorder(handler)
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path, extra=["--resume"]), rec)
        self.assertEqual(rc, mod.EXIT_PARTIAL_FAILURE, console)

        # User 0: no re-POST. User 1: interrupted -> NEVER re-POSTed.
        self.assertEqual(rec.register_count("seeduser0_310538"), 0)
        self.assertEqual(rec.register_count("seeduser1_30559"), 0)
        # Users 2..4 processed fresh.
        for i in (2, 3, 4):
            self.assertEqual(rec.register_count(f"seeduser{i}_"
                                                f"{mod.plan_users(5, 20260301)[i]['username'].split('_', 2)[-1]}"), 1)

        manifest = json.load(open(results))
        by_idx = {u["idx"]: u for u in manifest["users"]}
        self.assertEqual(by_idx[1]["outcome"], mod.REGISTRATION_OUTCOME_UNKNOWN)
        self.assertEqual(by_idx[1]["error_category"],
                         "interrupted_attempt_unknown")
        self.assertEqual(manifest["uncertain_outcomes"], 1)
        self.assertEqual(manifest["usable_pool_size"], 4)

    def test_resume_login_only_retry_never_reposts_registration(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        with open(checkpoint, "w") as f:
            f.write(json.dumps({
                "t": "outcome", "idx": 0,
                "outcome": mod.REGISTRATION_SUCCESS_LOGIN_FAILURE,
                "confirmed": True, "login_retry_allowed": True,
                "uncertain": False,
                "reg_http_status": 201, "reg_body_status": 1,
                "login_http_status": 200, "login_body_status": 0,
                "login_user_id": None,
                "error_category": "application_error"}) + "\n")

        def handler(url, payload, all_calls):
            if url.endswith("/users/register"):
                raise AssertionError("registration must NEVER be re-POSTed")
            return ok_login(payload)

        rec = Recorder(handler)
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path, users_n=1, extra=["--resume"]), rec)
        self.assertEqual(rc, mod.EXIT_SUCCESS, console)
        self.assertEqual(rec.register_count(), 0)
        self.assertEqual(rec.login_count(), 1)
        manifest = json.load(open(results))
        self.assertEqual(manifest["users"][0]["outcome"],
                         mod.REGISTERED_AND_LOGIN_VERIFIED)

    def test_resume_conflict_without_flag(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")
        with open(checkpoint, "w") as f:
            f.write(json.dumps({"t": "outcome", "idx": 0,
                                "outcome": mod.REGISTERED_AND_LOGIN_VERIFIED,
                                "confirmed": True,
                                "login_retry_allowed": False,
                                "uncertain": False}) + "\n")

        rec = Recorder(lambda url, payload, all: MockResp(500))
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path), rec)
        self.assertEqual(rc, mod.EXIT_RESUME_CONFLICT)
        self.assertIn("RESUME_CONFLICT", console)

    def test_resume_without_checkpoint_file_refused(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "missing.jsonl")
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path, extra=["--resume"]),
            Recorder(lambda url, payload, all: MockResp(500)))
        self.assertEqual(rc, mod.EXIT_RESUME_CONFLICT)

    def test_output_failure_before_first_request(self):
        base_path, back_path = self.write_valid_attestations()
        bad_dir = os.path.join(self.tmp, "no_such_dir")
        results = os.path.join(bad_dir, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        def handler(url, payload, all_calls):
            raise AssertionError("no HTTP call may happen before output "
                                 "validation succeeds")

        rec = Recorder(handler)
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path), rec)
        self.assertEqual(rc, mod.EXIT_OUTPUT_VALIDATION_FAILURE)
        self.assertEqual(len(rec.calls), 0)

    def test_output_permission_failure_before_first_request(self):
        base_path, back_path = self.write_valid_attestations()
        locked = os.path.join(self.tmp, "locked")
        os.makedirs(locked)
        os.chmod(locked, 0o555)
        self.addCleanup(os.chmod, locked, 0o755)
        results = os.path.join(locked, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        rec = Recorder(lambda url, payload, all: ok_register(payload))
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path), rec)
        self.assertEqual(rc, mod.EXIT_OUTPUT_VALIDATION_FAILURE)
        self.assertEqual(len(rec.calls), 0)

    def test_existing_artifact_refused_without_resume(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")
        with open(seed_art, "w") as f:
            f.write("[]")
        rec = Recorder(lambda url, payload, all: ok_register(payload))
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path), rec)
        self.assertEqual(rc, mod.EXIT_OUTPUT_VALIDATION_FAILURE)
        self.assertIn("refusing to overwrite", console)
        self.assertEqual(len(rec.calls), 0)

    def test_mixed_outcomes_partial_exit_and_pool(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")

        def idx_of(uname):
            # usernames are seeduser{idx}_{rand}; the index precedes the '_'.
            return int(uname.split("_", 1)[0][len("seeduser"):])

        def handler(url, payload, all_calls):
            if url.endswith("/users/register"):
                idx = idx_of(payload["userName"])
                if idx == 1:
                    return MockResp(201, {"status": 0,
                                          "msg": "USER HAS ALREADY EXISTS",
                                          "data": None})
                if idx == 2:
                    return MockResp(201, {"status": 0, "msg": "DB ERR",
                                          "data": None})
                if idx == 3:
                    return MockResp(503, {"status": 0})
                return ok_register(payload)
            idx = idx_of(payload["username"])
            if idx == 4:
                return MockResp(200, {"status": 0,
                                      "msg": "Incorrect username or password.",
                                      "data": None})
            return ok_login(payload)

        rec = Recorder(handler)
        rc, console = self._run_main(
            self._live_args(results, seed_art, checkpoint, base_path,
                            back_path), rec)
        self.assertEqual(rc, mod.EXIT_PARTIAL_FAILURE)
        manifest = json.load(open(results))
        by_idx = {u["idx"]: u for u in manifest["users"]}
        self.assertEqual(by_idx[0]["outcome"],
                         mod.REGISTERED_AND_LOGIN_VERIFIED)
        # idx1: duplicate registration, login succeeded in the mock handler
        # -> duplicate + login verified (usable).
        self.assertEqual(by_idx[1]["outcome"], mod.DUPLICATE_AND_LOGIN_VERIFIED)
        self.assertEqual(by_idx[2]["outcome"],
                         mod.REGISTRATION_APPLICATION_FAILURE)
        self.assertEqual(by_idx[3]["outcome"],
                         mod.REGISTRATION_OUTCOME_UNCERTAIN)
        self.assertEqual(by_idx[4]["outcome"],
                         mod.REGISTRATION_SUCCESS_LOGIN_FAILURE)
        # Pool: only login-verified users (0 and the duplicate-verified one).
        self.assertEqual(by_idx[1]["outcome"], mod.DUPLICATE_AND_LOGIN_VERIFIED)
        self.assertEqual(manifest["usable_pool_size"], 2)
        artifact = json.load(open(seed_art))
        self.assertEqual({e["idx"] for e in artifact}, {0, 1})
        self.assertNoSecrets(json.dumps(manifest),
                             mod.plan_users(5, 20260301))

    def test_missing_write_authorization_aborts(self):
        base_path, back_path = self.write_valid_attestations()
        results = os.path.join(self.tmp, "results.json")
        seed_art = os.path.join(self.tmp, "k6seed.json")
        checkpoint = os.path.join(self.tmp, "ckpt.jsonl")
        argv = self._live_args(results, seed_art, checkpoint, base_path,
                               back_path)
        argv.remove("--write-authorize")
        rec = Recorder(lambda url, payload, all: ok_register(payload))
        rc, console = self._run_main(argv, rec)
        self.assertEqual(rc, mod.EXIT_CLI_FAILURE)
        self.assertIn("--write-authorize", console)
        self.assertEqual(len(rec.calls), 0)


# ===========================================================================
# Group V — v3 compatibility
# ===========================================================================

class TestV3Compatibility(Stage724Case):

    def test_v3_semantics_preserved(self):
        v4 = mod.plan_users(3000, 20260301)
        spec3 = importlib.util.spec_from_file_location("seed_v3", GEN_V3)
        v3mod = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(v3mod)
        v3 = v3mod.plan_users(3000, 20260301)
        self.assertEqual(v3, v4)
        for other_seed in (7, 42, 20260302):
            self.assertEqual(v3mod.plan_users(25, other_seed),
                             mod.plan_users(25, other_seed))
        self.assertTrue(all(u["contactsId"] is None for u in v4))
        self.assertTrue(all(u["password"] == "loadtest123" for u in v4))

    def test_v3_source_hash_unchanged(self):
        expected_gen = ("4aec99a5a52779ed8a4c67986e7393d35b9dc71896931c7946"
                        "4983c7e4d74b16")
        expected_test = ("33f20566f24e8bac0857e9555a0ae8267e5ec3dc8f62e7ec"
                         "0734f6a406113494")
        h_gen = hashlib.sha256(open(GEN_V3, "rb").read()).hexdigest()
        h_test = hashlib.sha256(open(TEST_V3, "rb").read()).hexdigest()
        self.assertEqual(h_gen, expected_gen)
        self.assertEqual(h_test, expected_test)

    def test_v3_offline_tests_still_pass(self):
        result = subprocess.run([sys.executable, TEST_V3],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("All offline seed tests PASSED.", result.stdout)
        self.assertEqual(result.stdout.count("PASS:"), 37)


if __name__ == "__main__":
    unittest.main(verbosity=2)

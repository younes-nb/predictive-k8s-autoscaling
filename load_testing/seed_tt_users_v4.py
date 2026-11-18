#!/usr/bin/env python3
"""Deterministic seed generator v4 (v4.2) for Stage 8 — July-compatible semantics.

Generates a 3,000-user seed that preserves the historical July workload behavior:

  - 3,000 users with deterministic usernames, document numbers, phone numbers
  - Valid authentication credentials (username + password) for each user
  - contactsId = null in the load-generator user state (no contact creation)
  - Empty contacts database, matching the historical July condition

Stage 7.24 hardening (over the Stage 7.23 v4.1 implementation):

  - Registration/login parsers return STEP-level states; combined per-user
    outcomes are derived by an explicit state machine (v4.1 crashed in live
    mode: it compared against undefined REGISTERED/DUPLICATE constants).
  - Strict JSON typing: response "status" must be a real JSON integer
    (Python bool is rejected even though bool is a subclass of int).
  - Login verifies returned username == requested username and requires
    non-empty string token/userId/username.
  - 5xx registration responses are recorded as OUTCOME-UNCERTAIN (the server
    may have committed the request); 4xx is a definitive HTTP failure.
  - Registration POSTs are NEVER automatically retried after timeouts,
    transport errors, or 5xx.
  - Two separate output artifacts: an operational result manifest that
    contains NO credentials/PII, and a restricted k6 seed artifact (0600)
    containing only the identity fields the k6 script reads
    (username, accountId, contactsId — k6 performs its own logins with the
    global PW env; it never reads tokens or passwords from the seed file).
  - Durable per-user JSONL checkpoint: an "attempt started" record is fsynced
    BEFORE each registration POST so a restart can distinguish
    "never attempted" from "outcome unknown"; users with unknown outcomes are
    skipped on resume and never re-POSTed.
  - Baseline/backup attestations are verified against a detached SHA-256
    checksum file AND a detached Ed25519 signature over the exact file bytes,
    verified against an operator-approved public key.
  - Baseline attestation v2 identifies all seven independent MongoDB
    instances (five share the database name "ts" on separate servers).
  - Backup attestation v2 requires per-target evidence (artifact hash,
    off-cluster storage, restore test) — boolean-only coverage is rejected.

July-compatible workload semantics are unchanged from v3
(seed_tt_users_v3_july_semantics.py plan_users).

Exit codes:
  0 SUCCESS            — dry-run OK, preflight OK, or complete live seed
  1 PARTIAL_FAILURE    — live run finished; some users failed/uncertain
  2 CLI_FAILURE        — invalid CLI arguments
  3 GATE_FAILURE       — attestation/integrity/approval gate refused
  4 WRITE_FAILURE      — an output file could not be written/updated
  5 OUTPUT_VALIDATION_FAILURE — pre-run output validation failed
  6 RESUME_CONFLICT    — checkpoint state incompatible with invocation
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import random
import stat as stat_mod
import sys

GENERATOR_VERSION = "seed_tt_users_v4.py 2.0 (stage7.24 2026-11-12)"

# ---------------------------------------------------------------------------
# Canonical JSON (used when producing attestations; verification hashes the
# exact attestation file bytes, so verification needs no canonicalization).
# ---------------------------------------------------------------------------

def canonical_json_bytes(obj) -> bytes:
    """Deterministic canonical JSON encoding for attestation production."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_CLI_FAILURE = 2
EXIT_GATE_FAILURE = 3
EXIT_WRITE_FAILURE = 4
EXIT_OUTPUT_VALIDATION_FAILURE = 5
EXIT_RESUME_CONFLICT = 6

# ---------------------------------------------------------------------------
# Registration STEP states (per HTTP exchange, before combination)
# ---------------------------------------------------------------------------

REG_REGISTERED = "REG_REGISTERED"                     # 201 + status int 1
REG_DUPLICATE = "REG_DUPLICATE"                       # 201 + status int 0 + verified duplicate msg
REG_APPLICATION_FAILURE = "REG_APPLICATION_FAILURE"   # 201 + status int 0, non-duplicate msg
REG_HTTP_FAILURE = "REG_HTTP_FAILURE"                 # non-201 HTTP status (4xx = definitive)
REG_OUTCOME_UNKNOWN = "REG_OUTCOME_UNKNOWN"           # unparseable/unknown body or transport error

# ---------------------------------------------------------------------------
# Login STEP states
# ---------------------------------------------------------------------------

LOGIN_VERIFIED = "LOGIN_VERIFIED"                     # 200 + status int 1 + valid data + username match
LOGIN_APPLICATION_FAILURE = "LOGIN_APPLICATION_FAILURE"  # 200 + status int 0
LOGIN_HTTP_FAILURE = "LOGIN_HTTP_FAILURE"             # non-200 HTTP status
LOGIN_OUTCOME_UNKNOWN = "LOGIN_OUTCOME_UNKNOWN"       # transport error/timeout on login
LOGIN_INVALID_RESPONSE = "LOGIN_INVALID_RESPONSE"     # unparseable/invalid login body

# ---------------------------------------------------------------------------
# Combined per-user outcomes (the documented state machine)
# ---------------------------------------------------------------------------

REGISTERED_AND_LOGIN_VERIFIED = "REGISTERED_AND_LOGIN_VERIFIED"
DUPLICATE_AND_LOGIN_VERIFIED = "DUPLICATE_AND_LOGIN_VERIFIED"
REGISTRATION_SUCCESS_LOGIN_FAILURE = "REGISTRATION_SUCCESS_LOGIN_FAILURE"
DUPLICATE_LOGIN_FAILURE = "DUPLICATE_LOGIN_FAILURE"
REGISTRATION_APPLICATION_FAILURE = "REGISTRATION_APPLICATION_FAILURE"
REGISTRATION_HTTP_FAILURE = "REGISTRATION_HTTP_FAILURE"
REGISTRATION_OUTCOME_UNKNOWN = "REGISTRATION_OUTCOME_UNKNOWN"
REGISTRATION_OUTCOME_UNCERTAIN = "REGISTRATION_OUTCOME_UNCERTAIN"
LOGIN_APPLICATION_FAILURE = "LOGIN_APPLICATION_FAILURE"
LOGIN_HTTP_FAILURE = "LOGIN_HTTP_FAILURE"
LOGIN_INVALID_RESPONSE = "LOGIN_INVALID_RESPONSE"

COMBINED_OUTCOMES = {
    REGISTERED_AND_LOGIN_VERIFIED,
    DUPLICATE_AND_LOGIN_VERIFIED,
    REGISTRATION_SUCCESS_LOGIN_FAILURE,
    DUPLICATE_LOGIN_FAILURE,
    REGISTRATION_APPLICATION_FAILURE,
    REGISTRATION_HTTP_FAILURE,
    REGISTRATION_OUTCOME_UNKNOWN,
    REGISTRATION_OUTCOME_UNCERTAIN,
    LOGIN_APPLICATION_FAILURE,
    LOGIN_HTTP_FAILURE,
    LOGIN_INVALID_RESPONSE,
}

# Outcomes whose registration side effect is CONFIRMED (no register re-POST
# would ever be needed; login retry may be allowed for confirmed registrations).
CONFIRMED_OUTCOMES = {
    REGISTERED_AND_LOGIN_VERIFIED,
    DUPLICATE_AND_LOGIN_VERIFIED,
    REGISTRATION_APPLICATION_FAILURE,
    REGISTRATION_HTTP_FAILURE,        # 4xx only (see classify_registration_http)
}

# Outcomes where re-attempting ONLY the login is safe. Login is side-effect
# free (deployed TokenServiceImpl.getToken reads the user repository and
# builds a JWT; it performs no writes — bytecode-verified Stage 7.19/7.20).
LOGIN_RETRY_ALLOWED_OUTCOMES = {
    REGISTRATION_SUCCESS_LOGIN_FAILURE,
    DUPLICATE_LOGIN_FAILURE,
}

# Uncertain outcomes: the registration POST may have committed server-side.
# These are never retried automatically and always require operator review.
UNCERTAIN_OUTCOMES = {
    REGISTRATION_OUTCOME_UNKNOWN,
    REGISTRATION_OUTCOME_UNCERTAIN,
}

# ---------------------------------------------------------------------------
# Known baseline discrepancy identifiers (Stage 7.21/7.22/7.23 verified)
# ---------------------------------------------------------------------------

DISC_USER_INIT_FDSE_MISSING = "USER_INIT_FDSE_MISSING"
DISC_PAYMENT_EXPECTED_ID_ABSENT = "PAYMENT_EXPECTED_ID_ABSENT"
DISC_PAYMENT_LIVE_ID_DIFFERS = "PAYMENT_LIVE_ID_DIFFERS"
DISC_PAYMENT_ADDMONEY_ABSENT = "PAYMENT_ADDMONEY_ABSENT"

KNOWN_DISCREPANCY_IDS = {
    DISC_USER_INIT_FDSE_MISSING,
    DISC_PAYMENT_EXPECTED_ID_ABSENT,
    DISC_PAYMENT_LIVE_ID_DIFFERS,
    DISC_PAYMENT_ADDMONEY_ABSENT,
}

# ---------------------------------------------------------------------------
# Seven-instance MongoDB inventory (Stage 7.21 topology, RUNTIME-VERIFIED).
# Five instances expose a database literally named "ts" — they are separate
# MongoDB servers, not one shared database.
# ---------------------------------------------------------------------------

EXPECTED_INSTANCES = [
    {
        "instance_id": "ts-user-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-user-mongo",
        "application": "ts-user-service",
        "database": "ts-user-mongo",
        "collections": [
            {"name": "user", "exists": True},
        ],
    },
    {
        "instance_id": "ts-auth-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-auth-mongo",
        "application": "ts-auth-service",
        "database": "ts-auth-mongo",
        "collections": [
            {"name": "user", "exists": True},
        ],
    },
    {
        "instance_id": "ts-contacts-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-contacts-mongo",
        "application": "ts-contacts-service",
        "database": "ts",
        "collections": [
            {"name": "contacts", "exists": True},
        ],
    },
    {
        "instance_id": "ts-order-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-order-mongo",
        "application": "ts-order-service",
        "database": "ts",
        "collections": [
            {"name": "orders", "exists": True},
        ],
    },
    {
        "instance_id": "ts-order-other-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-order-other-mongo",
        "application": "ts-order-other-service",
        "database": "ts",
        "collections": [
            {"name": "orders", "exists": True},
        ],
    },
    {
        "instance_id": "ts-payment-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-payment-mongo",
        "application": "ts-payment-service",
        "database": "ts",
        "collections": [
            {"name": "payment", "exists": True},
            {"name": "addMoney", "exists": False},
        ],
    },
    {
        "instance_id": "ts-inside-payment-mongo",
        "namespace": "train-ticket",
        "service_name": "ts-inside-payment-mongo",
        "application": "ts-inside-payment-service",
        "database": "ts",
        "collections": [
            {"name": "payment", "exists": True},
            {"name": "addMoney", "exists": True},
        ],
    },
]

EXPECTED_INSTANCE_IDS = [i["instance_id"] for i in EXPECTED_INSTANCES]

BASELINE_SCHEMA_VERSION = "train-ticket-baseline-attestation/v2"
BACKUP_SCHEMA_VERSION = "train-ticket-backup-attestation/v2"

# ---------------------------------------------------------------------------
# Plan generation (byte-identical semantics to v3 plan_users)
# ---------------------------------------------------------------------------

def plan_users(count: int, rng_seed: int):
    """Generate deterministic user plans (identical to v3 for equal args).

    Returns a list of user dicts with:
      idx, username, password, documentNum, phoneNumber, contactsId=None
    Passwords/document numbers/phone numbers are INTERNAL plan data; they are
    never written to any output artifact (see artifact writers below).
    """
    rng = random.Random(rng_seed)
    users = []
    for i in range(count):
        uname = f"seeduser{i}_{rng.randint(0, 999999)}"
        docnum = rng.randint(10**9, 10**10 - 1)
        phone = f"138{rng.randint(10**7, 10**8 - 1)}"
        users.append({
            "idx": i,
            "username": uname,
            "password": "loadtest123",
            "documentNum": docnum,
            "phoneNumber": phone,
            "contactsId": None,  # July-compatible: no contact created
        })
    return users


# ---------------------------------------------------------------------------
# Strict JSON field helpers (bool is NOT an int here)
# ---------------------------------------------------------------------------

def is_strict_int(value) -> bool:
    """True only for real integers (Python bool is rejected)."""
    return type(value) is int


def is_nonempty_str(value) -> bool:
    return isinstance(value, str) and value != ""


# ---------------------------------------------------------------------------
# HTTP transport hook (monkeypatchable for testing)
# ---------------------------------------------------------------------------

_http_post = None


def set_http_post(fn):
    """Install a transport hook (tests)."""
    global _http_post
    _http_post = fn


def _get_http_post():
    global _http_post
    if _http_post is None:
        import requests  # imported lazily so offline modes need no dependency
        _http_post = requests.post
    return _http_post


# ---------------------------------------------------------------------------
# Registration response parsing (STEP state only)
# ---------------------------------------------------------------------------

def parse_registration_response(response) -> dict:
    """Classify one registration HTTP exchange.

    Success requires: HTTP status exactly 201, valid JSON dict, top-level
    "status" of strict int type equal to 1.
    Duplicate requires: HTTP 201, valid JSON, strict-int status 0 AND a
    verified duplicate message ("USER HAS ALREADY EXISTS" family).
    Everything else fails closed.

    A 5xx is REG_HTTP_FAILURE with outcome_uncertain=True: the server may
    have committed the request despite the error response.
    A transport failure (response is None) is REG_OUTCOME_UNKNOWN: the
    request may have been committed before the failure.
    """
    result = {
        "state": REG_OUTCOME_UNKNOWN,
        "http_status": None,
        "body_status": None,
        "error_category": None,
        "outcome_uncertain": False,
    }

    if response is None:
        result["error_category"] = "transport_or_timeout"
        result["outcome_uncertain"] = True
        return result

    status_code = getattr(response, "status_code", None)
    result["http_status"] = status_code

    if not is_strict_int(status_code):
        result["error_category"] = "invalid_http_status"
        result["outcome_uncertain"] = True
        return result

    if status_code != 201:
        result["state"] = REG_HTTP_FAILURE
        result["error_category"] = f"http_{status_code}"
        if 500 <= status_code <= 599:
            # A 5xx does not prove that no side effect occurred.
            result["outcome_uncertain"] = True
        return result

    try:
        body = response.json()
    except Exception:
        # 2xx with unparseable body: unknown application outcome.
        result["state"] = REG_OUTCOME_UNKNOWN
        result["error_category"] = "invalid_json"
        return result

    if not isinstance(body, dict):
        result["state"] = REG_OUTCOME_UNKNOWN
        result["error_category"] = "invalid_json"
        return result

    body_status = body.get("status")
    result["body_status"] = body_status

    if not is_strict_int(body_status):
        result["state"] = REG_OUTCOME_UNKNOWN
        result["error_category"] = "invalid_status_type"
        return result

    if body_status == 1:
        result["state"] = REG_REGISTERED
        return result

    if body_status == 0:
        msg = body.get("msg")
        msg_str = msg if isinstance(msg, str) else ""
        if "ALREADY EXISTS" in msg_str.upper():
            result["state"] = REG_DUPLICATE
        else:
            result["state"] = REG_APPLICATION_FAILURE
            result["error_category"] = "application_error"
        return result

    result["state"] = REG_OUTCOME_UNKNOWN
    result["error_category"] = f"unknown_status_{body_status}"
    return result


# ---------------------------------------------------------------------------
# Login response parsing (STEP state only)
# ---------------------------------------------------------------------------

def parse_login_response(response, requested_username: str) -> dict:
    """Classify one login HTTP exchange.

    Success requires: HTTP exactly 200, valid JSON dict, strict-int
    top-level status 1, data dict containing non-empty STRING userId,
    username, token, and returned username == requested_username.
    """
    result = {
        "state": LOGIN_INVALID_RESPONSE,
        "http_status": None,
        "body_status": None,
        "error_category": None,
        "token": None,          # internal only; never logged or manifested
        "user_id": None,
        "outcome_uncertain": False,
    }

    if response is None:
        result["state"] = LOGIN_OUTCOME_UNKNOWN
        result["error_category"] = "transport_or_timeout"
        return result

    status_code = getattr(response, "status_code", None)
    result["http_status"] = status_code

    if not is_strict_int(status_code):
        result["state"] = LOGIN_OUTCOME_UNKNOWN
        result["error_category"] = "invalid_http_status"
        return result

    if status_code != 200:
        result["state"] = LOGIN_HTTP_FAILURE
        result["error_category"] = f"http_{status_code}"
        return result

    try:
        body = response.json()
    except Exception:
        result["state"] = LOGIN_INVALID_RESPONSE
        result["error_category"] = "invalid_json"
        return result

    if not isinstance(body, dict):
        result["state"] = LOGIN_INVALID_RESPONSE
        result["error_category"] = "invalid_json"
        return result

    body_status = body.get("status")
    result["body_status"] = body_status

    if not is_strict_int(body_status):
        result["state"] = LOGIN_INVALID_RESPONSE
        result["error_category"] = "invalid_status_type"
        return result

    if body_status == 1:
        data = body.get("data")
        if not isinstance(data, dict):
            result["error_category"] = "invalid_data"
            return result
        token = data.get("token")
        user_id = data.get("userId")
        username = data.get("username")
        if not (is_nonempty_str(token) and is_nonempty_str(user_id)
                and is_nonempty_str(username)):
            result["error_category"] = "missing_or_invalid_data_fields"
            return result
        if username != requested_username:
            # Returned username must match the requested username exactly.
            result["error_category"] = "username_mismatch"
            return result
        result["state"] = LOGIN_VERIFIED
        result["token"] = token
        result["user_id"] = user_id
        return result

    if body_status == 0:
        result["state"] = LOGIN_APPLICATION_FAILURE
        result["error_category"] = "application_error"
        return result

    result["error_category"] = f"unknown_status_{body_status}"
    return result


# ---------------------------------------------------------------------------
# Combined per-user outcome state machine
# ---------------------------------------------------------------------------

def combine_outcome(reg: dict, login=None) -> dict:
    """Derive the documented per-user outcome from step results.

    Returns dict with:
      outcome          — one of COMBINED_OUTCOMES
      confirmed        — registration side effect definitively known
      login_retry_allowed — re-attempting ONLY login is safe (login writes
                            nothing; a confirmed registration is not re-POSTed)
      uncertain        — registration outcome may be unknown (never auto-retried)
    """
    reg_state = reg["state"]

    if reg_state == REG_REGISTERED:
        if login is None:
            raise ValueError("REG_REGISTERED requires a login result")
        if login["state"] == LOGIN_VERIFIED:
            outcome = REGISTERED_AND_LOGIN_VERIFIED
            confirmed, login_retry = True, False
        else:
            # Registration succeeded; login definitively failed or could not
            # be parsed. Login has no side effects and may be re-attempted,
            # but this remains a possible partial registration (the deployed
            # user-service discards the auth-service response body).
            outcome = REGISTRATION_SUCCESS_LOGIN_FAILURE
            confirmed, login_retry = True, True
        uncertain = False
    elif reg_state == REG_DUPLICATE:
        if login is None:
            raise ValueError("REG_DUPLICATE requires a login result")
        if login["state"] == LOGIN_VERIFIED:
            outcome = DUPLICATE_AND_LOGIN_VERIFIED
            confirmed, login_retry = True, False
        else:
            outcome = DUPLICATE_LOGIN_FAILURE
            confirmed, login_retry = True, True
        uncertain = False
    elif reg_state == REG_APPLICATION_FAILURE:
        outcome = REGISTRATION_APPLICATION_FAILURE
        confirmed, login_retry, uncertain = True, False, False
    elif reg_state == REG_HTTP_FAILURE:
        # 4xx: definitive rejection (request not accepted). 5xx never lands
        # here — parse_registration_response flags 5xx as uncertain and the
        # caller maps it to REGISTRATION_OUTCOME_UNCERTAIN below.
        outcome = REGISTRATION_HTTP_FAILURE
        confirmed, login_retry, uncertain = True, False, False
    elif reg_state == REG_OUTCOME_UNKNOWN:
        outcome = REGISTRATION_OUTCOME_UNKNOWN
        confirmed, login_retry, uncertain = False, False, True
    else:
        raise ValueError(f"unknown registration state: {reg_state}")

    # A 5xx registration exchange is uncertain even though its step state is
    # REG_HTTP_FAILURE: distinguish via outcome_uncertain from the step parse.
    if reg_state == REG_HTTP_FAILURE and reg.get("outcome_uncertain"):
        outcome = REGISTRATION_OUTCOME_UNCERTAIN
        confirmed, login_retry, uncertain = False, False, True

    if outcome not in COMBINED_OUTCOMES:
        raise ValueError(f"unmapped combined outcome: {outcome}")

    return {
        "outcome": outcome,
        "confirmed": confirmed,
        "login_retry_allowed": login_retry,
        "uncertain": uncertain,
    }


# ---------------------------------------------------------------------------
# Attestation integrity: detached checksum + detached Ed25519 signature
# ---------------------------------------------------------------------------

class GateError(Exception):
    """Raised for any attestation-gate refusal."""


def read_file_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as exc:
        raise GateError(f"cannot read {path}: {exc}")


def verify_detached_checksum(attestation_path: str) -> None:
    """Require <path>.sha256 containing the SHA-256 of the exact file bytes."""
    data = read_file_bytes(attestation_path)
    actual = hashlib.sha256(data).hexdigest()

    sidecar = attestation_path + ".sha256"
    if not os.path.isfile(sidecar):
        raise GateError(f"missing detached checksum file: {sidecar}")
    raw = read_file_bytes(sidecar).decode("utf-8", "replace").strip()
    line = raw.splitlines()[0].strip() if raw else ""
    stored = line.split()[0] if line.split() else ""
    if len(stored) != 64 or any(c not in "0123456789abcdef" for c in stored.lower()):
        raise GateError(f"malformed detached checksum in {sidecar}")
    if stored.lower() != actual:
        raise GateError(
            f"detached checksum mismatch for {attestation_path}: "
            f"expected {actual}, got {stored.lower()}"
        )


def _load_ed25519_public_key(value: str):
    """Load an Ed25519 public key from a PEM file path or raw base64."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except Exception as exc:  # cryptography unavailable -> fail closed
        raise GateError(
            "signature verification requires the 'cryptography' package "
            f"(import failed: {exc})"
        )

    if os.path.isfile(value):
        try:
            with open(value, "rb") as f:
                pem = f.read()
            key = serialization.load_pem_public_key(pem)
        except Exception as exc:
            raise GateError(f"cannot load approved key PEM {value}: {exc}")
        if not isinstance(key, Ed25519PublicKey):
            raise GateError(f"approved key {value} is not an Ed25519 public key")
        return key

    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        raise GateError(
            f"approved key value is neither an existing file nor valid base64: {value[:16]}..."
        )
    if len(raw) != 32:
        raise GateError("approved base64 key must decode to 32 raw Ed25519 bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def key_fingerprint(public_key) -> str:
    """SHA-256 fingerprint over the raw 32-byte Ed25519 public key."""
    from cryptography.hazmat.primitives import serialization
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def verify_detached_signature(attestation_path: str, approved_keys, expected_fingerprint=None):
    """Verify a detached Ed25519 signature over the exact attestation bytes.

    approved_keys: list of CLI --approved-key values (PEM path or base64).
    expected_fingerprint: optional pinned SHA-256 hex of the raw public key;
    when set, every loaded key must either match it or the key is rejected as
    an unknown signer.
    """
    data = read_file_bytes(attestation_path)

    sig_path = attestation_path + ".sig"
    if not os.path.isfile(sig_path):
        raise GateError(f"missing detached signature file: {sig_path}")
    sig_b64 = read_file_bytes(sig_path).decode("utf-8", "replace").strip()
    try:
        signature = base64.b64decode(sig_b64, validate=True)
    except Exception:
        raise GateError(f"malformed detached signature in {sig_path} (not base64)")

    if not approved_keys:
        raise GateError("no approved key supplied (--approved-key); refusing attestation")

    from cryptography.exceptions import InvalidSignature

    matching_keys = []
    for value in approved_keys:
        key = _load_ed25519_public_key(value)
        fp = key_fingerprint(key)
        if expected_fingerprint is not None and fp != expected_fingerprint.lower():
            # Key fingerprint not pinned in the reviewed execution config.
            continue
        try:
            key.verify(signature, data)
            matching_keys.append(fp)
        except InvalidSignature:
            continue

    if not matching_keys:
        raise GateError(
            "signature verification failed: no approved key verified the "
            "detached signature (invalid signature or unknown signer)"
        )
    return matching_keys[0]


# ---------------------------------------------------------------------------
# Timestamp freshness
# ---------------------------------------------------------------------------

def parse_iso_utc(value, field: str) -> datetime.datetime:
    if not isinstance(value, str) or not value:
        raise GateError(f"missing or non-string {field}")
    ts = value
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(ts)
    except ValueError:
        raise GateError(f"unparseable {field}: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def check_freshness(iso_ts: str, field: str, stale_threshold: int,
                    future_skew: int) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    dt = parse_iso_utc(iso_ts, field)
    if dt > now + datetime.timedelta(seconds=future_skew):
        raise GateError(f"{field} is in the future: {iso_ts}")
    age = (now - dt).total_seconds()
    if age > stale_threshold:
        raise GateError(
            f"{field} is stale (age {int(age)}s > threshold {stale_threshold}s)"
        )


# ---------------------------------------------------------------------------
# Baseline attestation v2 validation
# ---------------------------------------------------------------------------

def _err(msg: str):
    raise GateError(msg)


def validate_baseline_attestation(att, stale_threshold: int, future_skew: int,
                                  approved_discrepancies) -> dict:
    """Validate structure, seven-instance inventory, identities, freshness.

    Returns a summary dict on success; raises GateError on any refusal.
    """
    if not isinstance(att, dict):
        _err("baseline attestation is not a JSON object")

    if att.get("schema_version") != BASELINE_SCHEMA_VERSION:
        _err(f"unsupported baseline schema_version: {att.get('schema_version')!r} "
             f"(expected {BASELINE_SCHEMA_VERSION!r})")

    for field in ("generated_at_utc", "source_audit_packages", "instances"):
        if field not in att:
            _err(f"baseline attestation missing required field: {field}")

    check_freshness(att["generated_at_utc"], "generated_at_utc",
                    stale_threshold, future_skew)

    packages = att.get("source_audit_packages")
    if not isinstance(packages, list) or not packages:
        _err("source_audit_packages must be a non-empty list")
    for pkg in packages:
        if not isinstance(pkg, dict):
            _err("source_audit_packages entries must be objects")
        for pf in ("package_id", "package_sha256"):
            if not is_nonempty_str(pkg.get(pf)):
                _err(f"source_audit_packages entry missing {pf}")

    instances = att["instances"]
    if not isinstance(instances, list):
        _err("instances must be a list")

    # Identity scan FIRST: unexpected and duplicated instance records must be
    # reported by name before any count-based refusal, so the gate error
    # names the actual defect rather than a generic count mismatch.
    seen_ids = set()
    for inst in instances:
        if not isinstance(inst, dict):
            _err("each instance must be an object")
        iid = inst.get("instance_id")
        if iid not in EXPECTED_INSTANCE_IDS:
            _err(f"unexpected or missing instance_id: {iid!r}")
        if iid in seen_ids:
            _err(f"duplicate instance_id: {iid}")
        seen_ids.add(iid)

    if len(instances) != len(EXPECTED_INSTANCES):
        _err(f"expected exactly {len(EXPECTED_INSTANCES)} MongoDB instances, "
             f"got {len(instances)}")

    missing = [i for i in EXPECTED_INSTANCE_IDS if i not in seen_ids]
    if missing:
        _err(f"missing MongoDB instance records: {missing}")

    seen_ids = set()
    seen_targets = set()
    by_id = {}
    for inst in instances:
        if not isinstance(inst, dict):
            _err("each instance must be an object")
        iid = inst.get("instance_id")
        if iid not in EXPECTED_INSTANCE_IDS:
            _err(f"unexpected or missing instance_id: {iid!r}")
        if iid in seen_ids:
            _err(f"duplicate instance_id: {iid}")
        seen_ids.add(iid)
        for f in ("namespace", "service_name", "service_uid", "application",
                  "database"):
            if not is_nonempty_str(inst.get(f)):
                _err(f"instance {iid} missing required field: {f}")

        expected_def = next(e for e in EXPECTED_INSTANCES
                            if e["instance_id"] == iid)
        if inst["database"] != expected_def["database"]:
            _err(f"instance {iid}: wrong database name "
                 f"{inst['database']!r} (expected {expected_def['database']!r})")

        target = (iid, inst["database"])
        if target in seen_targets:
            _err(f"duplicate MongoDB target: instance {iid} database {inst['database']}")
        seen_targets.add(target)

        # mongodb server identity/version where available
        if not is_nonempty_str(inst.get("mongodb_version", "")):
            _err(f"instance {iid} missing mongodb_version")

        cols = inst.get("collections")
        if not isinstance(cols, list):
            _err(f"instance {iid} missing collections list")
        col_names = set()
        for c in cols:
            if not isinstance(c, dict):
                _err(f"instance {iid}: collections entries must be objects")
            cname = c.get("name")
            if not is_nonempty_str(cname):
                _err(f"instance {iid}: collection missing name")
            if cname in col_names:
                _err(f"instance {iid}: duplicate collection record {cname}")
            col_names.add(cname)
            if "exists" not in c or not isinstance(c.get("exists"), bool):
                _err(f"instance {iid}/{cname}: missing boolean exists")
            if c["exists"]:
                if not is_strict_int(c.get("document_count")):
                    _err(f"instance {iid}/{cname}: existing collection requires "
                         f"integer document_count")
            else:
                # Absent collection: explicit absent state, count must be null
                if c.get("document_count") is not None:
                    _err(f"instance {iid}/{cname}: absent collection must have "
                         f"document_count null")
            if not isinstance(c.get("index_inventory_summary"), str):
                _err(f"instance {iid}/{cname}: missing index_inventory_summary")
            if not isinstance(c.get("identity_checks"), list):
                _err(f"instance {iid}/{cname}: missing identity_checks list")
            disc = c.get("discrepancy_ids", [])
            if not isinstance(disc, list):
                _err(f"instance {iid}/{cname}: discrepancy_ids must be a list")
            for d in disc:
                if d not in KNOWN_DISCREPANCY_IDS:
                    _err(f"unknown discrepancy identifier: {d!r} "
                         f"(instance {iid}/{cname})")

        # collection inventory must match the expected set exactly
        expected_cols = {c["name"]: c["exists"] for c in expected_def["collections"]}
        observed_cols = {}
        for c in cols:
            observed_cols[c["name"]] = c["exists"]
        if observed_cols != expected_cols:
            _err(f"instance {iid}: collection inventory mismatch "
                 f"(observed {sorted(observed_cols.items())}, "
                 f"expected {sorted(expected_cols.items())})")
        by_id[iid] = inst

    missing = [i for i in EXPECTED_INSTANCE_IDS if i not in seen_ids]
    if missing:
        _err(f"missing MongoDB instance records: {missing}")

    _validate_baseline_identities(by_id)

    # Discrepancies: observed must be a subset of approved and must include
    # the full known set.
    observed = set()
    for inst in instances:
        for c in inst.get("collections", []):
            for d in c.get("discrepancy_ids", []):
                observed.add(d)
    for d in KNOWN_DISCREPANCY_IDS:
        if d not in observed:
            _err(f"attestation does not report known discrepancy: {d}")
    for d in sorted(observed):
        if d not in approved_discrepancies:
            _err(f"observed discrepancy not approved by operator: {d}")

    return {
        "instances": sorted(seen_ids),
        "observed_discrepancies": sorted(observed),
    }


def _validate_baseline_identities(by_id: dict) -> None:
    """Require the initialization identity checks established by Stage 7.21.

    The order-other check is a ONE-ORDER BASELINE SENTINEL: Stage 7.22
    proved OrderOtherService.create() overwrites the deterministic input id
    with a random UUID before saving, so the document _id can never be used
    as the expected identity. accountId is a business field but is NOT
    unique in general (multiple orders can belong to one account), so the
    sentinel is the NARROW initialization-state claim: exactly one order
    document whose accountId equals the fdse_microservice userId.
    """
    # Document counts are recorded in the attestation but are NOT pinned to
    # the Stage 7.21 historical values here: a fresh audit is authoritative.
    user = by_id["ts-user-mongo"]["collections"][0]
    _require_identity(user, "userName", "fdse_microservice", present=False)

    auth = by_id["ts-auth-mongo"]["collections"][0]
    _require_identity(auth, "username", "fdse_microservice", present=True)
    _require_identity(auth, "username", "admin", present=True)

    contacts = by_id["ts-contacts-mongo"]["collections"][0]
    _require_identity(contacts, "name", "Contacts_One", present=True)
    _require_identity(contacts, "name", "Contacts_Two", present=True)

    orders = by_id["ts-order-mongo"]["collections"][0]
    for oid in ("5ad7750b-a68b-49c0-a8c0-32776b067703",
                "8177ac5a-61ac-42f4-83f4-bd7b394d0531",
                "d3c91694-d5b8-424c-9974-e14c89226e49"):
        _require_identity(orders, "_id", oid, present=True, note="binary_subtype_3")

    order_other = by_id["ts-order-other-mongo"]["collections"][0]
    # One-order baseline sentinel (Stage 7.25 erratum): accountId is NOT a
    # unique document identity — multiple orders can belong to one account.
    # The check pins the narrow initialization state instead: the collection
    # must contain EXACTLY one document (strict integer document_count == 1)
    # and that document must carry the expected accountId. The document _id
    # is never claimed: create() randomizes it (Stage 7.22).
    if not is_strict_int(order_other.get("document_count")) \
            or order_other["document_count"] != 1:
        _err("ts-order-other-mongo/orders one_order_baseline_sentinel: "
             "document_count must be the strict integer 1 (got "
             f"{order_other.get('document_count')!r})")
    _require_identity(order_other, "accountId", "fdse_microservice_userId",
                      present=True,
                      note="one_order_baseline_sentinel (exactly one document "
                           "with the expected accountId; _id randomized by "
                           "create())")

    payment = None
    for c in by_id["ts-payment-mongo"]["collections"]:
        if c["name"] == "payment":
            payment = c
    _require_identity(payment, "_id", "5ad7750ba68b49c0a8c035276b067701",
                      present=False)

    ipay = None
    iadd = None
    for c in by_id["ts-inside-payment-mongo"]["collections"]:
        if c["name"] == "payment":
            ipay = c
        if c["name"] == "addMoney":
            iadd = c
    _require_identity(ipay, "_id", "5ad7750ba68b49c0a8c035276b321701", present=True)
    _require_identity(iadd, "userId", "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
                      present=True)


def _require_identity(colrec, field: str, identity: str, present: bool,
                      note: str = None) -> None:
    if colrec is None:
        _err(f"missing collection record for identity check ({field}={identity})")
    checks = colrec.get("identity_checks", [])
    match = None
    for chk in checks:
        if not isinstance(chk, dict):
            _err("identity_checks entries must be objects")
        if chk.get("field") == field and chk.get("identity") == identity:
            match = chk
            break
    if match is None:
        _err(f"missing identity check: {field}={identity} "
             f"(collection {colrec.get('name')})")
    if match.get("present") is not present:
        _err(f"identity check mismatch: {field}={identity} reported "
             f"present={match.get('present')} but expected present={present}")
    if match.get("present") and note and note == "binary_subtype_3":
        if match.get("bson_type") != "binary_subtype_3":
            _err(f"identity check {field}={identity} must record bson_type "
                 f"binary_subtype_3")


# ---------------------------------------------------------------------------
# Backup attestation v2 validation
# ---------------------------------------------------------------------------

def validate_backup_attestation(att, stale_threshold: int, future_skew: int) -> dict:
    if not isinstance(att, dict):
        _err("backup attestation is not a JSON object")

    if att.get("schema_version") != BACKUP_SCHEMA_VERSION:
        _err(f"unsupported backup schema_version: {att.get('schema_version')!r} "
             f"(expected {BACKUP_SCHEMA_VERSION!r})")

    for field in ("generated_at_utc", "targets"):
        if field not in att:
            _err(f"backup attestation missing required field: {field}")

    check_freshness(att["generated_at_utc"], "generated_at_utc",
                    stale_threshold, future_skew)

    targets = att["targets"]
    if not isinstance(targets, list):
        _err("targets must be a list")

    # Identity scan FIRST (unexpected / duplicate / wrong database), then the
    # count, then missing targets — refusals name the actual defect.
    seen_targets = set()
    for t in targets:
        if not isinstance(t, dict):
            _err("each backup target must be an object")
        iid = t.get("instance_id")
        if iid not in EXPECTED_INSTANCE_IDS:
            _err(f"unexpected or missing backup instance_id: {iid!r}")
        if not is_nonempty_str(t.get("database")):
            _err(f"backup target {iid}: missing database")
        expected_def = next(e for e in EXPECTED_INSTANCES
                            if e["instance_id"] == iid)
        if t["database"] != expected_def["database"]:
            _err(f"backup target {iid}: wrong database name {t['database']!r} "
                 f"(expected {expected_def['database']!r})")
        key = (iid, t["database"])
        if key in seen_targets:
            _err(f"duplicate backup target: {key}")
        seen_targets.add(key)

    if len(targets) != len(EXPECTED_INSTANCES):
        _err(f"expected exactly {len(EXPECTED_INSTANCES)} backup targets, "
             f"got {len(targets)}")

    missing = [i for i in EXPECTED_INSTANCE_IDS
               if (i, next(e["database"] for e in EXPECTED_INSTANCES
                           if e["instance_id"] == i)) not in seen_targets]
    if missing:
        _err(f"missing backup targets: {missing}")

    seen_targets = set()
    for t in targets:
        if not isinstance(t, dict):
            _err("each backup target must be an object")
        iid = t.get("instance_id")
        if not is_nonempty_str(t.get("database")):
            _err(f"backup target {iid}: missing database")

        expected_def = next(e for e in EXPECTED_INSTANCES
                            if e["instance_id"] == iid)

        key = (iid, t["database"])
        if key in seen_targets:
            _err(f"duplicate backup target: {key}")
        seen_targets.add(key)

        for f in ("backup_method", "artifact_location", "artifact_sha256",
                  "mongodb_version", "off_cluster_storage_location",
                  "off_cluster_storage_verified", "restore_test_location",
                  "restore_tested_at_utc", "restore_result",
                  "restored_collection_counts", "evidence_reference",
                  "backup_created_at_utc"):
            if f not in t:
                _err(f"backup target {iid}: missing required field {f}")

        if not is_nonempty_str(t["backup_method"]):
            _err(f"backup target {iid}: empty backup_method")
        if not is_nonempty_str(t["artifact_location"]):
            _err(f"backup target {iid}: empty artifact_location")
        sha = t["artifact_sha256"]
        if not (isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdefABCDEF" for c in sha)):
            _err(f"backup target {iid}: artifact_sha256 must be 64 hex chars")
        if not is_nonempty_str(t["mongodb_version"]):
            _err(f"backup target {iid}: empty mongodb_version")

        # Off-cluster storage: an operator-asserted boolean plus a location.
        # The boolean is evidence supplied to the gate, NOT proof; the
        # operator must independently verify off-cluster storage.
        if t["off_cluster_storage_verified"] is not True:
            _err(f"backup target {iid}: off_cluster_storage_verified is not true")
        if not is_nonempty_str(t["off_cluster_storage_location"]):
            _err(f"backup target {iid}: empty off_cluster_storage_location")

        check_freshness(t["backup_created_at_utc"],
                        f"backup_created_at_utc ({iid})",
                        stale_threshold, future_skew)

        if not is_nonempty_str(t["restore_test_location"]):
            _err(f"backup target {iid}: empty restore_test_location")
        check_freshness(t["restore_tested_at_utc"],
                        f"restore_tested_at_utc ({iid})",
                        stale_threshold, future_skew)
        if t["restore_result"] != "verified":
            _err(f"backup target {iid}: restore_result is not 'verified'")
        counts = t["restored_collection_counts"]
        if not isinstance(counts, dict) or not counts:
            _err(f"backup target {iid}: restored_collection_counts must be a "
                 f"non-empty object")
        for cname, cnt in counts.items():
            if not is_strict_int(cnt) or cnt < 0:
                _err(f"backup target {iid}: restored count for {cname} must be "
                     f"a non-negative integer")
        # Collection-inventory cross-check (Stage 7.25 erratum): a backup
        # target is one instance/database pair; the collection inventory is
        # NESTED within each target. The restored counts must name EXACTLY
        # the collections that instance/database is expected to contain —
        # seven instance/database targets carrying the nine-entry inventory
        # (eight existing collections; ts-payment-mongo/ts/addMoney was
        # absent at backup time and restores as count 0).
        expected_names = {c["name"] for c in expected_def["collections"]}
        if set(counts.keys()) != expected_names:
            _err(f"backup target {iid}: restored_collection_counts keys must "
                 f"be exactly {sorted(expected_names)}, got "
                 f"{sorted(counts.keys())}")
        if not is_nonempty_str(t["evidence_reference"]):
            _err(f"backup target {iid}: empty evidence_reference")

    missing = [i for i in EXPECTED_INSTANCE_IDS
               if (i, next(e["database"] for e in EXPECTED_INSTANCES
                           if e["instance_id"] == i)) not in seen_targets]
    if missing:
        _err(f"missing backup targets: {missing}")

    return {"targets": sorted(k[0] for k in seen_targets)}


# ---------------------------------------------------------------------------
# Gate orchestration
# ---------------------------------------------------------------------------

def run_attestation_gate(baseline_path, backup_path, approved_keys,
                         expected_fingerprint, approved_discrepancies,
                         stale_threshold, future_skew):
    """Verify integrity, signature, schema, inventory, freshness, approval."""
    if not baseline_path or not backup_path:
        raise GateError("both --baseline-attestation and --backup-attestation "
                        "are required")

    # Integrity first: detached checksum over exact bytes.
    verify_detached_checksum(baseline_path)
    verify_detached_checksum(backup_path)

    # Authentication: detached Ed25519 signature against an approved key.
    verify_detached_signature(baseline_path, approved_keys, expected_fingerprint)
    verify_detached_signature(backup_path, approved_keys, expected_fingerprint)

    # Content.
    try:
        baseline = json.loads(read_file_bytes(baseline_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"baseline attestation is not valid JSON: {exc}")
    try:
        backup = json.loads(read_file_bytes(backup_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"backup attestation is not valid JSON: {exc}")

    if not approved_discrepancies:
        raise GateError("--approve-discrepancies is required: the operator must "
                        "explicitly approve the known discrepancy set")

    unknown_approved = [d for d in approved_discrepancies
                        if d not in KNOWN_DISCREPANCY_IDS]
    if unknown_approved:
        raise GateError(f"unknown discrepancy identifier in approval list: "
                        f"{unknown_approved}")

    bsum = validate_baseline_attestation(baseline, stale_threshold, future_skew,
                                         set(approved_discrepancies))
    bksum = validate_backup_attestation(backup, stale_threshold, future_skew)
    return baseline, backup, bsum, bksum


# ---------------------------------------------------------------------------
# Safe file writing helpers
# ---------------------------------------------------------------------------

def validate_writable_dir(path: str) -> None:
    if os.path.isdir(path):
        if not os.access(path, os.W_OK):
            raise PermissionError(f"directory not writable: {path}")
        return
    parent = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"destination directory does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"directory not writable: {parent}")


def atomic_write(path: str, data: bytes, mode: int) -> None:
    """Write bytes atomically with exact final permissions, then fsync.

    The temporary file is created with restrictive permissions from the very
    beginning (no window at group/world readability) and renamed over the
    destination in the same directory (same filesystem).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp-{os.getpid()}")
    fd = None
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_final = None
    except Exception:
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
    # Verify final permissions.
    st = os.stat(path)
    if stat_mod.S_IMODE(st.st_mode) != mode:
        os.chmod(path, mode)


class CheckpointWriter:
    """Append-only JSONL checkpoint with per-record fsync."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path)) or "."
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(self.fd, 0o600)

    def append(self, record: dict) -> None:
        if self.fd is None:
            raise OSError("checkpoint writer is closed")  # fail closed
        line = (json.dumps(record, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("utf-8")
        os.write(self.fd, line)
        os.fsync(self.fd)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def load_checkpoint_records(path: str):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"checkpoint {path} line {lineno} is not valid JSON: {exc}")
                if not isinstance(rec, dict):
                    raise ValueError(f"checkpoint {path} line {lineno} is not an object")
                records.append(rec)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint {path}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Logging (no secrets, ever)
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Seed generation (live)
# ---------------------------------------------------------------------------

def register_and_login_user(user, user_base, auth_base, timeout):
    """Run register + login for one user. Returns (combined, reg, login).

    Never retries a registration POST. Login POSTs are side-effect free and
    may be re-attempted by the caller on resume (documented policy).
    """
    http_post = _get_http_post()

    reg = None
    try:
        response = http_post(
            f"{user_base}/api/v1/userservice/users/register",
            json={"userName": user["username"], "password": user["password"],
                  "gender": 1, "documentType": 1,
                  "documentNum": user["documentNum"],
                  "email": f"{user['username']}@test.com"},
            timeout=timeout,
        )
        reg = parse_registration_response(response)
    except Exception as exc:
        # Transport failure/timeout: the POST may have been committed.
        reg = {
            "state": REG_OUTCOME_UNKNOWN,
            "http_status": None,
            "body_status": None,
            "error_category": f"transport_{type(exc).__name__}",
            "outcome_uncertain": True,
        }

    reg_state = reg["state"]
    if reg_state == REG_HTTP_FAILURE and reg.get("outcome_uncertain"):
        # 5xx: uncertain; do not attempt login with unverified credentials.
        return combine_outcome(reg, None), reg, None

    if reg_state not in (REG_REGISTERED, REG_DUPLICATE):
        # Definitive negative or unknown: no login attempted.
        return combine_outcome(reg, None), reg, None

    login = None
    try:
        response = http_post(
            f"{auth_base}/api/v1/users/login",
            json={"username": user["username"], "password": user["password"]},
            timeout=timeout,
        )
        login = parse_login_response(response, user["username"])
    except Exception as exc:
        login = {
            "state": LOGIN_OUTCOME_UNKNOWN,
            "http_status": None,
            "body_status": None,
            "error_category": f"transport_{type(exc).__name__}",
            "token": None,
            "user_id": None,
            "outcome_uncertain": True,
        }

    return combine_outcome(reg, login), reg, login


# ---------------------------------------------------------------------------
# Output artifact construction
# ---------------------------------------------------------------------------

def sanitized_plan_document(users, args, plan_sha: str) -> dict:
    """Dry-run plan WITHOUT credentials/PII (no passwords, document numbers,
    phone numbers, or tokens)."""
    return {
        "generator": GENERATOR_VERSION,
        "mode": "dry-run-plan",
        "count": len(users),
        "rng_seed": args.rng_seed,
        "plan_sha256": plan_sha,
        "contacts_semantics": "july_compatible_null_contactsId",
        "contact_creation": "disabled",
        "note": ("plan contains no credentials or PII; the per-user password "
                 "is supplied at runtime via the k6 PW environment variable "
                 "and is not recorded in this plan"),
        "requests_per_user": [
            f"POST {args.user_base}/api/v1/userservice/users/register",
            f"POST {args.auth_base}/api/v1/users/login",
        ],
        "users_preview_idx_username": [
            {"idx": u["idx"], "username": u["username"]} for u in users[:5]
        ],
    }


def build_results_manifest(users, outcomes, args, plan_sha, checkpoint_used):
    """Operational manifest: NO passwords, tokens, document numbers, phone
    numbers, payloads, response bodies, or unredacted auth errors."""
    counts = {}
    for o in outcomes.values():
        counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
    uncertain = sum(1 for o in outcomes.values() if o["uncertain"])
    login_retry = sum(1 for o in outcomes.values() if o["login_retry_allowed"])
    artifact_users = sorted(
        idx for idx, o in outcomes.items()
        if o["outcome"] in (REGISTERED_AND_LOGIN_VERIFIED,
                            DUPLICATE_AND_LOGIN_VERIFIED)
    )
    per_user = []
    for u in users:
        o = outcomes.get(u["idx"])
        if o is None:
            per_user.append({"idx": u["idx"], "username": u["username"],
                             "outcome": "NOT_PROCESSED",
                             "credential_artifact_written": False})
            continue
        per_user.append({
            "idx": u["idx"],
            "username": u["username"],
            "outcome": o["outcome"],
            "registration_http_status": o.get("reg_http_status"),
            "registration_body_status": o.get("reg_body_status"),
            "login_http_status": o.get("login_http_status"),
            "login_body_status": o.get("login_body_status"),
            "error_category": o.get("error_category"),
            "outcome_uncertain": o["uncertain"],
            "login_retry_allowed": o["login_retry_allowed"],
            "credential_artifact_written": u["idx"] in artifact_users,
        })
    return {
        "generator": GENERATOR_VERSION,
        "mode": "live-run",
        "rng_seed": args.rng_seed,
        "requested": args.count,
        "plan_sha256": plan_sha,
        "checkpoint_used": checkpoint_used,
        "outcome_counts": counts,
        "uncertain_outcomes": uncertain,
        "login_retry_allowed_count": login_retry,
        "credential_artifact_users": len(artifact_users),
        "usable_pool_size": len(artifact_users),
        "complete": len(users) == len(outcomes) and uncertain == 0
        and all(o["outcome"] in (REGISTERED_AND_LOGIN_VERIFIED,
                                 DUPLICATE_AND_LOGIN_VERIFIED)
                for o in outcomes.values()),
        "contacts_semantics": "july_compatible_null_contactsId",
        "users": per_user,
    }


def build_restricted_seed_artifact(users, outcomes, artifact_users):
    """Restricted k6 seed input: ONLY the fields the k6 script reads.

    load_testing/k6/trainticket.js state() reads s.username, s.accountId and
    s.contactsId from SEEDS entries and uses the GLOBAL PW env password; it
    performs its own per-VU logins and never reads tokens or per-user
    passwords from the seed file. This artifact therefore contains no
    password and no JWT.
    """
    entries = []
    for u in users:
        if u["idx"] not in artifact_users:
            continue
        o = outcomes[u["idx"]]
        entries.append({
            "idx": u["idx"],
            "username": u["username"],
            "accountId": o.get("login_user_id"),
            "contactsId": None,  # July-compatible semantics
        })
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Deterministic seed generator v4 (v4.2) — July-compatible "
                    "semantics with hardened execution gates")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--rng-seed", type=int, default=20260301)
    ap.add_argument("--out", help="dry-run plan output path (sanitized; no credentials)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--user-base", default="http://ts-user-service.train-ticket.svc.cluster.local:12342")
    ap.add_argument("--auth-base", default="http://ts-auth-service.train-ticket.svc.cluster.local:12340")
    ap.add_argument("--results", help="operational result manifest path (live; no credentials)")
    ap.add_argument("--seed-artifact", help="restricted k6 seed input path (live; created 0600)")
    ap.add_argument("--checkpoint", help="per-user JSONL checkpoint path (live; 0600)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --checkpoint; uncertain users are skipped, "
                         "never re-POSTed; confirmed registrations are never "
                         "re-registered (login-only retry)")
    ap.add_argument("--baseline-attestation")
    ap.add_argument("--backup-attestation")
    ap.add_argument("--approve-discrepancies",
                    help="comma-separated discrepancy IDs explicitly approved "
                         "by the operator (required for live and preflight)")
    ap.add_argument("--approved-key", action="append", default=[],
                    help="approved Ed25519 public key: PEM file path or "
                         "32-byte raw base64 (repeatable)")
    ap.add_argument("--expected-key-fingerprint",
                    help="pinned SHA-256 fingerprint of the approved raw "
                         "public key; keys with other fingerprints are "
                         "rejected as unknown signers")
    ap.add_argument("--write-authorize", action="store_true")
    ap.add_argument("--stale-threshold-seconds", type=int, default=86400)
    ap.add_argument("--future-skew-seconds", type=int, default=300)
    ap.add_argument("--timeout-seconds", type=int, default=30)
    return ap


def parse_discrepancy_arg(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv=None):
    args = build_parser().parse_args(argv)

    # --- CLI validation ---------------------------------------------------
    if args.count < 1:
        log("error: --count must be >= 1")
        return EXIT_CLI_FAILURE
    if args.stale_threshold_seconds < 0 or args.future_skew_seconds < 0 \
            or args.timeout_seconds <= 0:
        log("error: thresholds/timeout must be non-negative (timeout > 0)")
        return EXIT_CLI_FAILURE
    approved_discrepancies = parse_discrepancy_arg(args.approve_discrepancies)
    if args.approve_discrepancies is not None and not approved_discrepancies:
        # Flag present but empty -> CLI misuse (distinct from flag absent,
        # which is a gate refusal).
        log("error: --approve-discrepancies parsed to an empty list")
        return EXIT_CLI_FAILURE

    users = plan_users(args.count, args.rng_seed)
    plan_sha = hashlib.sha256(
        json.dumps(users, sort_keys=True).encode("utf-8")).hexdigest()

    # --- dry-run ------------------------------------------------------------
    if args.dry_run:
        if not args.out:
            log("error: --out is required for --dry-run")
            return EXIT_CLI_FAILURE
        try:
            validate_writable_dir(args.out)
            atomic_write(args.out,
                         json.dumps(sanitized_plan_document(users, args, plan_sha),
                                    indent=2).encode("utf-8"),
                         0o640)
        except (OSError, PermissionError) as exc:
            log(f"error: cannot write dry-run plan: {exc}")
            return EXIT_WRITE_FAILURE
        log(f"dry-run plan written: {args.out} (plan_sha256={plan_sha[:16]}...)")
        log("  plan sanitized: no passwords, tokens, document numbers, or phone numbers")
        return EXIT_SUCCESS

    # --- gate (preflight-only) -------------------------------------------------
    if args.preflight_only:
        try:
            _, _, bsum, bksum = run_attestation_gate(
                baseline_path=args.baseline_attestation,
                backup_path=args.backup_attestation,
                approved_keys=args.approved_key,
                expected_fingerprint=args.expected_key_fingerprint,
                approved_discrepancies=approved_discrepancies,
                stale_threshold=args.stale_threshold_seconds,
                future_skew=args.future_skew_seconds,
            )
        except GateError as exc:
            log(f"GATE_FAILURE: {exc}")
            return EXIT_GATE_FAILURE
        log("Preflight PASSED: baseline and backup attestations verified "
            f"({len(bsum['instances'])} instances, "
            f"{len(bksum['targets'])} backup targets, "
            f"discrepancies approved: {len(bsum['observed_discrepancies'])})")
        return EXIT_SUCCESS

    # --- live run: requires explicit write authorization ---------------------
    if not args.write_authorize:
        log("error: --write-authorize is required for live execution")
        return EXIT_CLI_FAILURE
    for req, name in ((args.results, "--results"),
                      (args.seed_artifact, "--seed-artifact"),
                      (args.checkpoint, "--checkpoint")):
        if not req:
            log(f"error: {name} is required for live execution")
            return EXIT_CLI_FAILURE

    # --- output pre-validation BEFORE any HTTP call ---------------------------
    try:
        validate_writable_dir(args.results)
        validate_writable_dir(args.seed_artifact)
        validate_writable_dir(args.checkpoint)
    except (OSError, PermissionError) as exc:
        log(f"OUTPUT_VALIDATION_FAILURE: {exc}")
        return EXIT_OUTPUT_VALIDATION_FAILURE

    if os.path.exists(args.seed_artifact) and not args.resume:
        log(f"OUTPUT_VALIDATION_FAILURE: restricted artifact already exists: "
            f"{args.seed_artifact} (refusing to overwrite without --resume)")
        return EXIT_OUTPUT_VALIDATION_FAILURE

    # --- gate before any artifact creation ------------------------------------
    try:
        _, _, bsum, bksum = run_attestation_gate(
            baseline_path=args.baseline_attestation,
            backup_path=args.backup_attestation,
            approved_keys=args.approved_key,
            expected_fingerprint=args.expected_key_fingerprint,
            approved_discrepancies=approved_discrepancies,
            stale_threshold=args.stale_threshold_seconds,
            future_skew=args.future_skew_seconds,
        )
    except GateError as exc:
        log(f"GATE_FAILURE: {exc}")
        return EXIT_GATE_FAILURE

    # --- checkpoint handling --------------------------------------------------
    outcomes = {}
    resumed_state = {}
    if args.resume:
        try:
            records = load_checkpoint_records(args.checkpoint)
        except FileNotFoundError:
            log(f"RESUME_CONFLICT: checkpoint file not found: {args.checkpoint}")
            return EXIT_RESUME_CONFLICT
        except ValueError as exc:
            log(f"RESUME_CONFLICT: {exc}")
            return EXIT_RESUME_CONFLICT
        for rec in records:
            idx = rec.get("idx")
            if not is_strict_int(idx):
                log("RESUME_CONFLICT: checkpoint record missing integer idx")
                return EXIT_RESUME_CONFLICT
            if rec.get("t") == "outcome":
                resumed_state.setdefault(idx, {})["outcome"] = rec
            elif rec.get("t") == "attempt":
                resumed_state.setdefault(idx, {})["attempt"] = rec
        for idx, st in resumed_state.items():
            rec = st.get("outcome")
            if rec is None:
                # attempt started but never terminalized: outcome unknown.
                outcomes[idx] = {
                    "outcome": REGISTRATION_OUTCOME_UNKNOWN,
                    "confirmed": False, "login_retry_allowed": False,
                    "uncertain": True,
                    "reg_http_status": None, "reg_body_status": None,
                    "login_http_status": None, "login_body_status": None,
                    "error_category": "interrupted_attempt_unknown",
                }
                continue
            outcomes[idx] = {
                "outcome": rec.get("outcome"),
                "confirmed": bool(rec.get("confirmed")),
                "login_retry_allowed": bool(rec.get("login_retry_allowed")),
                "uncertain": bool(rec.get("uncertain")),
                "reg_http_status": rec.get("reg_http_status"),
                "reg_body_status": rec.get("reg_body_status"),
                "login_http_status": rec.get("login_http_status"),
                "login_body_status": rec.get("login_body_status"),
                "login_user_id": rec.get("login_user_id"),
                "error_category": rec.get("error_category"),
            }
        log(f"Resume: {sum(1 for r in resumed_state.values() if 'outcome' in r)} "
            f"terminal records, "
            f"{sum(1 for r in resumed_state.values() if 'outcome' not in r)} "
            f"interrupted (treated as outcome-unknown, never re-POSTed)")
    else:
        if os.path.exists(args.checkpoint):
            try:
                if load_checkpoint_records(args.checkpoint):
                    log(f"RESUME_CONFLICT: non-empty checkpoint exists: "
                        f"{args.checkpoint} — use --resume or a new path")
                    return EXIT_RESUME_CONFLICT
            except ValueError as exc:
                log(f"RESUME_CONFLICT: {exc}")
                return EXIT_RESUME_CONFLICT

    # Create the restricted artifact skeleton (0600) BEFORE any HTTP call.
    try:
        if not os.path.exists(args.seed_artifact):
            atomic_write(args.seed_artifact, b"[]", 0o600)
        st = os.stat(args.seed_artifact)
        if stat_mod.S_IMODE(st.st_mode) != 0o600:
            os.chmod(args.seed_artifact, 0o600)
    except OSError as exc:
        log(f"OUTPUT_VALIDATION_FAILURE: cannot create restricted artifact: {exc}")
        return EXIT_OUTPUT_VALIDATION_FAILURE

    try:
        checkpoint = CheckpointWriter(args.checkpoint)
    except OSError as exc:
        log(f"OUTPUT_VALIDATION_FAILURE: cannot open checkpoint: {exc}")
        return EXIT_OUTPUT_VALIDATION_FAILURE

    # --- process users ---------------------------------------------------------
    log("Baseline and backup gates passed; starting live seed generation.")
    error_stop = None
    try:
        for u in users:
            idx = u["idx"]
            prior = outcomes.get(idx)

            if prior is not None and not prior["uncertain"] \
                    and not prior["login_retry_allowed"]:
                continue  # terminal, confirmed, nothing to do

            if prior is not None and prior["login_retry_allowed"] \
                    and prior["confirmed"]:
                # Confirmed registration: re-attempt LOGIN ONLY (safe: login
                # writes nothing). The registration POST is never re-sent.
                try:
                    response = _get_http_post()(
                        f"{args.auth_base}/api/v1/users/login",
                        json={"username": u["username"], "password": u["password"]},
                        timeout=args.timeout_seconds,
                    )
                    login = parse_login_response(response, u["username"])
                except Exception as exc:
                    login = {
                        "state": LOGIN_OUTCOME_UNKNOWN,
                        "http_status": None, "body_status": None,
                        "error_category": f"transport_{type(exc).__name__}",
                        "token": None, "user_id": None,
                        "outcome_uncertain": True,
                    }
                reg = {
                    "state": REG_REGISTERED if prior["outcome"] ==
                             REGISTRATION_SUCCESS_LOGIN_FAILURE else REG_DUPLICATE,
                    "http_status": prior.get("reg_http_status"),
                    "body_status": prior.get("reg_body_status"),
                    "error_category": None,
                    "outcome_uncertain": False,
                }
                combined = combine_outcome(reg, login)
                record = _checkpoint_record(idx, combined, reg, login)
                checkpoint.append({"t": "outcome", **record})
                outcomes[idx] = _outcome_from_combined(combined, reg, login)
                continue

            if prior is not None and prior["uncertain"]:
                continue  # never re-POST an uncertain registration

            # Fresh processing for this user: durable attempt marker FIRST.
            checkpoint.append({"t": "attempt", "idx": idx, "phase": "register"})
            combined, reg, login = register_and_login_user(
                u, args.user_base, args.auth_base, args.timeout_seconds)
            record = _checkpoint_record(idx, combined, reg, login)
            checkpoint.append({"t": "outcome", **record})
            outcomes[idx] = _outcome_from_combined(combined, reg, login)
    except KeyboardInterrupt:
        error_stop = "interrupted"
    finally:
        checkpoint.close()

    # --- finalize artifacts -----------------------------------------------------
    artifact_users = set(
        idx for idx, o in outcomes.items()
        if o["outcome"] in (REGISTERED_AND_LOGIN_VERIFIED,
                            DUPLICATE_AND_LOGIN_VERIFIED)
    )
    try:
        artifact_entries = build_restricted_seed_artifact(users, outcomes,
                                                          artifact_users)
        atomic_write(args.seed_artifact,
                     json.dumps(artifact_entries, indent=2).encode("utf-8"),
                     0o600)
        st = os.stat(args.seed_artifact)
        if stat_mod.S_IMODE(st.st_mode) != 0o600:
            raise PermissionError("restricted artifact lost 0600 permissions")
        atomic_write(args.results,
                     json.dumps(build_results_manifest(
                         users, outcomes, args, plan_sha,
                         checkpoint_used=bool(args.resume)), indent=2
                     ).encode("utf-8"), 0o640)
    except OSError as exc:
        log(f"WRITE_FAILURE: final artifact write failed: {exc}")
        log("Checkpoint remains durable; re-run with --resume to finalize.")
        return EXIT_WRITE_FAILURE

    if error_stop == "interrupted":
        log("Interrupted by operator; checkpoint durable. Resume with --resume.")
        return EXIT_PARTIAL_FAILURE

    uncertain = sum(1 for o in outcomes.values() if o["uncertain"])
    failures = sum(1 for o in outcomes.values()
                   if o["outcome"] not in (REGISTERED_AND_LOGIN_VERIFIED,
                                           DUPLICATE_AND_LOGIN_VERIFIED))
    unprocessed = sum(1 for u in users if u["idx"] not in outcomes)
    if unprocessed or uncertain:
        log(f"Live run incomplete: unprocessed={unprocessed}, "
            f"uncertain={uncertain}, failed={failures}")
        return EXIT_PARTIAL_FAILURE
    if failures:
        return EXIT_PARTIAL_FAILURE
    log(f"Live seed complete: {len(artifact_users)} usable users in restricted "
        f"artifact; results manifest at {args.results}")
    return EXIT_SUCCESS


def _checkpoint_record(idx, combined, reg, login):
    return {
        "idx": idx,
        "outcome": combined["outcome"],
        "confirmed": combined["confirmed"],
        "login_retry_allowed": combined["login_retry_allowed"],
        "uncertain": combined["uncertain"],
        "reg_http_status": reg.get("http_status"),
        "reg_body_status": reg.get("body_status"),
        "login_http_status": (login or {}).get("http_status"),
        "login_body_status": (login or {}).get("body_status"),
        "login_user_id": (login or {}).get("user_id"),
        "error_category": (login or {}).get("error_category") or reg.get("error_category"),
    }


def _outcome_from_combined(combined, reg, login):
    return {
        "outcome": combined["outcome"],
        "confirmed": combined["confirmed"],
        "login_retry_allowed": combined["login_retry_allowed"],
        "uncertain": combined["uncertain"],
        "reg_http_status": reg.get("http_status"),
        "reg_body_status": reg.get("body_status"),
        "login_http_status": (login or {}).get("http_status"),
        "login_body_status": (login or {}).get("body_status"),
        "login_user_id": (login or {}).get("user_id"),
        "error_category": (login or {}).get("error_category") or reg.get("error_category"),
    }


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Deterministic seed generator v3 for Stage 8 — July-compatible semantics.

Generates a 3,000-user seed that preserves the historical July workload behavior:

  - 3,000 users with deterministic usernames, document numbers, phone numbers
  - Valid authentication credentials (username + password) for each user
  - contactsId = null in the load-generator user state (no contact creation)
  - Empty contacts database, matching the historical July condition

This differs from seed_tt_users_v2.py which creates contact records for every
user (populating contactsId). The v2 behavior changes the execution paths of
a_preserve() and a_preserve_other() in trainticket_instr.js: with a populated
contactsId those functions proceed to preserve/preserveOther service calls;
with null contactsId they fall back to a_travel() — matching the historical
July behavior where the contacts database was empty and contactsId was null.

The exact original July user identities are unavailable. This generator
reproduces the relevant state and workload behavior, not the lost identities.

Usage:
  # Dry-run (offline, no API calls): generates a plan file only
  python3 seed_tt_users_v3_july_semantics.py \
    --count 3000 --rng-seed 20260301 --dry-run \
    --out seed_plan_v3_july.json

  # Preflight-only (offline, validates DB baseline, no writes):
  python3 seed_tt_users_v3_july_semantics.py \
    --count 3000 --rng-seed 20260301 --preflight-only

  # Live run (requires explicit approval + DB reset first):
  python3 seed_tt_users_v3_july_semantics.py \
    --count 3000 --rng-seed 20260301 \
    --out seed_users_v3_july.json

The live run performs an explicit baseline preflight before any writes.
It refuses unless the observed database state matches the verified pre-seed
baseline. There is no --allow-nonempty, --force, or equivalent bypass.
"""

import argparse
import hashlib
import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# HTTP transport hook (monkeypatchable for testing)
# ---------------------------------------------------------------------------
_http_post = None


def _get_http_post():
    """Return the HTTP POST callable, importing requests lazily on first use."""
    global _http_post
    if _http_post is None:
        import requests  # noqa: E402
        _http_post = requests.post
    return _http_post


GENERATOR_VERSION = "seed_tt_users_v3_july_semantics.py 1.4 (stage7.17 2026-11-10)"

# ---------------------------------------------------------------------------
# Exit code definitions (standardized across CLI, tests, and reports)
# ---------------------------------------------------------------------------
EXIT_CODES = {
    0: "SUCCESS — successful dry-run, successful preflight-only, or complete live seed",
    1: "PARTIAL_FAILURE — live seed completed with some registration/login failures; seed file written with partial state",
    2: "CLI_OR_PREFLIGHT_FAILURE — CLI argument failure (missing --out), or failed live preflight",
    3: "PREFLIGHT_ONLY_FAILURE — preflight-only execution failed",
    4: "WRITE_FAILURE — seed/plan output file could not be written",
}

# The following exit codes are returned by main():
#   0 — Preflight-only passed, or dry-run wrote plan, or live seed completed fully
#   1 — Live seed: partial registration/login failure (seed file written with partial state)
#   2 — Missing --out for dry-run/live, or preflight failed before any writes (live mode)
#   3 — Preflight-only failed (preflight-only mode)
#   4 — Seed file or plan file write failure (any mode that writes a file)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Verified pre-seed baseline (source-verified from TrainTicket InitData code)
#
# Every TrainTicket service uses MySQL (JPA/Hibernate), not MongoDB.
# Baselines below are derived from the service InitData.java / InitUser.java
# source files under the TrainTicket source tree.
#
# Identity fields are checked where the InitData code sets them explicitly.
# Records created with UUID.randomUUID() (e.g. auth admin) cannot be
# identity-checked and are verified by count only.
# ---------------------------------------------------------------------------

PRE_SEED_BASELINE = {
    "ts-user-mysql": {
        "host_env": "USER_MYSQL_HOST",
        "host_default": "ts-user-mysql",
        "port_env": "USER_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "USER_MYSQL_DATABASE",
        "db_default": "ts-user-mysql",
        "user_env": "USER_MYSQL_USER",
        "user_default": "root",
        "password_env": "USER_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-user-mysql",
            "port": 3306,
            "database": "ts-user-mysql",
        },
        "tables": {
            "user": {
                "expected_count": 1,
                "expected_identities": [
                    {
                        "user_name": "fdse_microservice",
                        "user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
                    }
                ],
                "identity_source": "ts-user-service/src/main/java/user/init/InitUser.java",
                "column_note": "JPA @Column(name='user_id') and @Column(name='user_name') — NOT userId/userName",
            },
        },
    },
    "ts-auth-mysql": {
        "host_env": "AUTH_MYSQL_HOST",
        "host_default": "ts-auth-mysql",
        "port_env": "AUTH_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "AUTH_MYSQL_DATABASE",
        "db_default": "ts-auth-mysql",
        "user_env": "AUTH_MYSQL_USER",
        "user_default": "root",
        "password_env": "AUTH_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-auth-mysql",
            "port": 3306,
            "database": "ts-auth-mysql",
        },
        "tables": {
            "auth_user": {
                "expected_count": 2,
                "expected_identities": [
                    {"user_name": "fdse_microservice"},
                    {"user_name": "admin"},
                ],
                "identity_source": "ts-auth-service/src/main/java/auth/init/InitUser.java",
                "column_note": "JPA @Column(name='user_name') — NOT username",
            },
        },
    },
    "ts-contacts-mysql": {
        "host_env": "CONTACTS_MYSQL_HOST",
        "host_default": "ts-contacts-mysql",
        "port_env": "CONTACTS_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "CONTACTS_MYSQL_DATABASE",
        "db_default": "ts-contacts-mysql",
        "user_env": "CONTACTS_MYSQL_USER",
        "user_default": "root",
        "password_env": "CONTACTS_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-contacts-mysql",
            "port": 3306,
            "database": "ts-contacts-mysql",
        },
        "tables": {
            "contacts": {
                "expected_count": 2,
                "expected_identities": [
                    {"name": "Contacts_One", "id": "4d2a46c7-71cb-4cf1-a5bb-b68406d9da6f"},
                    {"name": "Contacts_Two", "id": "4d2546c7-71cb-4cf1-a5bb-b68406d9da6f"},
                ],
                "identity_source": "ts-contacts-service/src/main/java/contacts/init/InitData.java",
            },
        },
    },
    "ts-order-mysql": {
        "host_env": "ORDER_MYSQL_HOST",
        "host_default": "ts-order-mysql",
        "port_env": "ORDER_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "ORDER_MYSQL_DATABASE",
        "db_default": "ts",
        "user_env": "ORDER_MYSQL_USER",
        "user_default": "root",
        "password_env": "ORDER_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-order-mysql",
            "port": 3306,
            "database": "ts",
        },
        "tables": {
            "orders": {
                "expected_count": 3,
                "expected_identities": [
                    {"id": "5ad7750b-a68b-49c0-a8c0-32776b067703"},
                    {"id": "8177ac5a-61ac-42f4-83f4-bd7b394d0531"},
                    {"id": "d3c91694-d5b8-424c-9974-e14c89226e49"},
                ],
                "identity_source": "ts-order-service/src/main/java/order/init/InitData.java",
            },
        },
    },
    "ts-order-other-mysql": {
        "host_env": "ORDER_OTHER_MYSQL_HOST",
        "host_default": "ts-order-other-mysql",
        "port_env": "ORDER_OTHER_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "ORDER_OTHER_MYSQL_DATABASE",
        "db_default": "ts-order-other-mysql",
        "user_env": "ORDER_OTHER_MYSQL_USER",
        "user_default": "root",
        "password_env": "ORDER_OTHER_MYSQL_PASSWORD",
        "password_default": "root",
        "endpoint_expected": {
            "host": "ts-order-other-mysql",
            "port": 3306,
            "database": "ts-order-other-mysql",
        },
        "tables": {
            "orders_other": {
                "expected_count": 1,
                "expected_identities": [
                    {"id": "4d2a46c7-71cb-4cf1-c5bb-b68406d9da6f"},
                ],
                "identity_source": "ts-order-other-service/src/main/java/other/init/InitData.java",
                "column_note": "JPA @Table(name='orders_other') — NOT 'orders'",
            },
        },
    },
    "ts-payment-mysql": {
        "host_env": "PAYMENT_MYSQL_HOST",
        "host_default": "ts-payment-mysql",
        "port_env": "PAYMENT_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "PAYMENT_MYSQL_DATABASE",
        "db_default": "ts-payment-mysql",
        "user_env": "PAYMENT_MYSQL_USER",
        "user_default": "root",
        "password_env": "PAYMENT_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-payment-mysql",
            "port": 3306,
            "database": "ts-payment-mysql",
        },
        "tables": {
            "payment": {
                "expected_count": 1,
                "expected_identities": [
                    {"id": "5ad7750ba68b49c0a8c035276b067701"},
                ],
                "identity_source": "ts-payment-service/src/main/java/com/trainticket/init/InitData.java",
            },
        },
    },
    "ts-inside-payment-mysql": {
        "host_env": "INSIDE_PAYMENT_MYSQL_HOST",
        "host_default": "ts-inside-payment-mysql",
        "port_env": "INSIDE_PAYMENT_MYSQL_PORT",
        "port_default": 3306,
        "db_env": "INSIDE_PAYMENT_MYSQL_DATABASE",
        "db_default": "ts-inside-payment-mysql",
        "user_env": "INSIDE_PAYMENT_MYSQL_USER",
        "user_default": "root",
        "password_env": "INSIDE_PAYMENT_MYSQL_PASSWORD",
        "password_default": "Abcd1234#",
        "endpoint_expected": {
            "host": "ts-inside-payment-mysql",
            "port": 3306,
            "database": "ts-inside-payment-mysql",
        },
        "tables": {
            "inside_money": {
                "expected_count": 1,
                "expected_identities": [
                    {"user_id": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"},
                ],
                "identity_source": "ts-inside-payment-service/src/main/java/inside_payment/init/InitData.java",
                "column_note": "JPA @Entity Money, field userId has no @Column; Hibernate implicit naming -> user_id",
            },
            "inside_payment": {
                "expected_count": 1,
                "expected_identities": [
                    {"id": "5ad7750ba68b49c0a8c035276b321701"},
                ],
                "identity_source": "ts-inside-payment-service/src/main/java/inside_payment/init/InitData.java",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Database connection layer (injectable for testing)
# ---------------------------------------------------------------------------

def get_mysql_connection_factory():
    """Return a callable that opens a pymysql connection to a given DB config.

    Imported lazily so the module can be loaded without pymysql installed.
    Tests inject a mock factory via set_connection_factory().
    The `table_name` kwarg is accepted but ignored by the real factory;
    mock factories may use it to route to canned rows.
    """
    import pymysql  # noqa: PLC0415  (lazy import, only needed for live runs)

    def connect(db_cfg, table_name=""):
        return pymysql.connect(
            host=db_cfg["host"],
            port=db_cfg["port"],
            user=db_cfg["user"],
            password=db_cfg["password"],
            db=db_cfg["database"],
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor,
        )
    return connect


_connection_factory = None


def set_connection_factory(factory):
    """Inject a connection factory (used by tests)."""
    global _connection_factory
    _connection_factory = factory


def _connect(db_cfg, table_name=""):
    """Open a connection using the configured (or injected) factory."""
    if _connection_factory is not None:
        return _connection_factory(db_cfg, table_name=table_name)
    return get_mysql_connection_factory()(db_cfg)


def db_config(db_key):
    """Build a connection config dict from PRE_SEED_BASELINE entry + env overrides."""
    entry = PRE_SEED_BASELINE[db_key]
    return {
        "host": os.environ.get(entry["host_env"], entry["host_default"]),
        "port": int(os.environ.get(entry["port_env"], entry["port_default"])),
        "user": os.environ.get(entry["user_env"], entry["user_default"]),
        "password": os.environ.get(entry["password_env"], entry["password_default"]),
        "database": os.environ.get(entry["db_env"], entry["db_default"]),
    }


# ---------------------------------------------------------------------------
# Plan generation (unchanged seed semantics)
# ---------------------------------------------------------------------------

def plan_users(count: int, rng_seed: int):
    """Generate deterministic user plans.

    Returns a list of user dicts with:
      idx, username, password, documentNum, phoneNumber
    No contactsId — contacts are not created (July-compatible semantics).
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
# Baseline preflight
# ---------------------------------------------------------------------------

def preflight_check(verbose=True):
    """Inspect every store in PRE_SEED_BASELINE and verify it matches.

    Prints observed vs expected for every checked collection.
    Returns (ok, results) where ok is True only if every table matches.
    results is a list of per-table result dicts.

    Fails closed: any mismatch, missing table, or connection error => ok=False.
    No database writes are performed.

    Endpoint validation: before checking table contents, the connection
    config (host, port, database) is compared against endpoint_expected
    from PRE_SEED_BASELINE. A mismatch fails closed for all tables in
    that store.

    Identity comparison: values are compared case-sensitively in Python
    after fetching from the database, because MySQL default collation is
    case-insensitive and would otherwise produce false positives.
    """
    results = []
    all_ok = True

    for db_key in sorted(PRE_SEED_BASELINE.keys()):
        entry = PRE_SEED_BASELINE[db_key]
        cfg = db_config(db_key)

        # --- Endpoint identity validation (host, port, database) ---
        endpoint_cfg = entry.get("endpoint_expected", {})
        endpoint_error = None
        if endpoint_cfg:
            if cfg["host"] != endpoint_cfg.get("host"):
                endpoint_error = (
                    f"host mismatch: expected={endpoint_cfg.get('host')}, "
                    f"observed={cfg['host']}"
                )
            elif cfg["port"] != endpoint_cfg.get("port"):
                endpoint_error = (
                    f"port mismatch: expected={endpoint_cfg.get('port')}, "
                    f"observed={cfg['port']}"
                )
            elif cfg["database"] != endpoint_cfg.get("database"):
                endpoint_error = (
                    f"database mismatch: expected={endpoint_cfg.get('database')}, "
                    f"observed={cfg['database']}"
                )

        for table_name, table_cfg in entry["tables"].items():
            result = {
                "database": db_key,
                "table": table_name,
                "host": cfg["host"],
                "port": cfg["port"],
                "expected_count": table_cfg["expected_count"],
                "observed_count": None,
                "count_match": False,
                "expected_identities": table_cfg.get("expected_identities", []),
                "observed_identities": [],
                "identity_match": False,
                "identity_source": table_cfg.get("identity_source", "unknown"),
                "error": None,
            }

            # Endpoint mismatch fails closed before any table query
            if endpoint_error:
                result["error"] = endpoint_error
                result["count_match"] = False
                result["identity_match"] = False
                all_ok = False
                results.append(result)
                continue

            try:
                conn = _connect(cfg, table_name=table_name)
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
                        row = cur.fetchone()
                        observed_count = row["cnt"] if row else 0
                        result["observed_count"] = observed_count
                        result["count_match"] = (
                            observed_count == table_cfg["expected_count"]
                        )

                        if table_cfg.get("expected_identities"):
                            # Build a SELECT that checks for each expected identity
                            observed = []
                            for ident in table_cfg["expected_identities"]:
                                where_parts = []
                                params = []
                                for col, val in ident.items():
                                    where_parts.append(f"`{col}` = %s")
                                    params.append(val)
                                where_sql = " AND ".join(where_parts)
                                cur.execute(
                                    f"SELECT * FROM {table_name} WHERE {where_sql}",
                                    tuple(params),
                                )
                                rows = cur.fetchall()
                                # Case-sensitive comparison in Python:
                                # MySQL default collation is case-insensitive,
                                # so we compare values explicitly after fetch.
                                found = any(
                                    all(
                                        str(row.get(col)) == str(val)
                                        for col, val in ident.items()
                                    )
                                    for row in rows
                                )
                                observed.append(found)
                            result["observed_identities"] = observed
                            result["identity_match"] = all(observed)

                    conn.close()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    raise
            except Exception as exc:
                result["error"] = str(exc)
                result["count_match"] = False
                result["identity_match"] = False

            if not result["count_match"]:
                all_ok = False
            if table_cfg.get("expected_identities") and not result["identity_match"]:
                all_ok = False

            results.append(result)

    if verbose:
        print("=== Pre-seed baseline preflight ===")
        for r in results:
            status = "OK" if (r["count_match"] and r["identity_match"] and not r["error"]) else "FAIL"
            print(f"  [{status}] {r['database']}.{r['table']} "
                  f"(@{r['host']}:{r['port']})")
            print(f"    expected_count={r['expected_count']} "
                  f"observed_count={r['observed_count']} "
                  f"count_match={r['count_match']}")
            if r["expected_identities"]:
                print(f"    expected_identities={r['expected_identities']}")
                print(f"    observed_identities={r['observed_identities']} "
                      f"identity_match={r['identity_match']}")
            if r["error"]:
                print(f"    error: {r['error']}")
            if r["identity_source"] != "unknown":
                print(f"    source: {r['identity_source']}")
        print(f"=== Preflight {'PASSED' if all_ok else 'FAILED'} ===")

    return all_ok, results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic seed generator v3 — July-compatible "
                    "semantics (no contact creation, null contactsId)"
    )
    ap.add_argument("--count", type=int, default=3000,
                    help="number of users to generate")
    ap.add_argument("--rng-seed", type=int, default=20260301,
                    help="fixed RNG seed -> reproducible user plans")
    ap.add_argument("--out", required=False,
                    help="output plan/seed file path (required for --dry-run and live run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit plan only; no API calls")
    ap.add_argument("--preflight-only", action="store_true",
                    help="validate the pre-seed DB baseline and exit; no writes")
    ap.add_argument("--user-base", default="http://ts-user-service.train-ticket.svc.cluster.local:12342")
    ap.add_argument("--auth-base", default="http://ts-auth-service.train-ticket.svc.cluster.local:12340")
    args = ap.parse_args(argv)

    # Preflight-only mode: validate baseline and exit (no --out needed)
    if args.preflight_only:
        ok, _ = preflight_check(verbose=True)
        return 0 if ok else 3

    # --dry-run and live run both require --out
    if not args.out:
        print("error: --out is required for --dry-run and live run", file=sys.stderr)
        return 2

    users = plan_users(args.count, args.rng_seed)
    plan_sha = hashlib.sha256(
        json.dumps(users, sort_keys=True).encode()).hexdigest()

    if args.dry_run:
        out = {
            "generator": GENERATOR_VERSION,
            "mode": "dry-run-plan",
            "count": args.count,
            "rng_seed": args.rng_seed,
            "plan_sha256": plan_sha,
            "contacts_semantics": "july_compatible_null_contactsId",
            "contact_creation": "disabled",
            "requests_per_user": [
                f"POST {args.user_base}/api/v1/userservice/users/register "
                f"{{userName, password:'loadtest123', gender:1, documentType:1, documentNum, email}}",
                f"POST {args.auth_base}/api/v1/users/login {{username, password}} -> token,userId",
                # No contact creation endpoint — July-compatible
            ],
            "exclusion_rule": "none (all users have valid credentials; contactsId=null)",
            "users_first_5": users[:5],
            "users_last_2": users[-2:],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"dry-run plan written: {args.out} (plan_sha256={plan_sha[:16]}...)")
        print(f"  users: {args.count}")
        print(f"  contactsId: null for all users (July-compatible)")
        print(f"  contact creation: disabled")
        return 0

    # --- Live run: explicit baseline preflight before ANY writes ---
    # The preflight must pass before any registration or API call occurs.
    preflight_ok, preflight_results = preflight_check(verbose=True)
    if not preflight_ok:
        print("REFUSING live run: pre-seed baseline validation failed. "
              "The database must be reset to the exact approved baseline "
              "before seed generation. There is no bypass option.",
              file=sys.stderr)
        # Print a compact summary of failures
        for r in preflight_results:
            if not (r["count_match"] and r["identity_match"]) or r["error"]:
                print(f"  FAILED: {r['database']}.{r['table']} "
                      f"(expected={r['expected_count']}, "
                      f"observed={r['observed_count']}, "
                      f"error={r['error']})", file=sys.stderr)
        return 2

    print("Pre-seed baseline validated. Proceeding with live seed generation.",
          file=sys.stderr)

    # Live run path (requires explicit approval; intentionally not executed
    # during Stage 7.16). Registers and authenticates each user but does NOT
    # create contacts. contactsId remains null in the seed file.
    http_post = _get_http_post()
    seeded = []
    failed = []
    for u in users:
        try:
            r = http_post(f"{args.user_base}/api/v1/userservice/users/register",
                          json={"userName": u["username"], "password": u["password"],
                                "gender": 1, "documentType": 1,
                                "documentNum": u["documentNum"],
                                "email": f"{u['username']}@test.com"}, timeout=30)
            if r.status_code not in (200, 201):
                print(f"[{u['idx']}] register HTTP {r.status_code}", flush=True)
                failed.append({"idx": u["idx"], "stage": "register",
                               "status": r.status_code})
                continue
            r = http_post(f"{args.auth_base}/api/v1/users/login",
                          json={"username": u["username"], "password": u["password"]},
                          timeout=30)
            if r.status_code not in (200, 201):
                print(f"[{u['idx']}] login HTTP {r.status_code}", flush=True)
                failed.append({"idx": u["idx"], "stage": "login",
                               "status": r.status_code})
                continue
            d = r.json()["data"]
            tok, aid = d["token"], d["userId"]
            # contactsId remains None — no contact creation (July-compatible)
            seeded.append({
                "username": u["username"],
                "password": u["password"],
                "accountId": aid,
                "token": tok,
                "contactsId": None,
            })
        except Exception as exc:
            print(f"[{u['idx']}] exception: {exc}", flush=True)
            failed.append({"idx": u["idx"], "stage": "exception", "error": str(exc)})
    # Write the seed file regardless — partial state must be recorded
    try:
        with open(args.out, "w") as f:
            json.dump({"generator": GENERATOR_VERSION, "rng_seed": args.rng_seed,
                       "requested": args.count, "seeded": len(seeded),
                       "failed": len(failed),
                       "failure_details": failed,
                       "complete": len(failed) == 0,
                       "contacts_semantics": "july_compatible_null_contactsId",
                       "users": seeded}, f)
    except Exception as exc:
        print(f"FATAL: cannot write seed file {args.out}: {exc}", file=sys.stderr)
        print(f"PARTIAL SEED STATE: {len(seeded)} users registered in DB, "
              f"{len(failed)} failed — no seed file written", file=sys.stderr)
        return 4
    if failed:
        print(f"PARTIAL SEED: {len(seeded)}/{args.count} seeded, "
              f"{len(failed)} failed -> {args.out}", file=sys.stderr)
        for f in failed[:10]:
            print(f"  FAILED idx={f['idx']} stage={f['stage']}", file=sys.stderr)
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more", file=sys.stderr)
        return 1
    print(f"seeded {len(seeded)}/{args.count} -> {args.out}")
    print(f"  contactsId: null for all {len(seeded)} users (July-compatible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
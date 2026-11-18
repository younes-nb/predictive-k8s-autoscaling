#!/usr/bin/env python3
"""Deterministic seed generator v2 for Stage 8 (Task B / Task G).

Differences from load_testing/seed_tt_users.py (kept for provenance):
  - deterministic usernames, documentNums and phone numbers via a fixed RNG seed
    (--rng-seed) so the 3,000-user plan is byte-reproducible;
  - --dry-run: no HTTP calls; emits the planned user list and the exact request
    sequence that would be executed against the cluster;
  - hard output gating: refuses to run live against a non-empty user DB unless
    --allow-nonempty is given (the operator is expected to run the approved
    reset first);
  - records generator version + arguments inside the seed file itself.

Register/login/contact-creation flow is IDENTICAL to the July-era generator
(seed_tt_users.py): POST userservice/users/register -> POST users/login ->
POST contactservice/contacts. Users whose contact creation fails are EXCLUDED
from the output seed file (uniform seed-state requirement: every seed user
either has a working contact or is not in the pool).
"""
import argparse
import hashlib
import json
import random
import sys

GENERATOR_VERSION = "seed_tt_users_v2.py 1.0 (stage7 closure 2026-11-10)"


def plan_users(count: int, rng_seed: int):
    rng = random.Random(rng_seed)
    users = []
    for i in range(count):
        uname = f"seeduser{i}_{rng.randint(0, 999999)}"
        docnum = rng.randint(10**9, 10**10 - 1)
        phone = f"138{rng.randint(10**7, 10**8 - 1)}"
        users.append({"idx": i, "username": uname, "password": "loadtest123",
                      "documentNum": docnum, "phoneNumber": phone})
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--rng-seed", type=int, default=20260301,
                    help="fixed RNG seed -> reproducible usernames/docnums")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="emit plan + request sequence only; no HTTP")
    ap.add_argument("--allow-nonempty", action="store_true")
    ap.add_argument("--user-base", default="http://ts-user-service.train-ticket.svc.cluster.local:12342")
    ap.add_argument("--auth-base", default="http://ts-auth-service.train-ticket.svc.cluster.local:12340")
    ap.add_argument("--contacts-base", default="http://ts-contacts-service.train-ticket.svc.cluster.local:12347")
    args = ap.parse_args()

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
            "requests_per_user": [
                f"POST {args.user_base}/api/v1/userservice/users/register "
                f"{{userName, password:'loadtest123', gender:1, documentType:1, documentNum, email}}",
                f"POST {args.auth_base}/api/v1/users/login {{username, password}} -> token,userId",
                f"POST {args.contacts_base}/api/v1/contactservice/contacts (Bearer) "
                f"{{accountId, name:'Tester', documentType:1, documentNumber, phoneNumber}} -> contactsId",
            ],
            "exclusion_rule": "users whose contact creation fails are excluded from the seed file",
            "users_first_5": users[:5],
            "users_last_2": users[-2:],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"dry-run plan written: {args.out} (plan_sha256={plan_sha[:16]}...)")
        return 0

    if not args.allow_nonempty:
        print("REFUSING live run: the live seed-generation requires the approved DB reset "
              "first (user DB must be empty). Pass --allow-nonempty only after the reset "
              "has been executed and verified.", file=sys.stderr)
        return 2

    # live run (requires approval per Stage 7 constraints; flow identical to July
    # generator).  Implemented but intentionally NOT executed during Stage 7.
    import requests  # noqa: E402  (only needed for the live path)
    seeded = []
    for u in users:
        r = requests.post(f"{args.user_base}/api/v1/userservice/users/register",
                          json={"userName": u["username"], "password": u["password"],
                                "gender": 1, "documentType": 1,
                                "documentNum": u["documentNum"],
                                "email": f"{u['username']}@test.com"}, timeout=30)
        if r.status_code not in (200, 201):
            print(f"[{u['idx']}] register {r.status_code}", flush=True)
            continue
        r = requests.post(f"{args.auth_base}/api/v1/users/login",
                          json={"username": u["username"], "password": u["password"]},
                          timeout=30)
        d = r.json()["data"]
        tok, aid = d["token"], d["userId"]
        c = requests.post(f"{args.contacts_base}/api/v1/contactservice/contacts",
                          headers={"Authorization": f"Bearer {tok}"},
                          json={"accountId": aid, "name": "Tester", "documentType": 1,
                                "documentNumber": str(u["documentNum"]),
                                "phoneNumber": u["phoneNumber"]}, timeout=30)
        cid = c.json().get("data", {}).get("id") if c.status_code in (200, 201) else None
        if cid is None:
            print(f"[{u['idx']}] contact FAILED -> excluded", flush=True)
            continue
        seeded.append({"username": u["username"], "password": u["password"],
                       "accountId": aid, "token": tok, "contactsId": cid})
    with open(args.out, "w") as f:
        json.dump({"generator": GENERATOR_VERSION, "rng_seed": args.rng_seed,
                   "requested": args.count, "seeded": len(seeded), "users": seeded}, f)
    print(f"seeded {len(seeded)}/{args.count} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

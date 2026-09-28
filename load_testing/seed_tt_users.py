#!/usr/bin/env python3
"""Pre-seed train-ticket load users: register + login + contacts.

Writes JSON [{username, password, accountId, token, contactsId}] for the
locust pool (avoids the on-start registration herd that timeouts auth).
Run from repo root (needs cluster access for the service URLs? No - it
writes the script; execute via a k8s Job or port-forward). Default simply
prints the kubectl job manifest.

Usage:
    python load_testing/seed_tt_users.py --count 120 --out /tmp/tt_users.json
    # then: kubectl -n loadgen create configmap tt-users --from-file=users.json=...
"""
import argparse
import json
import random
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--base", default="http://ts-auth-service.train-ticket.svc.cluster.local:12340",
                    help="auth base when run in-cluster; use port-forward base otherwise")
    ap.add_argument("--user-base", default="http://ts-user-service.train-ticket.svc.cluster.local:12342")
    ap.add_argument("--contacts-base", default="http://ts-contacts-service.train-ticket.svc.cluster.local:12347")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    users = []
    for i in range(args.count):
        uname = f"seeduser{i}_{random.randint(0, 999999)}"
        pw = "loadtest123"
        try:
            r = requests.post(f"{args.user_base}/api/v1/userservice/users/register",
                              json={"userName": uname, "password": pw, "gender": 1,
                                    "documentType": 1,
                                    "documentNum": f"{random.randint(10**9, 10**10 - 1)}",
                                    "email": f"{uname}@test.com"}, timeout=30)
            if r.status_code not in (200, 201):
                print(f"[{i}] register {r.status_code}", flush=True)
                continue
            r = requests.post(f"{args.base}/api/v1/users/login",
                              json={"username": uname, "password": pw}, timeout=30)
            d = r.json()["data"]
            tok, aid = d["token"], d["userId"]
            c = requests.post(
                f"{args.contacts_base}/api/v1/contactservice/contacts",
                headers={"Authorization": f"Bearer {tok}"},
                json={"accountId": aid, "name": "Tester", "documentType": 1,
                      "documentNumber": f"{random.randint(10**9, 10**10 - 1)}",
                      "phoneNumber": f"138{random.randint(10**7, 10**8 - 1)}"},
                timeout=30)
            cid = c.json().get("data", {}).get("id") if c.status_code in (200, 201) else None
            users.append({"username": uname, "password": pw, "accountId": aid,
                          "token": tok, "contactsId": cid})
            print(f"[{i}] ok {uname}", flush=True)
        except Exception as e:
            print(f"[{i}] FAIL {e}", flush=True)
    with open(args.out, "w") as f:
        json.dump(users, f)
    print(f"seeded {len(users)}/{args.count} -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())

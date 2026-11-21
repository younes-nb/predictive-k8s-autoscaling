#!/usr/bin/env python3
"""Parallel TrainTicket seed generator.

The upstream load_testing/seed_tt_users.py registers users one at a time, which
is fine for a few hundred accounts but takes hours for the ~12k needed when the
VU cap is derived from MAX_REQUESTS instead of from the seed count.

Same three calls per user, same output schema (idx / username / accountId /
contactsId), but N workers and bounded retries.
"""
import argparse
import json
import random
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PW = "loadtest123"
_lock = threading.Lock()
_ok = 0
_fail = 0


def post(url, payload, token=None, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode() or "{}")


def one_user(args, i):
    uname = f"seeduser{i}_{random.randint(0, 999999)}"
    # register (retry: the first call after a cold reset can hit a stale
    # mongo pool and return 500)
    for _ in range(3):
        try:
            st, _ = post(f"{args.base}/api/v1/userservice/users/register",
                         {"userName": uname, "password": PW, "gender": 1,
                          "documentType": 1,
                          "documentNum": f"{random.randint(10**9, 10**10-1)}",
                          "email": f"{uname}@test.com"})
            if st in (200, 201):
                break
        except Exception:
            pass
    else:
        raise RuntimeError("register failed")

    st, d = post(f"{args.base}/api/v1/users/login",
                 {"username": uname, "password": PW})
    if st != 200 or not d.get("data"):
        raise RuntimeError(f"login failed status={st}")
    tok, aid = d["data"]["token"], d["data"]["userId"]

    cid = None
    try:
        st, c = post(f"{args.base}/api/v1/contactservice/contacts",
                     {"accountId": aid, "name": "Tester", "documentType": 1,
                      "documentNumber": f"{random.randint(10**9, 10**10-1)}",
                      "phoneNumber": f"138{random.randint(10**7, 10**8-1)}"},
                     token=tok)
        if st in (200, 201):
            cid = (c.get("data") or {}).get("id")
    except Exception:
        pass
    return {"idx": i, "username": uname, "accountId": aid, "contactsId": cid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://10.10.1.3:30080")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start-idx", type=int, default=1)
    a = ap.parse_args()

    users = [None] * a.count
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one_user, a, i): i
                for i in range(a.start_idx, a.start_idx + a.count)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                users[i - a.start_idx] = f.result()
                with _lock:
                    global _ok
                    _ok += 1
            except Exception as exc:
                with _lock:
                    global _fail
                    _fail += 1
                if _fail <= 5:
                    print(f"[idx {i}] FAIL {exc}", file=sys.stderr)
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{a.count} done (ok={_ok} fail={_fail})",
                      file=sys.stderr, flush=True)

    good = [u for u in users if u]
    with open(a.out, "w") as fh:
        json.dump(good, fh)
    print(f"seeded {len(good)}/{a.count} -> {a.out} (failed={_fail})")


if __name__ == "__main__":
    main()
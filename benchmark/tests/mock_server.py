#!/usr/bin/env python3
"""Async TrainTicket stand-in for generator validation.

Serves enough of the API for load_testing/k6/trainticket.js: a real
HS256-shaped JWT from /api/v1/users/login and 200s elsewhere. Built on asyncio
because the 2,999-VU sizing test needs a few hundred requests per second and
ThreadingHTTPServer becomes the bottleneck.

Config (environment):
  PORT              listen port (default 18099)
  LATENCY_MS        base latency for every response (default 0)
  LOGIN_LATENCY_MS  extra latency for /api/v1/users/login (default 0)
  FAIL_PCT          percentage of non-login requests answered 503 (default 0)
  TOKEN_TTL         token lifetime in seconds (default 3600). TEST ONLY: the
                    population renewal test shortens it so many cycles fit in a
                    short run. The production contract is always 3600.
  LOG               set to 1 to log every request

Never sends traffic anywhere; it only listens.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import uuid

PORT = int(os.environ.get("PORT", "18099"))
LATENCY_MS = float(os.environ.get("LATENCY_MS", "0"))
LOGIN_LATENCY_MS = float(os.environ.get("LOGIN_LATENCY_MS", "0"))
FAIL_PCT = float(os.environ.get("FAIL_PCT", "0"))
TOKEN_TTL = int(os.environ.get("TOKEN_TTL", "3600"))
LOG = os.environ.get("LOG") == "1"
SECRET = b"synth-secret-not-a-real-key"

counter = {"requests": 0, "logins": 0, "failures": 0}


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_jwt(username: str, ttl: int) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": username, "roles": ["ROLE_USER"],
               "id": str(uuid.uuid4()), "iat": now, "exp": now + ttl}
    h = b64url(json.dumps(header, separators=(",", ":")).encode())
    p = b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(SECRET, ("%s.%s" % (h, p)).encode(), hashlib.sha256).digest()
    return "%s.%s.%s" % (h, p, b64url(sig))


def body_for(path: str) -> bytes:
    if "travelservice/trips" in path:
        return b"[]"                       # k6 asserts Array.isArray
    if "orders" in path and path.endswith("/orders"):
        return json.dumps({"status": 1, "msg": "ok", "data": []}).encode()
    return json.dumps({"status": 1, "msg": "ok", "data": []}).encode()


async def handle(reader, writer):
    try:
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=30)
            except (asyncio.TimeoutError, ConnectionError):
                break
            if not line:
                break
            try:
                method, path, _ = line.decode().split(" ", 2)
            except ValueError:
                break
            length = 0
            while True:
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
                if h.lower().startswith(b"content-length:"):
                    try:
                        length = int(h.split(b":", 1)[1].strip())
                    except ValueError:
                        length = 0
            payload = await reader.readexactly(length) if length else b""

            counter["requests"] += 1
            is_login = path.endswith("/api/v1/users/login")

            if FAIL_PCT > 0 and not is_login and (counter["requests"] % 100) < FAIL_PCT:
                counter["failures"] += 1
                status, obj = 503, {"status": 0, "msg": "synthetic failure"}
            elif is_login:
                counter["logins"] += 1
                username = "user"
                if payload:
                    try:
                        username = json.loads(payload).get("username", "user")
                    except Exception:
                        pass
                obj = {"status": 1, "msg": "login success",
                       "data": {"userId": str(uuid.uuid4()),
                                "username": username,
                                "token": make_jwt(username, TOKEN_TTL)}}
                status = 200
            else:
                status, obj = 200, None
                if "travelservice/trips" not in path:
                    obj = {"status": 1, "msg": "ok", "data": []}

            body = body_for(path) if obj is None else json.dumps(obj).encode()

            delay = (LOGIN_LATENCY_MS if is_login else LATENCY_MS) / 1000.0
            if delay > 0:
                await asyncio.sleep(delay)

            head = (
                "HTTP/1.1 %d %s\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "X-Request-Id: %s\r\n"
                "\r\n"
                % (status, "OK" if status == 200 else "Service Unavailable",
                   len(body), str(uuid.uuid4()))
            ).encode()
            writer.write(head + body)
            await writer.drain()

            if LOG:
                print("%s %s -> %d" % (method, path, status), flush=True)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def stats():
    while True:
        await asyncio.sleep(30)
        print("stats: %s" % counter, flush=True)


async def main():
    # backlog must exceed the VU count: the default 100 drops SYNs once
    # thousands of VUs connect simultaneously, which looks like a generator
    # stall rather than a server limitation.
    server = await asyncio.start_server(handle, "127.0.0.1", PORT, backlog=8192)
    print("synth listening on 127.0.0.1:%d latency=%sms login=%sms fail=%s%% "
          "ttl=%ds" % (PORT, LATENCY_MS, LOGIN_LATENCY_MS, FAIL_PCT, TOKEN_TTL),
          flush=True)
    asyncio.create_task(stats())
    async with server:
        await server.serve_forever()


asyncio.run(main())
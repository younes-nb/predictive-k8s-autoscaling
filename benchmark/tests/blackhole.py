#!/usr/bin/env python3
"""Blackhole TCP listener: accepts connections and never sends a byte.

Simulates the failure the ts-user-service liveness probe must catch: the JVM is
alive and the listening socket is open (so a tcpSocket probe would pass), but
the application connector never answers (so curl gets no HTTP status).
"""
import socket
import sys
import threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 12399
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT))
srv.listen(200)
print("blackhole listening on 127.0.0.1:%d (accepts, never responds)" % PORT,
      flush=True)

held = []


def hold(conn):
    held.append(conn)   # keep it open, send nothing, ever


while True:
    conn, _ = srv.accept()
    threading.Thread(target=hold, args=(conn,), daemon=True).start()
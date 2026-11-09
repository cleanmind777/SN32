#!/usr/bin/env python3
"""Keep a public TCP port on bore.pub forwarded to the local miner axon."""

from __future__ import annotations

import json
import socket
import sys
import threading

REMOTE_HOST = "159.223.110.159"
CONTROL_PORT = 7835
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8091


def send_frame(sock: socket.socket, obj) -> None:
    sock.sendall(json.dumps(obj, separators=(",", ":")).encode() + b"\0")


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for item in (src, dst):
            try:
                item.close()
            except OSError:
                pass


def handle(conn_id: str) -> None:
    remote = socket.create_connection((REMOTE_HOST, CONTROL_PORT), timeout=15)
    send_frame(remote, {"Accept": conn_id})
    try:
        local = socket.create_connection((LOCAL_HOST, LOCAL_PORT), timeout=15)
    except OSError as exc:
        print(f"local axon unavailable: {exc}", flush=True)
        remote.close()
        return
    threading.Thread(target=pipe, args=(remote, local), daemon=True).start()
    threading.Thread(target=pipe, args=(local, remote), daemon=True).start()


def main() -> None:
    requested = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ctrl = socket.create_connection((REMOTE_HOST, CONTROL_PORT), timeout=15)
    ctrl.settimeout(None)
    send_frame(ctrl, {"Hello": requested})
    buf = b""
    while b"\0" not in buf:
        chunk = ctrl.recv(256)
        if not chunk:
            raise SystemExit("bore control closed during hello")
        buf += chunk
    frame, buf = buf.split(b"\0", 1)
    hello = json.loads(frame.decode())
    if not isinstance(hello, dict) or "Hello" not in hello:
        raise SystemExit(f"unexpected hello: {hello}")
    print(f"REMOTE_IP={REMOTE_HOST}", flush=True)
    print(f"REMOTE_PORT={hello['Hello']}", flush=True)
    while True:
        while b"\0" not in buf:
            chunk = ctrl.recv(256)
            if not chunk:
                print("CONTROL_EOF", flush=True)
                return
            buf += chunk
        frame, buf = buf.split(b"\0", 1)
        msg = json.loads(frame.decode())
        if msg == "Heartbeat" or msg == {"Heartbeat": None}:
            continue
        if isinstance(msg, dict) and "Connection" in msg:
            threading.Thread(target=handle, args=(msg["Connection"],), daemon=True).start()
        elif isinstance(msg, dict) and "Error" in msg:
            print(f"SERVER_ERROR {msg['Error']}", flush=True)
            return
        else:
            print(f"OTHER {msg}", flush=True)


if __name__ == "__main__":
    main()

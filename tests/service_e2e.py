#!/usr/bin/env python3
"""End-to-end check of the worker's background service on this machine (Linux systemd, macOS launchd, Windows
Startup item). Standard library only.

It starts a stand-in ORBIT API (worker download, installers, pairing that approves at once, heartbeat and poll),
then twice — once with `--install-service`, once through the real one-line installer (`curl … | sh` or
`irm … | iex`) — installs the worker as a service pointed at it, waits for heartbeats that can only come from the
service-run worker (the installer exits right after installing), kills it to check it is restarted (systemd,
launchd, or the supervisor behind the Windows Startup item), uninstalls, and checks it stops.

    python worker/tests/service_e2e.py [--home DIR]

--home redirects HOME (and the settings dir) to a scratch directory so a developer machine's real setup is
untouched; CI runners use their own home.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "orbit_worker.py"
STATE = {"heartbeats": 0, "versions": set(), "polls": 0, "paired": 0, "url": ""}


class FakeOrbit(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        try:
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):  # the worker was killed mid-poll (the restart check)
            pass

    def _text(self, text: str):
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json({"object": "list", "data": [{"id": "stub-model"}]})
        if self.path == "/worker/orbit_worker.py":
            return self._text(WORKER.read_text(encoding="utf-8"))
        if self.path in ("/install.sh", "/install.ps1"):
            return self._text((WORKER.parent / self.path[1:]).read_text(encoding="utf-8").replace("__API__", STATE["url"]))
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/network/pair/start":
            return self._json({"device_code": "orbd_test", "user_code": "TEST-2345", "expires_in": 60, "interval": 1,
                               "verification_url": f"{STATE['url']}/pair?code=TEST-2345"})
        if self.path == "/api/network/pair/poll":  # approved straight away: the browser step is tested elsewhere
            STATE["paired"] += 1
            return self._json({"status": "approved", "token": "orbw_service_test", "provider": "service-test",
                               "visibility": "private", "dashboard_url": f"{STATE['url']}/network"})
        if self.headers.get("Authorization") != "Bearer orbw_service_test":
            return self._json({"detail": "bad token"}, 401)
        if self.path == "/api/network/worker/heartbeat":
            STATE["heartbeats"] += 1
            STATE["versions"].add(body.get("version"))
            return self._json({"ok": True, "provider": "service-test", "paused": False, "visibility": "private"})
        if self.path.startswith("/api/network/worker/poll"):
            STATE["polls"] += 1
            time.sleep(1)
            return self._json({"job": None})
        self._json({"error": "not found"}, 404)


def wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise SystemExit(f"FAIL: timed out waiting for {what}")


def kill_and_expect_restart(settings: Path, pid: str) -> None:
    before = STATE["heartbeats"]
    os.kill(int(pid), signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)

    def restarted():
        try:
            new = (settings / "worker.pid").read_text().strip()
        except OSError:
            return False
        return new != pid and STATE["heartbeats"] > before
    wait_for(restarted, 60, "the killed worker to be restarted")
    print(f"ok: killed pid {pid}; it was restarted as pid {(settings / 'worker.pid').read_text().strip()}")


def run_supervisor_case(settings: Path, env: dict, url: str) -> None:
    """The supervisor the Windows Startup item uses, exercised directly (on any OS)."""
    print("--- worker --supervise")
    STATE.update(heartbeats=0, polls=0)
    proc = subprocess.Popen([sys.executable, str(WORKER), "--home", str(settings), "--orbit-url", url, "--token",
                             "orbw_service_test", "--upstream", f"{url}/v1", "--supervise"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for(lambda: STATE["heartbeats"] >= 1 and (settings / "worker.pid").exists(), 60, "the supervised worker")
    pid = (settings / "worker.pid").read_text().strip()
    assert pid != str(proc.pid), "the worker runs as a child of the supervisor"
    kill_and_expect_restart(settings, pid)
    if sys.platform == "win32":
        # What --uninstall-service does for the Startup item: it leaves a stop file the supervisor watches for.
        settings.mkdir(parents=True, exist_ok=True)
        (settings / "stop").write_text("")
    else:
        proc.terminate()  # the supervisor stops its worker on the way out
    proc.wait(30)
    time.sleep(3)
    before = STATE["polls"]
    time.sleep(8)
    if STATE["polls"] != before:
        raise SystemExit("FAIL: the worker kept running after its supervisor stopped")
    print("ok: stopping the supervisor stopped the worker")


def run_case(name: str, install_cmd: list[str], settings: Path, env: dict) -> None:
    print(f"--- {name}")
    STATE.update(heartbeats=0, polls=0)
    install = subprocess.run(install_cmd, env=env, capture_output=True, text=True, timeout=180)
    print(install.stdout.strip(), install.stderr.strip(), sep="\n")
    if install.returncode != 0:
        raise SystemExit(f"FAIL: install exited with {install.returncode}")
    wait_for(lambda: STATE["heartbeats"] >= 1 and STATE["polls"] >= 1, 90, "the service-run worker to connect")
    print(f"ok: service worker connected ({STATE['heartbeats']} heartbeat(s), {STATE['polls']} poll(s), "
          f"{', '.join(sorted(v for v in STATE['versions'] if v))})")
    wait_for(lambda: (settings / "worker.pid").exists(), 30, "the pid file")
    pid = (settings / "worker.pid").read_text().strip()
    print(f"ok: worker running as pid {pid}")
    # systemd and launchd restart a worker that dies; on Windows the Startup item's supervisor does
    kill_and_expect_restart(settings, pid)
    out = subprocess.run([sys.executable, str(settings / "orbit_worker.py"), "--home", str(settings),
                          "--uninstall-service"], env=env, capture_output=True, text=True, timeout=60)
    print(out.stdout.strip())
    if "Removed" not in out.stdout:
        raise SystemExit("FAIL: uninstall did not remove a service")
    time.sleep(3)
    before = STATE["polls"]
    time.sleep(8)
    if STATE["polls"] != before:
        raise SystemExit(f"FAIL: the worker kept polling after uninstall ({before} -> {STATE['polls']})")
    print("ok: worker stopped after uninstall")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--home", help="scratch HOME for the test (default: the real one)")
    p.add_argument("--only", choices=["supervisor", "direct", "installer"])
    args = p.parse_args()
    env = dict(os.environ)
    scratch = Path(args.home or tempfile.mkdtemp(prefix="orbit-svc-"))
    if args.home:
        scratch.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(scratch)
        if sys.platform == "win32":
            env["USERPROFILE"] = str(scratch)
            env["APPDATA"] = str(scratch / "AppData" / "Roaming")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = ThreadingHTTPServer(("127.0.0.1", port), FakeOrbit)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = STATE["url"] = f"http://127.0.0.1:{port}"

    if args.only in (None, "supervisor"):
        run_supervisor_case(scratch / ".orbit-supervisor-test", env, url)
    if args.only in (None, "direct"):
        settings = scratch / ".orbit-service-test"
        run_case("worker --install-service", [sys.executable, str(WORKER), "--home", str(settings), "--orbit-url", url,
                                              "--token", "orbw_service_test", "--upstream", f"{url}/v1",
                                              "--install-service"], settings, env)
    if args.only in (None, "installer"):
        # The installers put everything in ~/.orbit; the worker finds the upstream from UPSTREAM_URL.
        home_dir = Path(env.get("USERPROFILE") if sys.platform == "win32" else env.get("HOME") or Path.home())
        settings = home_dir / ".orbit"
        inst_env = dict(env, UPSTREAM_URL=f"{url}/v1", ORBIT_URL=url)
        cmd = (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"irm {url}/install.ps1 | iex"]
               if sys.platform == "win32" else ["sh", "-c", f"curl -fsSL {url}/install.sh | sh -s -- --no-browser"])
        run_case("one-line installer", cmd, settings, inst_env)
        if STATE["paired"] < 1:
            raise SystemExit("FAIL: the installer did not pair")
        print("ok: the installer paired the worker with no token copied by hand")
    server.shutdown()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

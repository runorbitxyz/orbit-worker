#!/usr/bin/env python3
"""ORBIT compute worker: serve inference from your own GPU to the ORBIT network (or privately, for yourself).

It connects a model server you already run (Ollama, LM Studio, vLLM, llama.cpp, or anything OpenAI-compatible) to
ORBIT. It asks ORBIT for jobs, runs each one on your model server, and sends the answer back.

What it does
  - Connects only outward: to the ORBIT server you choose (over HTTPS) and to your model server. It opens no ports,
    so you need no port forwarding.
  - Sends your model server one kind of request: a chat completion for a model you serve. It builds that request
    itself from a fixed list of fields, caps its length, and turns streaming off.
  - Sends ORBIT the model server's answer (only the standard parts of a chat completion), the list of models you
    serve, the kind of model server (e.g. "Ollama", never its address), and the machine description you see during
    pairing (OS, GPU, memory, CPU). Not your machine's name, your files, or anything else about you.
  - Keeps its settings in ~/.orbit (worker.json, readable only by you, holds your worker token) and logs there
    when it runs as a service.

What it never does
  - It never runs commands, code or tools that ORBIT or a model sends. Tool calls are text it passes back.
  - It never reads, uploads or changes your files. It reads only its own settings and, on Linux, /proc/meminfo and
    /proc/cpuinfo to describe your hardware. It never logs prompts or answers.
  - It never updates itself. The code you read is the code that runs.
  - It never sends your worker token anywhere but the ORBIT server. It refuses plain-HTTP ORBIT URLs (except on
    this machine or your local network) and does not follow redirects.
  - It never sends your model server's API key anywhere but the model server you name.

The only programs it starts: nvidia-smi and sysctl to describe your hardware; and, only when you ask it to run in
the background, your OS's own service tools (systemctl and loginctl on Linux, launchctl on macOS, PowerShell to write a
Startup shortcut on Windows, where it also runs a copy of itself as a supervisor).

Standard library only (Python 3.9+). Quick start, which finds your model server and pairs in the browser:

    curl -fsSL https://<orbit>/install.sh | sh

Or run it yourself:

    python3 orbit_worker.py --orbit-url https://<orbit>              # pair, then serve
    python3 orbit_worker.py --orbit-url https://<orbit> --setup      # pair, run in the background, exit
    python3 orbit_worker.py --uninstall-service                      # stop the background worker and remove it

Options can also come from the environment: ORBIT_URL, ORBIT_WORKER_TOKEN, UPSTREAM_URL, UPSTREAM_API_KEY,
WORKER_MODELS, WORKER_CONCURRENCY, WORKER_MAX_TOKENS, ORBIT_HOME.

Source, releases and checksums: https://github.com/runorbitxyz/orbit-worker
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Optional

VERSION = "orbit-worker/0.4.1"
DEFAULT_ORBIT_URL = "https://api.runorbit.xyz"  # the public ORBIT network
log = logging.getLogger("orbit-worker")

# Local model servers tried, in order, when --upstream is "auto".
KNOWN_UPSTREAMS = [
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("vLLM", "http://localhost:8000/v1"),
    ("llama.cpp", "http://localhost:8080/v1"),
]
# The only request fields passed to your model server (standard OpenAI chat-completion fields).
CHAT_FIELDS = ("messages", "tools", "tool_choice", "temperature", "top_p", "stop", "seed", "response_format",
               "parallel_tool_calls")
# The only parts of your model server's answer passed back to ORBIT.
REPLY_FIELDS = ("id", "object", "created", "model")
MESSAGE_FIELDS = ("role", "content", "tool_calls", "reasoning_content", "reasoning", "refusal")
DEFAULT_MAX_TOKENS = 4096  # longest answer generated for one job, unless you set --max-tokens
MAX_REPLY_BYTES = 16 * 2**20  # largest reply read from ORBIT or the model server
JOB_TIMEOUT = (10, 600)  # seconds a job may run on your model server: ORBIT's deadline, kept within this range


class HttpError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {printable(body)}")
        self.status = status
        self.body = body


class JobRefused(Exception):
    """A job this worker will not run (e.g. for a model you don't serve)."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Redirects are refused: following one would resend the Authorization header to wherever it points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HttpError(code, f"redirected to {newurl}; use that address directly")


_opener = urllib.request.build_opener(_NoRedirects)


def request_json(method: str, url: str, body: Optional[dict] = None, headers: Optional[dict] = None,
                 timeout: float = 60) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json", "User-Agent": VERSION, **(headers or {})})
    try:
        with _opener.open(req, timeout=timeout) as resp:
            raw = resp.read(MAX_REPLY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read(2000).decode(errors="replace")) from None
    if len(raw) > MAX_REPLY_BYTES:
        raise ValueError(f"reply larger than {MAX_REPLY_BYTES // 2**20} MB")
    try:
        return json.loads(raw) if raw else None
    except RecursionError:  # nested deeper than Python's parser goes: as invalid as any other bad reply
        raise ValueError("reply nested too deeply") from None


def printable(value: Any, limit: int = 200) -> str:
    """Text from a server, safe to print or log: control characters (terminal escape codes, newlines) removed."""
    return "".join(ch if ch.isprintable() else " " for ch in str(value))[:limit]


def is_local(host: str) -> bool:
    """This machine or your local network: localhost, a private address, a .local name, or a single-label name
    (e.g. a Docker service)."""
    if not host:
        return False
    if host == "localhost" or host.endswith(".local") or ("." not in host and ":" not in host):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private or addr.is_link_local


def clean_reply(data: Any) -> dict[str, Any]:
    """What goes back to ORBIT: the answer, why it stopped, and the token counts. Anything else your model server adds
    to its reply (timings, server details, extra fields) stays on your machine."""
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if first is None or not isinstance(first.get("message"), dict):
        raise ValueError("the model server's reply is not a chat completion")
    message = {k: first["message"][k] for k in MESSAGE_FIELDS if k in first["message"]}
    out: dict[str, Any] = {k: data[k] for k in REPLY_FIELDS if isinstance(data.get(k), (str, int))}
    out["choices"] = [{"index": 0, "message": message, "finish_reason": first.get("finish_reason")}]
    usage = data.get("usage")
    if isinstance(usage, dict):
        out["usage"] = {k: (v if isinstance(v, int) else {dk: dv for dk, dv in v.items() if isinstance(dv, int)})
                        for k, v in usage.items()
                        if isinstance(v, int) or (k.endswith("_details") and isinstance(v, dict))}
    return out


def server_kind(upstream: str) -> str:
    """What pairing tells ORBIT about your model server: its kind if it is a well-known local one, never its address."""
    return next((name for name, url in KNOWN_UPSTREAMS if url == upstream.rstrip("/")), "custom server")


def check_orbit_url(url: str) -> str:
    """The worker token is a credential: only send it over HTTPS, or plain HTTP to this machine / your network."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and is_local(parts.hostname or "")):
        return url.rstrip("/")
    raise SystemExit(f"Refusing to send your worker token to {url}: use an https:// ORBIT URL.")


# --- this machine --------------------------------------------------------------------------------------
def home() -> Path:
    return Path(os.environ.get("ORBIT_HOME") or Path.home() / ".orbit")


def load_config() -> dict[str, Any]:
    try:
        return json.loads((home() / "worker.json").read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict[str, Any]) -> Path:
    """Write settings readable only by you (they include the worker token), atomically."""
    path = home() / "worker.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # private from the moment it exists
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(tmp, 0o600)  # in case an older file existed with wider permissions
    tmp.replace(path)
    return path


def list_models(upstream: str, key: str = "", timeout: float = 5) -> list[str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = request_json("GET", f"{upstream.rstrip('/')}/models", headers=headers, timeout=timeout)
    return [m["id"] for m in (data or {}).get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)]


def detect_upstream() -> Optional[tuple[str, str, list[str]]]:
    """The first local model server that lists at least one model. No API key is sent while looking."""
    for name, url in KNOWN_UPSTREAMS:
        try:
            models = list_models(url, timeout=2)
        except Exception:
            continue
        if models:
            return name, url, models
    return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_hardware() -> dict[str, str]:
    """OS, GPU, memory and CPU, shown on your rig's card so people can see what serves them."""
    hw: dict[str, str] = {"os": f"{platform.system()} {platform.machine()}"}
    if shutil.which("nvidia-smi"):
        rows = [r.split(",") for r in _run(["nvidia-smi", "--query-gpu=name,memory.total",
                                            "--format=csv,noheader"]).splitlines() if r.strip()]
        if rows:
            names = sorted({r[0].strip() for r in rows})
            hw["gpu"] = f"{len(rows)}x {' + '.join(names)}" if len(rows) > 1 else names[0]
            hw["vram"] = rows[0][1].strip() if len(rows[0]) > 1 else ""
    if platform.system() == "Darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        if chip:
            hw.setdefault("gpu", chip if "Apple" in chip else "")
            hw["cpu"] = chip
        if mem.isdigit():
            hw["memory"] = f"{int(mem) / 2**30:.0f} GB" + (" unified" if "Apple" in chip else "")
    elif Path("/proc/meminfo").exists():
        try:
            kb = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
            hw["memory"] = f"{kb / 2**20:.0f} GB"
            cpu = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name")]
            if cpu:
                hw["cpu"] = cpu[0]
        except (OSError, IndexError, ValueError):
            pass
    hw["cores"] = str(os.cpu_count() or "?")
    return {k: v for k, v in hw.items() if v}


# --- pairing -------------------------------------------------------------------------------------------
def pair(orbit_url: str, models: list[str], upstream: str, visibility: Optional[str] = None,
         open_browser: bool = True) -> dict[str, Any]:
    """Device pairing: show a link and a code, wait for you to approve it in the browser, return the token. What is
    sent: the models, the hardware description, the worker version, the OS and the kind of model server; you see all
    of it on the approval page. Not the machine's name (it often carries a person's) or the server's address."""
    start = request_json("POST", f"{orbit_url}/api/network/pair/start", {
        "models": models, "hardware": detect_hardware(), "version": VERSION, "upstream": server_kind(upstream),
        "platform": platform.system(), "visibility": visibility}, timeout=20)
    url, code = printable(start["verification_url"], 300), printable(start["user_code"], 20)
    bar = "─" * 58
    print(f"\n{bar}\n  Pair this machine with your ORBIT account\n\n  1. Open   {url}\n  2. Check the code   {code}\n"
          f"  3. Approve (public to earn $ORBIT, or private to serve only you)\n{bar}\n", flush=True)
    if open_browser and sys.stdout.isatty() and urllib.parse.urlsplit(url).scheme in ("https", "http"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    deadline = time.monotonic() + min(float(start.get("expires_in", 900)), 3600)
    interval = min(max(float(start.get("interval", 3)), 1), 30)
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            reply = request_json("POST", f"{orbit_url}/api/network/pair/poll", {"device_code": start["device_code"]},
                                 timeout=20)
        except HttpError as exc:
            if exc.status in (404, 410):
                raise SystemExit(f"Pairing failed: {printable(exc.body)}")
            continue
        except OSError:
            continue
        if reply.get("status") == "approved":
            print(f"✓ Paired as “{printable(reply['provider'], 60)}” ({printable(reply.get('visibility', 'public'), 10)} rig).\n",
                  flush=True)
            return reply
    raise SystemExit("The pairing code expired. Run the worker again to get a new one.")


# --- running in the background ---------------------------------------------------------------------------
SERVICE_LABEL = "chain.orbit.worker"


def _service_names() -> tuple[str, str, str]:
    """(systemd unit, launchd label, Windows Startup item) for the background service. ORBIT_SERVICE_NAME gives a
    second, separately named service (tests use it so they never touch a real install)."""
    name = os.environ.get("ORBIT_SERVICE_NAME", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", name) or name == "orbit-worker":
        return "orbit-worker", SERVICE_LABEL, "ORBIT Worker"
    return name, f"chain.orbit.{name}", f"ORBIT {name}"


def _service_args(script: Path, python: str) -> list[str]:
    """How the service starts the worker; everything else comes from the saved settings in --home."""
    return [python, str(script), "--home", str(home()), "--log-file", str(home() / "worker.log")]


def _systemd_quote(arg: str) -> str:
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _check(cmd: list[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"Could not {what}: {(res.stderr or res.stdout).strip() or f'exit code {res.returncode}'}")


def _windows_startup() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def install_service(script: Path, python: str) -> str:
    """Start the worker at login and keep it running, without administrator rights: a systemd user service on
    Linux, a launchd agent on macOS, and a Startup shortcut (running supervise()) on Windows."""
    args = _service_args(script, python)
    system = platform.system()
    unit_name, label, item = _service_names()
    if system == "Linux":
        if not shutil.which("systemctl"):
            raise SystemExit("systemd was not found; run the worker under your own supervisor (e.g. tmux or supervisord).")
        unit = Path.home() / ".config" / "systemd" / "user" / f"{unit_name}.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text("[Unit]\nDescription=ORBIT compute worker\nAfter=network-online.target\nWants=network-online.target\n\n"
                        f"[Service]\nExecStart={' '.join(_systemd_quote(a) for a in args)}\nRestart=always\nRestartSec=10\n"
                        "Environment=PYTHONUNBUFFERED=1\nNoNewPrivileges=yes\n\n[Install]\nWantedBy=default.target\n")
        _check(["systemctl", "--user", "daemon-reload"], "reload systemd (is a user session running?)")
        _check(["systemctl", "--user", "enable", "--now", unit_name], f"start the {unit_name} service")
        _run(["loginctl", "enable-linger", os.environ.get("USER", "")])  # keep running after logout, where allowed
        return f"systemd user service '{unit_name}' (logs: journalctl --user -u {unit_name} -f)"
    if system == "Darwin":
        import plistlib

        plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        logfile = str(home() / "worker.log")
        plist.write_bytes(plistlib.dumps({
            "Label": label, "ProgramArguments": args, "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 10, "StandardOutPath": logfile, "StandardErrorPath": logfile,
            "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"}}))
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        _check(["launchctl", "load", "-w", str(plist)], "load the launchd agent")
        return f"launchd agent '{label}' (logs: tail -f {logfile})"
    if system == "Windows":
        # Windows has no per-user service manager that restarts a crashed program, so the Startup shortcut runs
        # supervise(), which does.
        args = [*args, "--supervise"]
        pythonw = Path(python).with_name("pythonw.exe")
        exe = str(pythonw if pythonw.exists() else python)
        shortcut = _windows_startup() / f"{item}.lnk"
        shortcut.parent.mkdir(parents=True, exist_ok=True)
        quote = lambda v: v.replace("'", "''")  # noqa: E731  (PowerShell single-quoted string)
        arguments = " ".join(f'"{a}"' for a in args[1:])
        _check(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{quote(str(shortcut))}');"
                f"$s.TargetPath='{quote(exe)}';$s.Arguments='{quote(arguments)}';$s.WindowStyle=7;$s.Save()"],
               "create the Startup shortcut")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) \
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([exe, *args[1:]], creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Startup item '{item}' (runs at login; logs: {home() / 'worker.log'})"
    raise SystemExit("No supported service manager found; run the worker in a terminal or under your own supervisor.")


def supervise(child_args: list[str]) -> int:
    """Run the worker as a child process and restart it whenever it dies, with backoff. Stops when the worker exits
    cleanly (e.g. its token was revoked) or when --uninstall-service leaves a stop file in --home."""
    stop_file = home() / "stop"
    stop_file.unlink(missing_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    backoff = 5.0
    child: Optional[subprocess.Popen] = None

    def on_terminate(*_):  # stopping the supervisor stops its worker too
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, on_terminate)
    while True:
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *child_args], creationflags=flags)
        while child.poll() is None:
            if stop_file.exists():
                child.terminate()
                child.wait(30)
                stop_file.unlink(missing_ok=True)
                log.info("stopped by --uninstall-service")
                return 0
            time.sleep(1)
        if child.returncode == 0:
            log.info("worker exited cleanly; supervisor stopping")
            return 0
        if time.monotonic() - started > 60:
            backoff = 5.0
        log.warning("worker exited with code %s; restarting in %.0fs", child.returncode, backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


def uninstall_service() -> str:
    unit_name, label, item = _service_names()
    unit = Path.home() / ".config" / "systemd" / "user" / f"{unit_name}.service"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    shortcut = _windows_startup() / f"{item}.lnk"
    if unit.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", unit_name], capture_output=True)
        unit.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        return "Removed the systemd user service."
    if plist.exists():
        subprocess.run(["launchctl", "unload", "-w", str(plist)], capture_output=True)
        plist.unlink()
        return "Removed the launchd agent."
    if platform.system() == "Windows" and shortcut.exists():
        shortcut.unlink()
        home().mkdir(parents=True, exist_ok=True)
        (home() / "stop").write_text("")  # the running supervisor sees this, stops the worker and exits
        return "Removed the Startup item and stopped the worker."
    return "No ORBIT worker service is installed."


# --- the worker ----------------------------------------------------------------------------------------
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Worker:
    def __init__(self, orbit_url: str, token: str, upstream: str, upstream_key: str = "",
                 models: Optional[list[str]] = None, poll_wait: float = 20, heartbeat_every: float = 15,
                 capacity: int = 1, hardware: Optional[dict[str, str]] = None, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.orbit_url = orbit_url.rstrip("/")
        self.token = token
        self.upstream = upstream.rstrip("/")
        self.upstream_key = upstream_key
        self.models = models or []  # empty: serve every model the model server lists
        self.serving: set[str] = set(self.models)  # what was last advertised to ORBIT: the only models jobs may use
        self.poll_wait = poll_wait
        self.heartbeat_every = heartbeat_every
        self.capacity = capacity
        self.hardware = hardware
        self.max_tokens = max_tokens
        self.jobs_done = 0
        self.paused = False
        self._connected = False

    # --- talking to ORBIT ----------------------------------------------------------------------------
    def orbit(self, path: str, body: Optional[dict] = None, timeout: float = 30) -> Any:
        return request_json("POST", f"{self.orbit_url}{path}", body or {},
                            {"Authorization": f"Bearer {self.token}"}, timeout)

    def heartbeat(self) -> None:
        models = self.models or list_models(self.upstream, self.upstream_key, timeout=10)
        body: dict[str, Any] = {"models": models, "version": VERSION, "capacity": self.capacity}
        if self.hardware is not None and not self._connected:
            body["hardware"] = self.hardware  # once per start
        reply = self.orbit("/api/network/worker/heartbeat", body)
        self.serving = set(models)
        if not self._connected:
            log.info("connected as %s (%s rig), serving %s (capacity %d)", printable(reply.get("provider"), 60),
                     printable(reply.get("visibility", "public"), 10), ", ".join(models) or "nothing", self.capacity)
        if bool(reply.get("paused")) != self.paused:
            self.paused = bool(reply.get("paused"))
            log.info("rig %s from the ORBIT dashboard", "paused" if self.paused else "resumed")
        self._connected = True

    def heartbeat_loop(self, stop: threading.Event) -> None:
        """On its own thread, so a worker busy with a long generation still shows as online."""
        while not stop.is_set():
            try:
                self.heartbeat()
            except HttpError as exc:
                if exc.status == 401:
                    log.error("ORBIT rejected the worker token (rotated or retired). Pair again: "
                              "python3 orbit_worker.py --pair")
                    stop.set()
                    return
                log.warning("heartbeat failed: %s", exc)
            except (OSError, ValueError) as exc:
                log.warning("heartbeat failed (is your model server running at %s?): %s", self.upstream, exc)
            except Exception as exc:  # never let one bad reply stop the heartbeat: the rig would look offline
                log.warning("heartbeat failed unexpectedly: %s", printable(repr(exc)))
            stop.wait(self.heartbeat_every)

    # --- jobs ----------------------------------------------------------------------------------------
    def chat_request(self, job: dict[str, Any]) -> dict[str, Any]:
        """The request sent to your model server, built here rather than forwarded: a non-streaming chat completion,
        for a model you serve, with standard fields only and a bounded answer length."""
        request = job.get("request") if isinstance(job.get("request"), dict) else {}
        model = job.get("model")
        if not isinstance(model, str) or model not in self.serving:
            raise JobRefused(f"model {printable(model, 60)!r} is not one this rig serves")
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise JobRefused("the job has no messages")
        body = {k: request[k] for k in CHAT_FIELDS if k in request}
        try:
            asked = int(request.get("max_tokens") or self.max_tokens)
        except (TypeError, ValueError):
            asked = self.max_tokens
        body.update(model=model, stream=False, max_tokens=max(1, min(asked, self.max_tokens)))
        return body

    def run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("id", ""))
        if not JOB_ID.match(job_id):
            log.warning("ignored a job with an invalid id")
            return
        started = time.monotonic()
        headers = {"Authorization": f"Bearer {self.upstream_key}"} if self.upstream_key else {}
        try:
            timeout = min(max(float(job.get("timeout_seconds") or 170), JOB_TIMEOUT[0]), JOB_TIMEOUT[1])
        except (TypeError, ValueError):
            timeout = 170.0
        try:
            body = self.chat_request(job)
            reply = request_json("POST", f"{self.upstream}/chat/completions", body, headers, timeout=timeout)
            payload: dict[str, Any] = {"response": clean_reply(reply)}
        except Exception as exc:  # report a short reason (never local details) instead of letting the job time out
            payload = {"error": self._reason(exc)}
            log.warning("job %s failed: %s", job_id[:8], printable(exc))
        try:
            self.orbit(f"/api/network/worker/jobs/{job_id}/result", payload)
        except HttpError as exc:
            log.warning("result for job %s not accepted: %s", job_id[:8], exc)
            return
        self.jobs_done += 1
        log.info("job %s (%s) %s in %.1fs", job_id[:8], printable(job.get("model"), 60),
                 "failed" if "error" in payload else "done", time.monotonic() - started)

    @staticmethod
    def _reason(exc: Exception) -> str:
        if isinstance(exc, JobRefused):
            return f"refused: {exc}"
        if isinstance(exc, HttpError):
            return f"model server returned HTTP {exc.status}"
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return "model server timed out"
        if isinstance(exc, OSError):
            return "model server unreachable"
        if isinstance(exc, ValueError):
            return "model server returned an invalid reply"
        return "worker error"

    def run(self, stop: Optional[threading.Event] = None) -> None:
        """Serve one job at a time until stopped."""
        self.serve(stop or threading.Event(), concurrency=1)

    def serve(self, stop: threading.Event, concurrency: int = 1) -> None:
        self.capacity = max(1, concurrency)
        threading.Thread(target=self.heartbeat_loop, args=(stop,), daemon=True, name="heartbeat").start()
        loops = [threading.Thread(target=self.job_loop, args=(stop,), daemon=True) for _ in range(self.capacity - 1)]
        for t in loops:
            t.start()
        self.job_loop(stop)
        for t in loops:
            t.join()

    def job_loop(self, stop: threading.Event) -> None:
        while not self._connected and not stop.is_set():  # take jobs only once ORBIT knows what this rig serves
            stop.wait(0.5)
        backoff = 1.0
        while not stop.is_set():
            try:
                reply = self.orbit(f"/api/network/worker/poll?wait={self.poll_wait}", timeout=self.poll_wait + 15)
                job = reply.get("job") if isinstance(reply, dict) else None
                if isinstance(job, dict):
                    self.run_job(job)
                backoff = 1.0
            except HttpError as exc:
                if exc.status == 401:
                    log.error("ORBIT rejected the worker token. Pair again: python3 orbit_worker.py --pair")
                    stop.set()
                    return
                log.warning("ORBIT error: %s (retrying in %.0fs)", exc, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 60)
            except (OSError, ValueError) as exc:
                log.warning("connection problem: %s (retrying in %.0fs)", printable(exc), backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as exc:  # a job slot that dies would leave the rig serving fewer jobs, or none
                log.warning("unexpected error: %s (retrying in %.0fs)", printable(repr(exc)), backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 60)


NO_SERVER_HELP = """No local model server found (tried Ollama :11434, LM Studio :1234, vLLM :8000, llama.cpp :8080).

Quickest start, with Ollama (https://ollama.com/download):
    ollama pull qwen3:8b        # bigger models earn more per token, e.g. qwen3:32b or llama3.3:70b
then run this again. Or name your server: --upstream http://host:port/v1 (and --upstream-key if it needs one)
"""


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # legacy Windows consoles (cp1252) can't print ✓ or box drawing
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    env = os.environ.get
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--home")
    known, _ = pre.parse_known_args(argv)
    if known.home:  # where settings and logs live (services pass it explicitly)
        os.environ["ORBIT_HOME"] = known.home
    cfg = load_config()
    p = argparse.ArgumentParser(description="Serve ORBIT inference jobs from a local OpenAI-compatible model server.")
    p.add_argument("--orbit-url", default=env("ORBIT_URL") or cfg.get("orbit_url") or DEFAULT_ORBIT_URL,
                   help=f"the ORBIT server (default {DEFAULT_ORBIT_URL})")
    p.add_argument("--token", default=env("ORBIT_WORKER_TOKEN", ""),
                   help="worker token (default: the one saved by pairing in ~/.orbit/worker.json). Prefer pairing or "
                        "ORBIT_WORKER_TOKEN: other users of this machine can see command-line arguments")
    p.add_argument("--upstream", default=env("UPSTREAM_URL") or cfg.get("upstream") or "auto",
                   help="your model server's OpenAI-compatible URL, or 'auto' to find one on this machine")
    p.add_argument("--upstream-key", default=env("UPSTREAM_API_KEY", ""), help="API key for --upstream, if it needs one")
    p.add_argument("--models", default=env("WORKER_MODELS") or ",".join(cfg.get("models") or []),
                   help="comma-separated models to serve (default: every model your server lists)")
    p.add_argument("--concurrency", type=int, default=int(env("WORKER_CONCURRENCY") or cfg.get("concurrency") or 1))
    p.add_argument("--max-tokens", type=int, default=int(env("WORKER_MAX_TOKENS") or cfg.get("max_tokens") or DEFAULT_MAX_TOKENS),
                   help=f"longest answer to generate for one job (default {DEFAULT_MAX_TOKENS})")
    p.add_argument("--pair", action="store_true", help="pair this machine again (ignores a saved token)")
    p.add_argument("--private", action="store_true", help="suggest a private rig (serves only you, free) when pairing")
    p.add_argument("--setup", action="store_true", help="pair if needed, install the background service, and exit")
    p.add_argument("--install-service", action="store_true", help="install the background service and exit")
    p.add_argument("--uninstall-service", action="store_true", help="stop and remove the background service")
    p.add_argument("--no-browser", action="store_true", help="do not open the pairing link automatically")
    p.add_argument("--home", help="settings directory (default: ~/.orbit or ORBIT_HOME)")
    p.add_argument("--log-file", help="write logs to this file instead of the terminal")
    p.add_argument("--supervise", action="store_true", help="run the worker as a child and restart it if it dies")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        **({"filename": args.log_file} if args.log_file else {}))

    if args.uninstall_service:
        print(uninstall_service())
        return 0
    if args.supervise:
        return supervise([a for a in (argv if argv is not None else sys.argv[1:]) if a != "--supervise"])
    orbit_url = check_orbit_url(args.orbit_url)

    upstream, models = args.upstream, [m.strip() for m in args.models.split(",") if m.strip()]
    if upstream == "auto":
        if args.upstream_key:
            print("--upstream-key is only sent to a server you name: add --upstream http://host:port/v1", file=sys.stderr)
            return 2
        found = detect_upstream()
        if found is None:
            print(NO_SERVER_HELP, file=sys.stderr)
            return 1
        name, upstream, available = found
        print(f"✓ Found {name} at {upstream} serving {', '.join(printable(m, 60) for m in available)}", flush=True)

    token = "" if args.pair else (args.token or (cfg.get("token") if cfg.get("orbit_url", orbit_url) == orbit_url else ""))
    if not token:
        try:
            listed = models or list_models(upstream, args.upstream_key)
        except Exception:
            listed = models
        reply = pair(orbit_url, listed, upstream, "private" if args.private else None, not args.no_browser)
        token = reply["token"]
        cfg["dashboard_url"] = reply.get("dashboard_url")
    settings = {"orbit_url": orbit_url, "token": token, "upstream": upstream, "models": models,
                "concurrency": args.concurrency, "max_tokens": args.max_tokens}
    if any(cfg.get(k) != v for k, v in settings.items()):
        try:
            log.info("saved worker settings to %s", save_config({**cfg, **settings}))
        except OSError as exc:  # read-only home (e.g. a container): keep running with the settings in memory
            log.warning("could not save worker settings (%s); set ORBIT_HOME to a writable directory", exc)

    if args.setup or args.install_service:
        script = Path(__file__).resolve()
        target = home() / "orbit_worker.py"
        if script != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(script, target)
        try:
            where = install_service(target, sys.executable)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            if not args.setup:
                return 1
            print("Starting in the foreground instead (Ctrl+C to stop).", flush=True)
        else:
            dashboard = printable(load_config().get("dashboard_url") or "the ORBIT web app → Network", 200)
            print(f"✓ ORBIT worker installed as a {where}.\n  It starts at login and restarts if it stops. "
                  f"Manage the rig (pause, private/public, payouts) at {dashboard}.\n"
                  f"  To remove it: python3 {target} --uninstall-service", flush=True)
            return 0

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    pidfile = home() / "worker.pid"  # for status checks; nothing ever kills by it
    try:
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(str(os.getpid()))
    except OSError:
        pidfile = None
    worker = Worker(orbit_url, token, upstream, args.upstream_key, models, hardware=detect_hardware(),
                    max_tokens=max(1, args.max_tokens))
    thread = threading.Thread(target=worker.serve, args=(stop, max(1, args.concurrency)), daemon=True)
    thread.start()
    try:
        while thread.is_alive() and not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
    finally:
        if pidfile is not None:
            pidfile.unlink(missing_ok=True)
    if not stop.is_set():  # the worker stopped on its own: exit with an error, so the background service restarts it
        log.error("the worker stopped unexpectedly")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

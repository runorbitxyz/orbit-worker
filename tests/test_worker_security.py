"""The open-source worker, against a hostile ORBIT server: what reaches the operator's model server, where the worker
token can go, and what leaves the operator's machine. Uses a fake ORBIT and a fake model server on localhost."""
import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:  # in the ORBIT monorepo
    from worker import orbit_worker
except ImportError:  # in the published worker repository, next to orbit_worker.py
    import orbit_worker
HttpError, Worker, check_orbit_url = orbit_worker.HttpError, orbit_worker.Worker, orbit_worker.check_orbit_url
printable, request_json = orbit_worker.printable, orbit_worker.request_json


class Fake:
    """A localhost HTTP server that records requests and answers from a routes dict: path -> (status, body, headers)."""

    def __init__(self):
        self.requests, self.routes = [], {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                status, reply, headers = fake.routes.get(self.path.split("?")[0], (404, {}, {}))
                raw = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _serve

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self):
        return [r["path"].split("?")[0] for r in self.requests]


@pytest.fixture
def servers():
    orbit, model = Fake(), Fake()
    model.routes["/v1/chat/completions"] = (200, {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}, {})
    orbit.routes["/api/network/worker/jobs/job1/result"] = (200, {"ok": True}, {})
    yield orbit, model
    orbit.server.shutdown()
    model.server.shutdown()


def worker_for(orbit, model, **kw):
    return Worker(orbit.url, "orbw_secret", f"{model.url}/v1", upstream_key="upstream-secret",
                  models=["qwen3:8b"], **kw)


def job(**overrides):
    return {"id": "job1", "model": "qwen3:8b", "timeout_seconds": 30,
            "request": {"model": "qwen3:8b", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 64},
            **overrides}


def test_only_a_bounded_standard_chat_request_reaches_the_model_server(servers):
    orbit, model = servers
    hostile = job(request={"messages": [{"role": "user", "content": "x"}], "max_tokens": 10**9, "stream": True, "n": 50,
                           "options": {"num_ctx": 10**7}, "keep_alive": -1, "model": "private-finetune",
                           "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]})
    worker_for(orbit, model, max_tokens=512).run_job(hostile)
    sent = model.requests[0]["body"]
    assert sent == {"messages": [{"role": "user", "content": "x"}], "tools": hostile["request"]["tools"],
                    "model": "qwen3:8b", "stream": False, "max_tokens": 512}
    assert orbit.requests[-1]["body"] == {"response": {"choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": None}]}}


def test_jobs_for_models_the_rig_does_not_serve_never_reach_the_model_server(servers):
    orbit, model = servers
    worker_for(orbit, model).run_job(job(model="my-private-finetune"))
    assert model.requests == []
    assert orbit.requests[-1]["body"]["error"].startswith("refused: model 'my-private-finetune'")


def test_failures_are_reported_without_local_details(servers):
    orbit, model = servers
    model.server.shutdown()
    model.server.server_close()  # the model server is down
    worker_for(orbit, model).run_job(job())
    assert orbit.requests[-1]["body"] == {"error": "model server unreachable"}  # no URL, port or path


def test_bad_job_ids_are_ignored(servers):
    orbit, model = servers
    worker_for(orbit, model).run_job(job(id="../../admin/payouts"))
    assert model.requests == [] and orbit.requests == []


def test_redirects_are_refused_so_credentials_never_follow_them(servers):
    orbit, model = servers
    thief = Fake()
    orbit.routes["/api/network/worker/poll"] = (302, b"", {"Location": f"{thief.url}/steal"})
    with pytest.raises(HttpError) as exc:
        worker_for(orbit, model).orbit("/api/network/worker/poll?wait=0")
    assert exc.value.status == 302 and thief.requests == []
    thief.server.shutdown()


def test_oversized_replies_are_refused(servers, monkeypatch):
    orbit, model = servers
    monkeypatch.setattr(orbit_worker, "MAX_REPLY_BYTES", 1000)
    model.routes["/v1/chat/completions"] = (200, b"x" * 5000, {})
    worker_for(orbit, model).run_job(job())
    assert orbit.requests[-1]["body"] == {"error": "model server returned an invalid reply"}
    orbit.routes["/big"] = (200, b"[" + b"1," * 2000 + b"1]", {})
    with pytest.raises(ValueError):
        request_json("GET", f"{orbit.url}/big")


def test_the_upstream_key_goes_only_to_the_model_server_and_the_token_only_to_orbit(servers):
    orbit, model = servers
    worker_for(orbit, model).run_job(job())
    assert {r["auth"] for r in model.requests} == {"Bearer upstream-secret"}
    assert {r["auth"] for r in orbit.requests} == {"Bearer orbw_secret"}


def test_auto_detection_sends_no_api_key(monkeypatch):
    probe = Fake()
    probe.routes["/v1/models"] = (200, {"data": [{"id": "m"}]}, {})
    monkeypatch.setattr(orbit_worker, "KNOWN_UPSTREAMS", [("Probe", f"{probe.url}/v1")])
    assert orbit_worker.detect_upstream() == ("Probe", f"{probe.url}/v1", ["m"])
    assert probe.requests[0]["auth"] is None
    probe.server.shutdown()


@pytest.mark.parametrize("url", ["https://api.runorbit.xyz", "http://localhost:8000", "http://127.0.0.1:8000",
                                 "http://api:8000", "http://192.168.1.20:8000", "http://gpu-box.local:8000"])
def test_the_token_may_go_over_https_or_to_this_machine_or_network(url):
    assert check_orbit_url(url + "/") == url


@pytest.mark.parametrize("url", ["http://api.runorbit.xyz", "http://8.8.8.8", "ftp://api.runorbit.xyz", "api.runorbit.xyz"])
def test_the_token_never_goes_over_plain_http_to_the_internet(url):
    with pytest.raises(SystemExit):
        check_orbit_url(url)


def test_server_text_cannot_inject_terminal_escapes():
    assert printable("rig\x1b]0;pwned\x07\nname") == "rig ]0;pwned  name"
    assert len(printable("x" * 1000)) == 200


def test_settings_are_private_from_the_moment_they_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_HOME", str(tmp_path / "orbit"))
    os.umask(0o022)
    path = orbit_worker.save_config({"token": "orbw_secret"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not (path.parent / "worker.tmp").exists()


def test_the_worker_waits_to_be_connected_before_taking_jobs(servers):
    orbit, model = servers
    stop = threading.Event()
    w = worker_for(orbit, model, poll_wait=0)
    t = threading.Thread(target=w.job_loop, args=(stop,), daemon=True)
    t.start()
    t.join(1.5)
    assert "/api/network/worker/poll" not in orbit.paths()  # no heartbeat yet: ORBIT doesn't know what it serves
    stop.set()
    t.join(5)


def test_only_the_standard_parts_of_an_answer_go_back_to_orbit(servers):
    orbit, model = servers
    model.routes["/v1/chat/completions"] = (200, {
        "id": "c1", "model": "qwen3:8b", "system_fingerprint": "fp_host_details", "x_server": {"path": "/home/me/models"},
        "choices": [{"index": 0, "finish_reason": "stop", "logprobs": {"secret": 1},
                     "message": {"role": "assistant", "content": "hi", "reasoning_content": "because", "extra": "x"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "timings": {"prompt_ms": 1.5}, "server": "llama",
                  "completion_tokens_details": {"reasoning_tokens": 1, "note": "x"}}}, {})
    worker_for(orbit, model).run_job(job())
    assert orbit.requests[-1]["body"]["response"] == {
        "id": "c1", "model": "qwen3:8b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi", "reasoning_content": "because"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 1}}}


def test_a_reply_that_is_not_a_chat_completion_is_reported_as_invalid(servers):
    orbit, model = servers
    model.routes["/v1/chat/completions"] = (200, {"error": "loading model", "path": "/home/me"}, {})
    worker_for(orbit, model).run_job(job())
    assert orbit.requests[-1]["body"] == {"error": "model server returned an invalid reply"}


def test_pairing_sends_neither_the_machine_name_nor_the_model_server_address(monkeypatch):
    orbit = Fake()
    orbit.routes["/api/network/pair/start"] = (200, {"device_code": "d", "user_code": "ABCD-2345", "interval": 1,
                                                     "expires_in": 5, "verification_url": f"{orbit.url}/network"}, {})
    orbit.routes["/api/network/pair/poll"] = (200, {"status": "approved", "token": "orbw_t", "provider": "rig"}, {})
    monkeypatch.setattr(orbit_worker.socket, "gethostname", lambda: "janes-macbook-pro")
    orbit_worker.pair(orbit.url, ["qwen3:8b"], "http://user:pass@10.0.0.5:9000/v1", open_browser=False)
    sent = orbit.requests[0]["body"]
    assert "hostname" not in sent and sent["upstream"] == "custom server"
    assert "janes" not in json.dumps(sent) and "10.0.0.5" not in json.dumps(sent) and "pass" not in json.dumps(sent)
    assert set(sent) == {"models", "hardware", "version", "upstream", "platform", "visibility"}
    assert orbit_worker.server_kind("http://localhost:11434/v1/") == "Ollama"
    orbit.server.shutdown()


def test_the_default_server_is_the_public_network_over_https():
    assert orbit_worker.DEFAULT_ORBIT_URL == "https://api.runorbit.xyz"
    assert check_orbit_url(orbit_worker.DEFAULT_ORBIT_URL) == orbit_worker.DEFAULT_ORBIT_URL


def test_a_reply_nested_too_deeply_is_invalid_not_fatal(servers):
    orbit, model = servers
    model.routes["/v1/chat/completions"] = (200, b"[" * 200_000 + b"]" * 200_000, {})
    worker_for(orbit, model).run_job(job())
    assert orbit.requests[-1]["body"] == {"error": "model server returned an invalid reply"}


def test_job_slots_survive_unexpected_errors(servers):
    orbit, model = servers
    orbit.routes["/api/network/worker/poll"] = (200, b"[" * 200_000 + b"]" * 200_000, {})
    stop = threading.Event()
    w = worker_for(orbit, model, poll_wait=0)
    w._connected = True
    w.run_job = lambda _job: (_ for _ in ()).throw(RuntimeError("boom"))
    t = threading.Thread(target=w.job_loop, args=(stop,), daemon=True)
    t.start()
    t.join(1.5)
    assert t.is_alive()  # a bad reply (or a bug) is retried with backoff, never the end of the job slot
    orbit.routes["/api/network/worker/poll"] = (200, {"job": job()}, {})
    t.join(2.5)
    assert t.is_alive()
    stop.set()
    t.join(5)

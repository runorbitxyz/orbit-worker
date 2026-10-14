# ORBIT worker

Serve AI inference from your own GPU to the [ORBIT](https://runorbit.xyz) network and earn $ORBIT, or keep it private
and run your own agent on it for free.

The worker is **one Python file** (`orbit_worker.py`, standard library only, Python 3.9+). It connects a model server
you already run (Ollama, LM Studio, vLLM, llama.cpp, or anything OpenAI-compatible) to ORBIT: it asks ORBIT for jobs,
runs each one on your model server, and sends the answer back. Read it before you run it; it is meant to be read.

## Quick start

Install a model server and pull a model, for example with [Ollama](https://ollama.com/download):

```sh
ollama pull qwen3:8b
```

Then, on the same machine:

```sh
# macOS and Linux
curl -fsSL https://api.runorbit.xyz/install.sh | sh

# Windows (PowerShell)
irm https://api.runorbit.xyz/install.ps1 | iex
```

The installer downloads `orbit_worker.py` to `~/.orbit`, finds your model server, and shows a short pairing code. It
opens [Run a GPU](https://runorbit.xyz/network) in your browser: check that the code matches, choose **public** (earns
$ORBIT) or **private** (serves only you, free), and approve. The worker then runs in the background, starts when you
log in, and restarts if it stops.

To remove it: `python3 ~/.orbit/orbit_worker.py --uninstall-service`.

Prefer not to pipe a script into your shell? Download `orbit_worker.py` from a [release](../../releases), check it
(see below), and run it yourself:

```sh
python3 orbit_worker.py --setup                     # pair, install the background service, exit
python3 orbit_worker.py                             # or run it in this terminal instead
python3 orbit_worker.py --private --setup           # a rig that serves only you
python3 orbit_worker.py --upstream http://127.0.0.1:8000/v1 --models qwen3:32b   # name the server and models
```

`python3 orbit_worker.py --help` lists every option. Options can also come from the environment: `ORBIT_URL`,
`ORBIT_WORKER_TOKEN`, `UPSTREAM_URL`, `UPSTREAM_API_KEY`, `WORKER_MODELS`, `WORKER_CONCURRENCY`, `WORKER_MAX_TOKENS`,
`ORBIT_HOME`. There is also a `Dockerfile` for running it in a container next to your model server.

## What it does, and never does

It **does**:

- Connect only outward: to the ORBIT server (over HTTPS) and to your model server. It opens no ports, so it works behind
  NAT with no port forwarding.
- Send your model server one kind of request: a non-streaming chat completion for a model you serve. It builds that
  request itself from a fixed list of standard fields and caps the answer length; anything else a job asks for is
  dropped.
- Send ORBIT the standard parts of each answer (the message, why it stopped, token counts), the models you serve, the
  kind of model server (for example "Ollama", never its address), and the hardware description you see when pairing
  (OS, GPU, memory, CPU).
- Keep its settings in `~/.orbit/worker.json` (readable only by you; it holds your worker token) and, when it runs in
  the background, its log in `~/.orbit/worker.log`.

It **never**:

- Runs commands, code or tools sent to it. Tool calls are text it passes back.
- Reads, uploads or changes your files. It reads only its own settings and, on Linux, `/proc/meminfo` and
  `/proc/cpuinfo` to describe your hardware.
- Logs prompts or answers.
- Sends your machine's name, your model server's address, or anything your model server adds to its answers (timings,
  server details).
- Updates itself. The code you read is the code that runs.
- Sends your worker token anywhere but the ORBIT server. It refuses plain-HTTP ORBIT addresses (except this machine
  or your local network) and does not follow redirects.
- Sends your model server's API key anywhere but the model server you name. Auto-detection sends no key at all.

The only programs it starts are `nvidia-smi` and `sysctl`, to describe your hardware, and, only when you ask it to run
in the background, your OS's service tools: `systemctl` and `loginctl` on Linux, `launchctl` on macOS, PowerShell
(to create a Startup shortcut) on Windows. On Windows that shortcut runs a copy of the worker as a supervisor that
restarts it if it dies.

## Earning, and keeping the network honest

- **Public rigs** serve public missions and anyone who opts in, and earn their share of each accepted call (90% by
  default). Bigger models earn more per token. Earnings are paid in $ORBIT to the wallet you link, in regular
  batches.
- **Bonds.** Rigs serving bigger models (medium tier and up) stake $ORBIT from their payout wallet to earn from paid
  traffic.
- **Checks.** Rigs get unannounced test questions with known answers, harder for bigger models. Wrong answers lower a
  rig's reputation, and with it how much work it gets, and cost part of its bond.
- **Private rigs** serve only you, for free. Nobody else's requests ever reach them.

See the [ORBIT docs](https://runorbit.xyz/docs#gpu) for how payouts, bonds and pricing work.

## Checking what you run

Every release attaches `orbit_worker.py`, the installers and a `SHA256SUMS` file, with build provenance signed by
GitHub Actions for this repository. To check a downloaded worker:

```sh
sha256sum orbit_worker.py            # compare with SHA256SUMS in the release (macOS: shasum -a 256)
gh attestation verify orbit_worker.py --repo runorbitxyz/orbit-worker
```

The ORBIT server serves the same file the installers download (`https://api.runorbit.xyz/worker/orbit_worker.py`);
its hash should match the latest release's.

## Development

```sh
python -m pytest -q tests/test_worker_security.py   # the worker against a hostile fake ORBIT and model server
python tests/service_e2e.py --home /tmp/orbit-svc    # installs, restarts and removes the background service here
```

CI runs both on Linux, macOS and Windows, and checks the worker still runs on Python 3.9. This repository is published
from the ORBIT monorepo; changes are welcome as issues and pull requests here.

## Security

Please report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)

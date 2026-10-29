# Yunex Demo — reliable LLM pipelines, running locally

Support material for a hackathon talk on **reliably using LLMs to extract data,
classify it, and act on it**. Everything runs on your own machine: no API keys,
no data leaving the laptop.

| Piece | What it does |
|---|---|
| [Ollama](https://ollama.com) | serves `gemma4` for generation and `embeddinggemma-2` for embeddings |
| [ChromaDB](https://trychroma.com) | vector store for the retrieval demo |
| [Streamlit](https://streamlit.io) | the UI, one page per demo beat |

> **Status: infrastructure only.** The stack comes up and the **Stack Health**
> page verifies it end to end. The three demo pages are stubs — the extraction,
> classification, and RAG logic is planned separately.

---

## Quick start

Pick the command matching your hardware. All three serve the same app and
differ only in *where inference runs*.

| Your machine | Command | One-time setup needed |
|---|---|---|
| **NVIDIA GPU** — Linux or Windows/WSL2 | `make up-gpu` | Linux only: [install the toolkit](#linux--nvidia) |
| **Apple Silicon Mac** | `make up-external` | [`brew install ollama`](#macos) |
| **No GPU**, or you just want it running | `make up` | none |

**If you have an NVIDIA card, use `make up-gpu`.** GPU inference is far faster
than CPU, and that gap compounds: a workflow makes several LLM calls back to
back, so seconds of per-call latency become tens of seconds of dead air on
stage. Prefer it whenever it's available to you.

> **`make up-gpu` is a request, not a guarantee.** Without
> `nvidia-container-toolkit` on the host it detects the problem, warns, and
> falls back to CPU — you get a working stack with no speedup. **Confirm it
> actually engaged** via the *Acceleration* row on Stack Health, which reports
> what Ollama really loaded onto the device rather than what you asked for.
> See [Linux + NVIDIA](#linux--nvidia) for the one-time install.

Then open <http://localhost:8501> and run **Stack Health** first.

First run downloads **~4.6 GB** of model weights, so give it time on slow wifi.
Weights persist in a Docker volume, so later starts are fast.

Plain Compose works too (`docker compose up -d`), and is the fallback where
`make` isn't available, such as native Windows. The Makefile is worth
preferring for two things Compose can't express: the GPU fallback above, and
running the model pull in the **foreground**, so you see download progress and
"Ready" means the models are genuinely present. Everything else in it is
convenience — `make help` lists the lot.

---

## Three ways to run

The app only ever reads `OLLAMA_HOST`. The profiles differ in *where Ollama
lives*, and nothing else — same image, same code.

| Command | Ollama runs | GPU | Use when |
|---|---|---|---|
| `make up-gpu` | container | **yes** (NVIDIA) | **preferred** — Linux with the toolkit, or Windows/WSL2 |
| `make up-external` | on the host | **yes** | **macOS**, or you already run Ollama natively |
| `make up` (or `docker compose up -d`) | container | no | no GPU available; zero host setup |

### Linux + NVIDIA

Docker cannot see the GPU until the container toolkit is installed. This is a
one-time host change:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it took:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Then `make up-gpu`. **If the toolkit is missing, `make up-gpu` still starts but
silently falls back to CPU** — the *Acceleration* row on Stack Health reports
what Ollama actually loaded onto the device, so check there rather than trusting
the command you ran.

### macOS

Docker on macOS has **no route to the Apple Silicon GPU**, so a containerised
Ollama is permanently CPU-only there. Run Ollama natively instead:

```bash
brew install ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve   # in its own terminal -- see note below
ollama pull gemma4:e2b-it-qat
ollama pull embeddinggemma-2:270m
make up-external                         # Chroma + Streamlit in Docker, Ollama on the host
```

`OLLAMA_HOST=0.0.0.0:11434` matters. Ollama binds to `127.0.0.1` by default,
and `host.docker.internal` reaches the host on a gateway address rather than
loopback — so a default-bound Ollama refuses the container's connection. The
same applies to a native Ollama on Windows. (The `external` profile was
verified on Linux, where the container Ollama already published on `0.0.0.0`;
the loopback case is the one to watch for on Docker Desktop.)

### Windows

**Use the WSL2 backend** (Docker Desktop's default). Inside a WSL2 shell
everything works as on Linux, `make` included, and `make up-gpu` gets the GPU
from the host NVIDIA driver alone — no in-container toolkit install needed.

From **native PowerShell or cmd** the stack still runs, but the Makefile does
not: it needs GNU `make` plus `grep`/`awk`, and `docker/preflight-gpu.sh` needs
a POSIX shell. Use Compose directly:

```powershell
docker compose up -d                                        # CPU
docker compose -f compose.yaml -f compose.gpu.yaml up -d     # NVIDIA
```

The GPU variant skips the preflight check, so if `nvidia-container-toolkit`
isn't available you'll get Docker's raw
`could not select device driver "nvidia"` error instead of an automatic CPU
fallback.

`.gitattributes` forces LF checkout on Windows. Don't remove it: `docker/*.sh`
is bind-mounted into Linux containers and run by `/bin/sh`, and with CRLF
endings `set -eu` becomes `set -eu<CR>`, so provisioning dies with
`set: Illegal option -` and no models get pulled.

---

## Hardware

Defaults target a **16 GB laptop**:

| | Size | Notes |
|---|---|---|
| `gemma4:e2b-it-qat` | 4.3 GB | "effective 2B", built for edge devices, native function calling |
| `embeddinggemma-2:270m` | 304 MB | 768-dim, 8K context |

Working set is roughly 6.5 GB with Chroma and Streamlit included.

**On an 8 GB machine**, drop to the 1B model in `.env`:

```bash
MODEL_MAIN=gemma3:1b-it-qat
```

### CPU is a usable fallback — and that is part of the point

Use a GPU if you have one. But CPU stays viable, which is what makes the demo
portable to any attendee's laptop: schema-constrained extraction emits only
~100–250 output tokens, so the slow path is slow-but-tolerable rather than
unusable. Measured with `make health` on this stack:

| Hardware | Throughput | ~200-token extraction |
|---|---|---|
| i7-13700K, 24 threads, CPU only | **42 tok/s** | ~5 s |
| typical 8-core laptop CPU (estimated) | ~12–15 tok/s | ~15 s |

These are CPU-only figures, and they are the *floor* — a GPU run is
substantially quicker. Run `make health` on your own machine for a real number
rather than trusting any estimate here; the *Generation speed* row gives you
tok/s and *Acceleration* tells you which path you actually got.

The infrastructure leans into the slow case so CPU stays presentable:

- `OLLAMA_KEEP_ALIVE=-1` — weights never unload mid-presentation.
- `make warm` — preloads both models so the first on-stage call isn't cold.
- `OLLAMA_NUM_CTX=8192` — the model's 128K window would blow an 8 GB card's KV
  cache and waste RAM for no demo benefit.

---

## Configuration

Everything is env-driven through a single `Settings` object
(`src/yunex_demo/config.py`). The stack boots with **no `.env` at all** — the
committed defaults match `.env.example`. Copy it to `.env` to override:

```bash
cp .env.example .env
```

Published host ports are configurable, since `8000` in particular is heavily
contested on dev machines:

| Variable | Default | Service |
|---|---|---|
| `APP_PUBLISHED_PORT` | `8501` | Streamlit |
| `CHROMA_PUBLISHED_PORT` | `8100` | ChromaDB (container-internal stays `8000`) |
| `OLLAMA_PUBLISHED_PORT` | `11434` | Ollama |

---

## Commands

```
make up-gpu        Start with NVIDIA acceleration (preferred, if you have a card)
make up-external   Start against a host Ollama (macOS)
make up            Start on CPU (works anywhere, no host setup)
make pull          Provision models ahead of time — do this before the event
make warm          Load models into memory before presenting
make health        Run the full stack check headlessly; non-zero exit on failure
make logs          Tail all service logs
make ps            Service status
make down          Stop, keeping models and vector data
make clean         Stop and delete volumes (re-downloads models)
make lint          Format and lint
make test          Unit tests (no running stack needed; live checks self-skip)
make test-live     Tests against the running stack, via its published ports
```

`make health` runs the same checks as the Stack Health page, so you can verify
the stack backstage without opening a browser:

```
[PASS] Ollama                        v0.40.13, 2 model(s) installed      (0.04s)
[PASS] ChromaDB                      reachable (API v1.0.0), 0 collection(s)
[PASS] Acceleration                  CPU -- no layers offloaded
[PASS] Generation speed              41.7 tok/s (64 tokens in 1.53s)
[PASS] Embed/store/query round trip  2 docs stored, 768-dim vectors
```

The round-trip check is the important one: it embeds through Ollama, writes to
Chroma, queries it back, and asserts the nearest neighbour is correct. If it
passes, the whole chain is genuinely wired up.

---

## Verifying the setup

### The stack

```bash
make health      # all five checks; exits non-zero on failure
make test-live   # 14 tests against the running stack
```

Or open <http://localhost:8501> → **Stack Health** → *Run checks*. Watch the
**Embed/store/query round trip** row: it's the only check that proves Ollama,
the embedder, and Chroma work *together* rather than just being individually
reachable.

To prove it works the way a fresh clone would experience it:

```bash
make clean && make up    # deletes volumes, re-downloads ~4.6GB, ~2 min
make health
```

### The `.env` deny rule

This one needs a **fresh Claude Code session** — permission config loads at
session start, so the session that wrote the rule cannot verify it.

```bash
printf 'DENY_RULE_CANARY=this-is-not-a-real-secret\n' > .env
```

Then start a new `claude` session in this directory and try all three:

| Ask | Expected |
|---|---|
| "read .env" | **blocked** by permissions |
| "grep for DENY_RULE_CANARY" | **blocked** |
| "read .env.example" | **works**, contents returned |

Pass means the canary value never appears in a reply. `/permissions` lists the
loaded rules if you want a quicker sanity check. Delete the `.env` afterwards.

Two things that look like failures but aren't:

- **`cat .env` via Bash is not blocked, by design.** It's the most natural way
  to try testing this and it will succeed. A path-glob rule on `Read`/`Grep`
  can't reliably separate `cat .env` from safe shell commands, so only those two
  tools are covered.
- **The `.env` must be inside the project tree.** The glob resolves relative to
  the working directory, so one in `/tmp` won't match even with a correct rule —
  indistinguishable from the rule not working.

---

## Troubleshooting

Symptoms seen while building this, and what each actually means.

| Symptom | Cause | Fix |
|---|---|---|
| Stack Health: `missing: gemma4:...` | `model-init` hasn't finished or failed | `make pull`. Briefly expected after a bare `docker compose up`, which pulls concurrently with the app — `make up` blocks until it's done |
| `model-init: set: Illegal option -` | CRLF line endings on `docker/*.sh` | `.gitattributes` was removed or git forces CRLF. `git config core.autocrlf false`, then re-checkout |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | another service owns that host port | set `CHROMA_PUBLISHED_PORT` (or `APP_`/`OLLAMA_`) in `.env` |
| `could not select device driver "nvidia"` | `nvidia-container-toolkit` missing | `make up-gpu` falls back to CPU automatically; raw `docker compose -f ... gpu.yaml` does not |
| Acceleration row says **CPU** but you ran `up-gpu` | the toolkit is missing and it fell back | expected — the row reports what Ollama really loaded, not the profile you asked for. See [Linux + NVIDIA](#linux--nvidia) |
| `external` profile: `Connection refused` | host Ollama bound to `127.0.0.1` | start it as `OLLAMA_HOST=0.0.0.0:11434 ollama serve` |
| `time: missing unit in duration "-1"` | `OLLAMA_KEEP_ALIVE` set to a non-numeric string | use seconds (`-1`, `300`) or a Go duration (`5m`) |
| Generation speed check fails, "too few tokens" | the model stopped early | a measurement guard, not a stack failure — a figure from <10 tokens isn't meaningful |
| Everything works but feels slow | CPU inference | `make warm` before presenting; on 8GB set `MODEL_MAIN=gemma3:1b-it-qat`. See [CPU is slow but workable](#cpu-is-a-usable-fallback--and-that-is-part-of-the-point) |
| Page edits don't show up | edited outside the bind mount | only `./src` is mounted; anything else needs a rebuild |

---

## Layout

```
compose.yaml                 base stack (CPU default)
compose.gpu.yaml             NVIDIA device reservation
compose.external.yaml        host Ollama instead of the container
docker/app.Dockerfile        uv-based Streamlit image
docker/pull-models.sh        idempotent model provisioning
src/yunex_demo/config.py     single source of truth for settings
src/yunex_demo/health.py     stack checks (page + CLI share these)
src/yunex_demo/warm.py       model preloading
src/yunex_demo/ui/           Streamlit pages
```

---

## Development setup

```bash
uv sync --all-groups
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
make test
```

`./src` is bind-mounted into the app container, so editing a page and
refreshing the browser is enough — no rebuild.

### Hooks

Two stages, from `.pre-commit-config.yaml`:

| Stage | Runs |
|---|---|
| **pre-commit** | whitespace/EOF/YAML/JSON fixers, `ruff check --fix`, `ruff format`, `detect-secrets`, `pytest` |
| **pre-push** | `pytest` again, as a last gate before code leaves the machine |

`ruff` stands in for black + flake8 + isort + autoflake + pycln: its `E`/`F`
rules cover flake8 at the same 100-character limit, `I` covers isort, `F401`
covers the unused-import strippers, and `ruff format` covers black. There is
deliberately **no `.flake8` file** — ruff's config in `pyproject.toml` would
silently shadow it, leaving a config that looks live but isn't.

Rules live only in `pyproject.toml` (`[tool.ruff]`), and the hooks carry no
inline rule args, so there is one source of truth.

The test suite is ~0.7 s offline, so both stages run the whole thing rather
than using a `pytest-testmon` fast path — scoping it would add a dependency and
a cache file to save nothing. Worth revisiting if the suite grows.

### Secret scanning

`detect-secrets` runs on every commit (not just on push — a leaked credential
gets worse the longer it is out there). It compares against
`.secrets.baseline`, which is committed and contains hashes, never the secrets
themselves. The current baseline has **no findings**.

If the hook flags something new:

```bash
uv run detect-secrets scan --baseline .secrets.baseline   # refresh
uv run detect-secrets audit .secrets.baseline             # triage each finding
```

The audit step is interactive and needs human judgment about what is actually
sensitive — don't regenerate the baseline just to silence the hook, since an
unaudited baseline means the scanner is quietly passing over real secrets.

### Tests

```bash
make test        # no stack needed; the live-stack checks skip themselves
make test-live   # against a running stack, via its published ports
```

`tests/test_pages.py` executes every Streamlit page through `AppTest`. HTTP 200
from the server proves nothing — pages run per session over a websocket, so a
bad `st.*` call only surfaces when the script actually renders.

### CI

`.github/workflows/ci.yml` runs on pushes to `main` and on every pull request.
It invokes the *same* `.pre-commit-config.yaml` via `pre-commit/action` rather
than re-declaring lint rules, so CI and local hooks cannot disagree about what
passing means — local hooks only protect contributors who ran
`pre-commit install` and never passed `--no-verify`.

CI starts no service containers: the live-stack checks self-skip, so it
verifies lint, config, and page rendering without pulling 4.6 GB of models.

### `.env` protection

`.claude/settings.json` is committed and denies Claude Code's `Read` and `Grep`
tools on real `.env` files while leaving `.env.example` readable. `Grep` matters
as much as `Read` because it returns matched lines, so a search for a variable
name could otherwise print its value.

Scope, honestly: it blocks `Read`/`Grep` calls that name a `.env` path or glob.
It does **not** block shell commands like `cat .env` — a path-glob permission
rule can't reliably tell those from safe ones. It covers everyone who opens this
repo with Claude Code, and complements (not replaces) a personal global rule in
`~/.claude/settings.json`.

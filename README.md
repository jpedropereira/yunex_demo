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

```bash
make up          # CPU, everything in Docker. Works anywhere.
```

Then open <http://localhost:8501> and run **Stack Health** first.

First run downloads **~4.6 GB** of model weights, so give it time on slow wifi.
Weights persist in a Docker volume, so later starts are fast.

---

## Three ways to run

The app only ever reads `OLLAMA_HOST`. The profiles differ in *where Ollama
lives*, and nothing else — same image, same code.

| Command | Ollama runs | GPU | Use when |
|---|---|---|---|
| `make up` | container | no | anywhere; zero host setup |
| `make up-gpu` | container | yes (NVIDIA) | Linux with the toolkit installed, or Windows/WSL2 |
| `make up-external` | on the host | yes | **macOS**, or you already run Ollama natively |

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
ollama serve                       # in its own terminal
ollama pull gemma4:e2b-it-qat
ollama pull embeddinggemma-2:270m
make up-external                   # Chroma + Streamlit in Docker, Ollama on the host
```

### Windows / WSL2

`make up-gpu` works on the host NVIDIA driver alone — no in-container toolkit
needed.

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

### CPU is slow but workable — and that is part of the point

Schema-constrained extraction emits only ~100–250 output tokens, which keeps
CPU inference tolerable. Measured with `make health` on this stack:

| Hardware | Throughput | ~200-token extraction |
|---|---|---|
| i7-13700K, 24 threads, CPU only | **42 tok/s** | ~5 s |
| typical 8-core laptop CPU (estimated) | ~12–15 tok/s | ~15 s |

Run `make health` on your own machine for a real number rather than trusting
the estimate. The infrastructure leans into the slow case:

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
make up            Start on CPU (default)
make up-gpu        Start with NVIDIA acceleration
make up-external   Start against a host Ollama (macOS)
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

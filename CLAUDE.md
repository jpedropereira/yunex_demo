# CLAUDE.md

## What this repo is

Demo code supporting a **hackathon presentation** on reliably using LLMs to
extract data, classify it, and act on it, all running locally (Ollama +
ChromaDB + Streamlit). The audience reads this code during and after the talk,
so the repo is **teaching material first** and software second.

It is **not production-ready, and isn't meant to be.** Don't add
production features (auth, multi-tenancy, retries-with-backoff frameworks,
observability stacks, plugin systems) unless asked.

`README.md` covers setup, commands, and troubleshooting. Don't repeat it here.

## Stack

Everything runs locally: no API keys, no hosted LLM providers. Don't introduce
a cloud LLM API (OpenAI, Anthropic, etc.) unless asked.

**LLMs** (served by Ollama, names set in `config.py` / `.env`):

| Role | Model | Notes |
|---|---|---|
| Generation (`MODEL_MAIN`) | `gemma4:e2b-it-qat` | effective-2B edge model, ~4.3 GB, native function calling |
| Embeddings (`MODEL_EMBED`) | `embeddinggemma-2:270m` | 768-dim vectors, 8K context |
| Fallback for 8 GB machines | `gemma3:1b-it-qat` | swap in via `MODEL_MAIN` |

These are small models running on laptops, often on CPU. Keep prompts short,
outputs small (schema-constrained), and context under `OLLAMA_NUM_CTX` (8192).

**Services and libraries:**

- **Ollama** `0.40.13` — model server; Python client `ollama`, or `httpx` for raw API calls
- **ChromaDB** `1.5.9` — vector store for the RAG demo
- **Streamlit** `1.66.1` — UI, one page per demo beat in `src/yunex_demo/ui/pages/`
- **Pydantic** + **pydantic-settings** — schemas for structured output, and config
- **Python 3.12**, managed with **uv**; everything runs in Docker Compose
- **Tooling:** ruff, pytest, pre-commit, detect-secrets

## Priorities, in order

1. **Safety and correctness.** Best practices still apply, especially around
   security. Never commit secrets, validate untrusted input (LLM output is
   untrusted input too), don't build shell commands or queries by string
   concatenation, keep the `.env` deny rule in `.claude/settings.json` intact.

   **Exception: deliberate counter-examples.** Some demo beats may need code
   that skips validation, to show what goes wrong without it. Write that only
   when explicitly asked, never on your own initiative. When you do, label it
   clearly in a comment and the page text as an intentional bad example, and
   keep it isolated so it can't be mistaken for (or reused as) the safe path.
2. **Readability.** Someone at the hackathon should understand a file in one
   read, top to bottom.
3. **Simplicity.** Choose the most direct solution that works. When in doubt,
   leave it out.

## How to write code here

- **Pedagogic over clever.** Prefer explicit, linear code a newcomer can follow
  over abstractions, metaprogramming, or one-liners that need decoding.
- **No over-engineering.** No base classes, factories, registries, or config
  knobs for a single use case. Three similar lines beat a premature helper.
  Add an abstraction only when it makes the code *easier* to read.
- **Name things for the reader.** Descriptive names, short functions, type
  hints on public functions.
- **Comments explain *why*.** Each module opens with a docstring saying what it
  demonstrates. Comment the non-obvious decision (as in `config.py` and
  `health.py`), not what the line already says.
- **Keep the demo point visible.** Each UI page is one beat of the talk; the
  technique it shows (schema-constrained output, validation, grounding, RAG)
  should be easy to spot, not buried under plumbing.
- **Deterministic by default.** Generation uses `temperature=0` and a fixed
  seed: a reliability talk can't have a demo that answers differently each run.
- **Few dependencies.** Reach for the standard library or what's already in
  `pyproject.toml` before adding a package.

## Project conventions

- Configuration goes through `settings` in `src/yunex_demo/config.py`; never
  read `os.environ` directly.
- Keep logic out of Streamlit pages where practical: plain functions in
  `src/yunex_demo/`, called by the pages (the pattern `health.py` follows).
- Lint/format with **ruff** only (config in `pyproject.toml`). Don't add
  black, flake8, or isort.
- Run `make lint` and `make test` before declaring work done. Every page must
  still render under `tests/test_pages.py`.
- Add tests for logic that matters to the demo, and keep them fast and
  readable. No heavy mocking frameworks.

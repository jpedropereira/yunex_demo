# Contributing

How to work on this repo. To run the demo, see the [README](README.md).

## Setup

```bash
uv sync --all-groups
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
make test
```

`./src` is bind-mounted into the app container, so editing a page and refreshing the
browser is enough, with no rebuild.

`.gitattributes` forces LF checkout on Windows. Don't remove it: `docker/*.sh` is
bind-mounted into Linux containers and run by `/bin/sh`, and with CRLF endings
`set -eu` becomes `set -eu<CR>`, so provisioning dies with `set: Illegal option -` and
no models get pulled.

## Hooks

Two stages, from `.pre-commit-config.yaml`:

| Stage | Runs |
|---|---|
| **pre-commit** | whitespace/EOF/YAML/JSON fixers, `ruff check --fix`, `ruff format`, `detect-secrets`, `pytest` |
| **pre-push** | `pytest` again, as a last gate before code leaves the machine |

`ruff` stands in for black, flake8, isort, autoflake and pycln: its `E`/`F` rules cover
flake8 at the same 100-character limit, `I` covers isort, `F401` covers the unused-import
strippers, and `ruff format` covers black. There is deliberately **no `.flake8` file**:
ruff's config in `pyproject.toml` would silently shadow it, leaving a config that looks
live but isn't.

Rules live only in `pyproject.toml` (`[tool.ruff]`), and the hooks carry no inline rule
args, so there is one source of truth.

The offline test suite takes about 10 s, so both stages run all of it rather than using a
`pytest-testmon` fast path, which would add a dependency and a cache file.

## Secret scanning

`detect-secrets` runs on every commit, not just on push, because a leaked credential gets
worse the longer it is out there. It compares against `.secrets.baseline`, which is
committed and contains hashes, never the secrets themselves.

If the hook flags something new:

```bash
uv run detect-secrets scan --baseline .secrets.baseline   # refresh
uv run detect-secrets audit .secrets.baseline             # triage each finding
```

The audit step is interactive and needs human judgment about what is actually sensitive.
Don't regenerate the baseline just to silence the hook, since an unaudited baseline means
the scanner is quietly passing over real secrets.

## Tests

```bash
make test        # no stack needed; the live-stack checks skip themselves
make test-live   # against a running stack, via its published ports
```

`tests/test_pages.py` executes every Streamlit page through `AppTest`. HTTP 200 from the
server proves nothing: pages run per session over a websocket, so a bad `st.*` call only
surfaces when the script actually renders.

`make test-live` runs from the host, which is why it passes `CHROMA_PORT=8100` explicitly.

To check the stack the way a fresh clone would see it (this deletes the models, filings,
portfolio and saved prices):

```bash
make clean && make up    # re-downloads about 4.6 GB
make health
make index-all           # needs SEC_USER_AGENT in .env
```

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on every pull request. It invokes
the *same* `.pre-commit-config.yaml` via `pre-commit/action` rather than re-declaring lint
rules, so CI and local hooks cannot disagree about what passing means. Local hooks only
protect contributors who ran `pre-commit install` and never passed `--no-verify`.

CI starts no service containers: the live-stack checks skip themselves, so it verifies
lint, config and page rendering without pulling 4.6 GB of models.

## `.env` protection

`.claude/settings.json` is committed and denies Claude Code's `Read` and `Grep` tools on
real `.env` files while leaving `.env.example` readable. `Grep` matters as much as `Read`
because it returns matched lines, so a search for a variable name could otherwise print
its value.

It blocks `Read`/`Grep` calls that name a `.env` path or glob. It does **not** block shell
commands like `cat .env`, because a path-glob permission rule can't reliably tell those
from safe ones. It covers everyone who opens this repo with Claude Code, and complements
(not replaces) a personal global rule in `~/.claude/settings.json`.

### Testing the rule

This needs a **fresh Claude Code session**, because permission config loads at session
start. Use a throwaway `.env`, and back up any real one first, since this overwrites it:

```bash
printf 'DENY_RULE_CANARY=this-is-not-a-real-secret\n' > .env
```

Then start a new `claude` session in this directory and try all three:

| Ask | Expected |
|---|---|
| "read .env" | **blocked** by permissions |
| "grep for DENY_RULE_CANARY" | **blocked** |
| "read .env.example" | **works**, contents returned |

Pass means the canary value never appears in a reply. `/permissions` lists the loaded
rules if you want a quicker check. Delete the `.env` afterwards.

Two things that look like failures but aren't:

- **`cat .env` via Bash is not blocked, by design**, as described above.
- **The `.env` must be inside the project tree.** The glob resolves relative to the working
  directory, so one in `/tmp` won't match even with a correct rule.

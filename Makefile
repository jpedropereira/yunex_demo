# Yunex demo -- infrastructure targets.
#
# Three ways to run, differing only in how Ollama is reached:
#   make up           CPU, everything in Docker. Works anywhere, no host setup.
#   make up-gpu       NVIDIA. Linux needs nvidia-container-toolkit (see README).
#   make up-external  Ollama on the host. Required on macOS.

COMPOSE      := docker compose
COMPOSE_FILES ?=
GPU_FILES    := -f compose.yaml -f compose.gpu.yaml
EXT_FILES    := -f compose.yaml -f compose.external.yaml
APP_PORT     ?= 8501
CHROMA_PORT  ?= 8100
OLLAMA_PORT  ?= 11434

.DEFAULT_GOAL := help
.PHONY: help up up-gpu up-external _up_local down pull warm health logs ps lint test test-live clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# Bring up the local-LLM stack in three explicit phases. Model provisioning
# runs in the foreground on purpose: `up --wait` can report a previously
# completed model-init as satisfied and return before the *new* pull finishes,
# which would print "Ready" while Stack Health still fails. Running it in the
# foreground also shows download progress, which matters on a 4.6GB first run.
_up_local:
	$(COMPOSE) $(COMPOSE_FILES) up -d --wait ollama chroma
	$(COMPOSE) $(COMPOSE_FILES) run --rm model-init
	$(COMPOSE) $(COMPOSE_FILES) up -d --build --wait app

up: ## Start the stack on CPU (default; first run downloads ~4.6GB of models)
	@$(MAKE) --no-print-directory _up_local
	@echo "Ready: http://localhost:$(APP_PORT)  (open Stack Health first)"

up-gpu: ## Start the stack with NVIDIA GPU acceleration (falls back to CPU if unavailable)
	@if ./docker/preflight-gpu.sh; then \
		$(MAKE) --no-print-directory _up_local COMPOSE_FILES="$(GPU_FILES)" && \
		echo "Ready: http://localhost:$(APP_PORT)" && \
		echo "Confirm the GPU was really used -- see the Acceleration row in Stack Health."; \
	else \
		$(MAKE) --no-print-directory up; \
	fi

up-external: ## Start the stack against an Ollama running on the host (macOS path)
	@command -v ollama >/dev/null 2>&1 \
		|| echo "WARNING: no 'ollama' on PATH. Install it and run 'ollama serve' first."
	$(COMPOSE) $(EXT_FILES) up -d --build --wait
	@echo "Ready: http://localhost:$(APP_PORT)"

pull: ## Provision models ahead of time (do this before the event, on good wifi)
	$(COMPOSE) up -d --wait ollama
	$(COMPOSE) run --rm model-init

warm: ## Load models into memory so the first on-stage call is not cold
	$(COMPOSE) exec -T app python -m yunex_demo.warm

health: ## Run the full stack check headlessly (exits non-zero on failure)
	$(COMPOSE) exec -T app python -m yunex_demo.health

logs: ## Tail logs from every service
	$(COMPOSE) logs -f --tail=100

ps: ## Show service status
	$(COMPOSE) ps

down: ## Stop the stack (keeps models and vector data)
	$(COMPOSE) down --remove-orphans

clean: ## Stop and delete all volumes -- re-downloads models on next up
	$(COMPOSE) down -v --remove-orphans

lint: ## Format and lint
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run unit tests (no running stack required; live checks self-skip)
	uv run pytest -q

test-live: ## Run tests against the running stack, via its published ports
	OLLAMA_HOST=http://localhost:$(OLLAMA_PORT) \
	CHROMA_HOST=localhost \
	CHROMA_PORT=$(CHROMA_PORT) \
	uv run pytest -q -rs

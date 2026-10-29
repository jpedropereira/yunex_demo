# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependency layer first, so editing source does not reinstall the world.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# README.md is referenced by pyproject's metadata, so hatchling needs it to
# build the project. It lives in the source layer, not the dependency layer,
# so editing it does not invalidate the cached dependency install above.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

EXPOSE 8501

# fileWatcherType=poll: bind-mounted source does not reliably deliver inotify
# events under Docker Desktop, and hot reload dies silently without it.
CMD ["streamlit", "run", "src/yunex_demo/ui/Home.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=poll", \
     "--browser.gatherUsageStats=false"]

"""End-to-end stack checks for the demo infrastructure.

Pure functions returning dataclasses -- no Streamlit imports -- so the same
checks back both the Stack Health page and ``python -m yunex_demo.health``,
which is what you run backstage before presenting.

Each check captures its own exception and returns a failed result rather than
raising, so one broken service never hides the state of the others.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from yunex_demo.config import settings

# Loading the model is all the acceleration check needs, so it generates a
# single token.
_PING_PROMPT = "Reply with exactly: ready"

# The benchmark needs a prompt that actually runs for a while -- a prompt the
# model can satisfy in two tokens produces a throughput figure dominated by
# noise. Temperature 0 plus a fixed seed keeps the token count stable run to run.
_BENCH_PROMPT = "Write three sentences explaining why structured output matters."
_BENCH_MAX_TOKENS = 64

# Below this, a tok/s figure is not worth reporting as a measurement.
_MIN_MEANINGFUL_TOKENS = 10

_ROUNDTRIP_DOCS = [
    "Traffic signal controller at junction 14 reported a lamp failure.",
    "Quarterly invoice for roadside sensor maintenance, net 30 terms.",
]


@dataclass
class CheckResult:
    """Outcome of a single check."""

    name: str
    ok: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None


def _timed(name: str, fn) -> CheckResult:
    """Run ``fn`` and wrap failures into a result instead of propagating them."""
    started = time.perf_counter()
    try:
        result = fn()
        result.elapsed_s = time.perf_counter() - started
        return result
    except Exception as exc:  # noqa: BLE001 - a failed check is data, not a crash
        return CheckResult(
            name=name,
            ok=False,
            summary=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def _ollama_client():
    from ollama import Client

    return Client(host=settings.ollama_host, timeout=120)


def check_ollama() -> CheckResult:
    """Server reachable, and both required models actually present."""

    def run() -> CheckResult:
        version = "unknown"
        with httpx.Client(timeout=10) as http:
            resp = http.get(f"{settings.ollama_host.rstrip('/')}/api/version")
            resp.raise_for_status()
            version = resp.json().get("version", "unknown")

        installed = [m.model for m in _ollama_client().list().models]
        missing = [m for m in settings.required_models if m not in installed]

        return CheckResult(
            name="Ollama",
            ok=not missing,
            summary=(
                f"v{version}, {len(installed)} model(s) installed"
                if not missing
                else f"v{version}, missing: {', '.join(missing)}"
            ),
            details={
                "host": settings.ollama_host,
                "version": version,
                "installed": installed,
                "missing": missing,
            },
        )

    return _timed("Ollama", run)


def check_acceleration() -> CheckResult:
    """Report what Ollama actually loaded onto the GPU.

    Deliberately does not trust the compose profile: it runs a real generation,
    then reads ``/api/ps`` and compares resident VRAM against total model size.
    A `gpu` profile on a host without nvidia-container-toolkit degrades silently
    to CPU, and this is the check that catches it.
    """

    def run() -> CheckResult:
        client = _ollama_client()
        client.generate(
            model=settings.model_main,
            prompt=_PING_PROMPT,
            keep_alive=settings.ollama_keep_alive,
            options={"num_predict": 1, "num_ctx": settings.ollama_num_ctx},
        )

        loaded = [m for m in client.ps().models if m.model == settings.model_main]
        if not loaded:
            return CheckResult(
                name="Acceleration",
                ok=False,
                summary="model is not resident after a generation call",
                details={"model": settings.model_main},
            )

        entry = loaded[0]
        total = entry.size or 0
        vram = entry.size_vram or 0
        pct = (vram / total * 100) if total else 0.0

        if vram == 0:
            mode, detail = "CPU", "no layers offloaded"
        elif pct >= 99:
            mode, detail = "GPU", "fully offloaded"
        else:
            mode, detail = "GPU (partial)", f"{pct:.0f}% offloaded, rest on CPU"

        return CheckResult(
            name="Acceleration",
            ok=True,  # CPU is a supported mode, not a failure
            summary=f"{mode} -- {detail}",
            details={
                "mode": mode,
                "size_bytes": total,
                "size_vram_bytes": vram,
                "offload_pct": round(pct, 1),
            },
        )

    return _timed("Acceleration", run)


def benchmark_generation() -> CheckResult:
    """Measure generation throughput so slow-path surprises surface early."""

    def run() -> CheckResult:
        resp = _ollama_client().generate(
            model=settings.model_main,
            prompt=_BENCH_PROMPT,
            keep_alive=settings.ollama_keep_alive,
            options={
                "num_predict": _BENCH_MAX_TOKENS,
                "num_ctx": settings.ollama_num_ctx,
                "temperature": settings.temperature,
                "seed": settings.seed,
            },
        )

        tokens = resp.eval_count or 0
        # eval_duration is nanoseconds.
        duration_s = (resp.eval_duration or 0) / 1e9
        tps = tokens / duration_s if duration_s else 0.0

        meaningful = tokens >= _MIN_MEANINGFUL_TOKENS
        summary = f"{tps:.1f} tok/s ({tokens} tokens in {duration_s:.2f}s)"
        if not meaningful:
            summary += " -- too few tokens to be a reliable measure"

        return CheckResult(
            name="Generation speed",
            ok=tps > 0 and meaningful,
            summary=summary,
            details={
                "model": settings.model_main,
                "tokens_per_second": round(tps, 2),
                "eval_count": tokens,
                "eval_duration_s": round(duration_s, 3),
            },
        )

    return _timed("Generation speed", run)


# --------------------------------------------------------------------------- #
# Chroma
# --------------------------------------------------------------------------- #


def _chroma_client():
    import chromadb

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def check_chroma() -> CheckResult:
    """Heartbeat plus a collection listing, to prove the API is serving."""

    def run() -> CheckResult:
        client = _chroma_client()
        client.heartbeat()
        collections = [c.name for c in client.list_collections()]

        api_version = "unknown"
        try:
            with httpx.Client(timeout=10) as http:
                # Reports the API contract version ("1.0.0"), not the server build.
                resp = http.get(f"{settings.chroma_url}/api/v2/version")
                if resp.status_code == 200:
                    api_version = resp.text.strip('"')
        except httpx.HTTPError:
            pass  # cosmetic; the heartbeat above is the real check

        return CheckResult(
            name="ChromaDB",
            ok=True,
            summary=f"reachable (API v{api_version}), {len(collections)} collection(s)",
            details={
                "url": settings.chroma_url,
                "api_version": api_version,
                "collections": collections,
            },
        )

    return _timed("ChromaDB", run)


def check_roundtrip() -> CheckResult:
    """Embed -> upsert -> query -> verify, then clean up.

    The only check that exercises Ollama, the embedding model, and Chroma
    *together*. If this is green the infrastructure is genuinely wired up.
    """

    def run() -> CheckResult:
        chroma = _chroma_client()
        # Chroma requires 3-512 chars of [a-zA-Z0-9._-], starting and ending
        # alphanumeric -- a leading underscore is rejected.
        name = f"healthcheck-{uuid.uuid4().hex[:8]}"

        embeddings = (
            _ollama_client()
            .embed(
                model=settings.model_embed,
                input=_ROUNDTRIP_DOCS,
            )
            .embeddings
        )
        dim = len(embeddings[0])

        # embedding_function=None: we supply vectors ourselves, and we must not
        # let Chroma fall back to downloading its default ONNX embedder.
        collection = chroma.create_collection(name=name, embedding_function=None)
        try:
            ids = [f"doc-{i}" for i in range(len(_ROUNDTRIP_DOCS))]
            collection.upsert(ids=ids, documents=_ROUNDTRIP_DOCS, embeddings=embeddings)

            # Query with the first document's own vector; it must come back first.
            got = collection.query(query_embeddings=[embeddings[0]], n_results=1)
            top_id = got["ids"][0][0]
            matched = top_id == ids[0]
            stored = collection.count()
        finally:
            chroma.delete_collection(name=name)

        return CheckResult(
            name="Embed/store/query round trip",
            ok=matched and stored == len(_ROUNDTRIP_DOCS),
            summary=(
                f"{stored} docs stored, {dim}-dim vectors, nearest neighbour correct"
                if matched
                else f"nearest neighbour was {top_id!r}, expected {ids[0]!r}"
            ),
            details={
                "embed_model": settings.model_embed,
                "dimensions": dim,
                "docs_stored": stored,
                "top_match": top_id,
            },
        )

    return _timed("Embed/store/query round trip", run)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

CHECKS = (
    check_ollama,
    check_chroma,
    check_acceleration,
    benchmark_generation,
    check_roundtrip,
)


def run_all() -> list[CheckResult]:
    """Run every check in order. Ordered so connectivity fails before inference."""
    return [check() for check in CHECKS]


def main() -> int:
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name.ljust(width)}  {r.summary}  ({r.elapsed_s:.2f}s)")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} checks failed.")
        return 1
    print(f"All {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

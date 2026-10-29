"""Preload models into memory so the first on-stage call is not a cold start.

On CPU a cold load of an effective-2B model costs several seconds on top of
generation. Run ``make warm`` before presenting; combined with
``OLLAMA_KEEP_ALIVE=-1`` the weights then stay resident.
"""

from __future__ import annotations

import sys
import time

from ollama import Client

from yunex_demo.config import settings


def main() -> int:
    client = Client(host=settings.ollama_host, timeout=600)

    for label, fn in (
        (
            f"chat model {settings.model_main}",
            lambda: client.generate(
                model=settings.model_main,
                prompt="ok",
                keep_alive=settings.ollama_keep_alive,
                options={"num_predict": 1, "num_ctx": settings.ollama_num_ctx},
            ),
        ),
        (
            f"embedding model {settings.model_embed}",
            lambda: client.embed(
                model=settings.model_embed,
                input="warmup",
                keep_alive=settings.ollama_keep_alive,
            ),
        ),
    ):
        started = time.perf_counter()
        print(f"warming {label} ...", flush=True)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"  resident in {time.perf_counter() - started:.2f}s")

    resident = [m.model for m in client.ps().models]
    print(f"loaded: {', '.join(resident) if resident else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

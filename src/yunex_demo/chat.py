"""The bare model: a chat loop with nothing but a system prompt.

No tools, no retrieval, no output schema. This is the baseline the later demo
beats improve on -- it answers fluently, but nothing checks that the answer is
well-formed, grounded, or true.

No Streamlit imports, so the logic stays testable and the page stays thin.
"""

from __future__ import annotations

from collections.abc import Iterator

from ollama import Client

from yunex_demo.config import settings

SYSTEM_PROMPT = "You are a helpful assistant"


def build_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend the system prompt to the user/assistant turns so far.

    The model is stateless: every call resends the whole conversation, so the
    system prompt goes first on every request, not just the first one.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history]


def stream_reply(history: list[dict[str, str]]) -> Iterator[str]:
    """Stream the model's next reply, one text fragment at a time."""
    client = Client(host=settings.ollama_host, timeout=120)
    stream = client.chat(
        model=settings.model_main,
        messages=build_messages(history),
        stream=True,
        keep_alive=settings.ollama_keep_alive,
        options={
            "num_ctx": settings.ollama_num_ctx,
            "temperature": settings.temperature,
            "seed": settings.seed,
        },
    )
    for chunk in stream:
        yield chunk.message.content or ""

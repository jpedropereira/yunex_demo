"""Smoke-test every Streamlit page by actually executing its script.

HTTP 200 from the server proves nothing: Streamlit runs page scripts per
session over a websocket, so a typo in a page only surfaces when it renders.
AppTest executes the script and collects any exception.

The Stack Health page talks to Ollama and Chroma, so that one test skips
itself when the stack is not reachable -- `make test` stays green offline,
and verifies for real when the stack is up.
"""

import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
from streamlit.testing.v1 import AppTest

from yunex_demo.config import settings

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "yunex_demo" / "ui"

STUB_PAGES = [
    "pages/1_One_Off.py",
    "pages/2_Workflow.py",
    "pages/3_RAG_Agent.py",
]


def _run(relative: str, timeout: int = 180) -> AppTest:
    app = AppTest.from_file(str(UI_DIR / relative), default_timeout=timeout)
    app.run()
    return app


@pytest.mark.parametrize("page", ["Home.py", *STUB_PAGES])
def test_page_renders_without_exception(page):
    app = _run(page)
    assert not app.exception, f"{page} raised: {[e.value for e in app.exception]}"
    assert app.title, f"{page} rendered no title"


@pytest.mark.parametrize("page", STUB_PAGES)
def test_stub_pages_say_they_are_placeholders(page):
    """Stub pages must be obviously unfinished, so nobody demos an empty page."""
    app = _run(page)
    assert app.info, f"{page} has no placeholder notice"


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _stack_is_up() -> bool:
    ollama = urlparse(settings.ollama_host)
    return _reachable(ollama.hostname or "localhost", ollama.port or 11434) and _reachable(
        settings.chroma_host, settings.chroma_port
    )


@pytest.mark.skipif(not _stack_is_up(), reason="stack not reachable; start it with `make up`")
def test_stack_health_page_runs_every_check():
    """Exercises Ollama and Chroma for real. Skipped when the stack is down."""
    app = _run("pages/0_Stack_Health.py", timeout=600)
    assert not app.exception, f"Stack Health raised: {[e.value for e in app.exception]}"

    # One success banner when everything passes, otherwise an error banner.
    assert app.success or app.error, "no overall verdict rendered"

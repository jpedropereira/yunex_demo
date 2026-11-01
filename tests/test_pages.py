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

CHATBOT_PAGE = "pages/1_Bare_Chatbot.py"
ONE_OFF_PAGE = "pages/2_One_Off.py"
WORKFLOW_PAGE = "pages/3_Workflow.py"

STUB_PAGES = [
    "pages/4_Embedding_Search.py",
    "pages/5_RAG_Agent.py",
]


def _run(relative: str, timeout: int = 180) -> AppTest:
    app = AppTest.from_file(str(UI_DIR / relative), default_timeout=timeout)
    app.run()
    return app


@pytest.mark.parametrize(
    "page", ["Home.py", CHATBOT_PAGE, ONE_OFF_PAGE, WORKFLOW_PAGE, *STUB_PAGES]
)
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


@pytest.mark.skipif(not _stack_is_up(), reason="stack not reachable; start it with `make up`")
def test_bare_chatbot_answers_a_question():
    """Sends one real message to the model. Skipped when the stack is down."""
    app = _run(CHATBOT_PAGE)
    app.chat_input[0].set_value("Reply with one short sentence: what is a traffic light?").run()
    assert not app.exception, f"chatbot raised: {[e.value for e in app.exception]}"
    assert not app.error, f"model call failed: {[e.value for e in app.error]}"

    roles = [message.name for message in app.chat_message]
    assert roles == ["user", "assistant"]


def test_one_off_waits_for_an_upload():
    """With no file uploaded, the page asks for one instead of reading from disk.

    The real extraction is tested in test_extract.py: AppTest cannot upload files.
    """
    app = _run(ONE_OFF_PAGE)
    assert not app.exception, f"One-off raised: {[e.value for e in app.exception]}"
    assert app.info, "no prompt to upload a PDF"
    assert not app.button, "the run button showed before any upload"


def _table_with_column(app: AppTest, column: str) -> list[dict] | None:
    """The rows of the table that has this column, or None if the page isn't showing it."""
    for table in app.dataframe:
        if column in table.value.columns:
            return table.value.to_dict("records")
    return None


def _all_tables(app: AppTest) -> list[list[dict]]:
    return [table.value.to_dict("records") for table in app.dataframe]


def test_workflow_explains_the_pipeline_before_anything_runs():
    app = _run(WORKFLOW_PAGE)
    assert not app.exception, f"Workflow raised: {[e.value for e in app.exception]}"
    charts = app.get("graphviz_chart")
    assert len(charts) == 1, "no pipeline diagram on a fresh page"
    assert "Classify" in charts[0].proto.spec
    assert _table_with_column(app, "Validate") is None, "results grid showed before any run"


def test_workflow_explains_each_step_in_prose_before_anything_runs():
    app = _run(WORKFLOW_PAGE)
    explanations = app.expander[0]
    assert explanations.label == "What each step does"
    assert explanations.proto.expanded, "the explanations should be open before a run"
    assert len(explanations.markdown) == 5
    assert explanations.markdown[0].value.startswith("**1. Classify (LLM).**")
    assert explanations.markdown[4].value.startswith("**5. Load (Code).**")


def test_workflow_load_sample_inbox_then_again_then_reset(fake_model):
    """The acceptance run, with a fake model: AppTest runs the page in this process."""
    app = _run(WORKFLOW_PAGE)
    # Without this the test would sit through the pauses meant for the audience.
    app.toggle(key="slow_down").set_value(False)

    app.button(key="run_samples").click().run()
    assert not app.exception, f"Workflow raised: {[e.value for e in app.exception]}"

    grid = _table_with_column(app, "Validate")
    assert [row["File"] for row in grid] == [
        "trade_confirmation_solaris.pdf",
        "capital_call_acme.pdf",
        "trade_confirmation_helios.pdf",
    ]
    assert [row["Outcome"] for row in grid] == ["Loaded", "Loaded", "Needs review"]

    positions = _table_with_column(app, "Before → after")
    assert positions[0]["Before → after"] == "1,000 → 1,200"

    cash_calls = _table_with_column(app, "Amount due")
    assert cash_calls[0]["Amount due"] == "USD 100,000.00"
    assert cash_calls[0]["Due"] == "30 Nov 2026"
    assert cash_calls[0]["Status"] == "pending"

    review = _table_with_column(app, "Reason")
    assert len(review) == 1
    assert review[0]["File"] == "trade_confirmation_helios.pdf"
    assert "gross + commission = total" in review[0]["Reason"]

    # The last document is selected by default, so its failure is what's on screen.
    assert any(tab.label == "❌ Validate" for tab in app.tabs)

    # Every document gets its own raw outputs, whichever one is selected.
    code_blocks = [block.value for block in app.code]
    assert '{"doc_type": "capital_call"}' in code_blocks
    assert any('"reference": "TC-2026-10418"' in block for block in code_blocks)
    assert any('"reference": "TC-2026-10422"' in block for block in code_blocks)
    assert "(this step did not run)" in code_blocks, "Helios never reached Load"

    tables_before_second_run = [positions, cash_calls, review]
    app.button(key="run_samples").click().run()
    assert [
        _table_with_column(app, "Before → after"),
        _table_with_column(app, "Amount due"),
        _table_with_column(app, "Reason"),
    ] == tables_before_second_run, "the second run changed something"
    assert [row["Outcome"] for row in _table_with_column(app, "Validate")] == [
        "Already loaded",
        "Already loaded",
        "Needs review",
    ]

    app.button(key="reset").click().run()
    assert len(_all_tables(app)) == 1
    assert not any(header.value == "Raw outputs" for header in app.subheader)
    assert _table_with_column(app, "Before → after")[0]["Before → after"] == "1,000 → 1,000"

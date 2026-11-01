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
from conftest import ALPHA_ACCESSION, BETA_ACCESSION, filing_html, filing_url
from streamlit.testing.v1 import AppTest

from yunex_demo.config import settings
from yunex_demo.sec import ingest, search, store

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "yunex_demo" / "ui"

CHATBOT_PAGE = "pages/1_Bare_Chatbot.py"
ONE_OFF_PAGE = "pages/2_One_Off.py"
WORKFLOW_PAGE = "pages/3_Workflow.py"
EMBEDDING_PAGE = "pages/4_Embedding_Search.py"

STUB_PAGES = [
    "pages/5_RAG_Agent.py",
]


def _run(relative: str, timeout: int = 180) -> AppTest:
    app = AppTest.from_file(str(UI_DIR / relative), default_timeout=timeout)
    app.run()
    return app


@pytest.mark.parametrize(
    "page", ["Home.py", CHATBOT_PAGE, ONE_OFF_PAGE, WORKFLOW_PAGE, EMBEDDING_PAGE, *STUB_PAGES]
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


def test_embedding_page_explains_itself_when_chroma_is_down(monkeypatch):
    def chroma_is_down():
        raise ValueError("Could not connect to a Chroma server.")

    monkeypatch.setattr(store, "get_collection", chroma_is_down)
    app = _run(EMBEDDING_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "Could not reach ChromaDB" in app.error[0].value


def test_embedding_page_asks_for_the_user_agent_only_when_missing(monkeypatch, sec_collection):
    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)

    monkeypatch.setattr(settings, "sec_user_agent", "")
    assert any("SEC_USER_AGENT" in warning.value for warning in _run(EMBEDDING_PAGE).warning)

    monkeypatch.setattr(settings, "sec_user_agent", "Test Name test@example.com")
    assert not _run(EMBEDDING_PAGE).warning


def test_embedding_page_rejects_a_non_edgar_url_without_fetching(monkeypatch, sec_collection):
    def must_not_fetch(url):
        raise AssertionError("a rejected URL must never be fetched")

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "fetch_filing", must_not_fetch)

    app = _run(EMBEDDING_PAGE)
    app.text_input(key="url").set_value("https://evil.example/filing.htm")
    app.button(key="index").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "Not an EDGAR document URL" in app.error[0].value


def test_embedding_page_says_already_indexed_without_fetching(monkeypatch, two_filings):
    def must_not_fetch(url):
        raise AssertionError("an indexed filing must not be fetched again")

    monkeypatch.setattr(store, "get_collection", lambda: two_filings)
    monkeypatch.setattr(ingest, "fetch_filing", must_not_fetch)

    app = _run(EMBEDDING_PAGE)
    app.text_input(key="url").set_value(filing_url(ALPHA_ACCESSION))
    app.button(key="index").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert app.success[0].value == "Already indexed: 12 pages"


def test_embedding_page_shows_k_results_and_respects_the_filter(monkeypatch, two_filings):
    monkeypatch.setattr(store, "get_collection", lambda: two_filings)

    def result_headings() -> list[str]:
        return [item.value for item in app.markdown if item.value.startswith("#### ")]

    def result_lines() -> list[str]:
        return [item.value for item in app.text if "· page" in item.value]

    app = _run(EMBEDDING_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    table = app.dataframe[0].value
    assert list(table["Company"]) == ["ALPHA CORP", "BETA CORP"]

    app.text_input(key="question").set_value("export controls on sales to China").run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert len(result_headings()) == 5, "the default k is 5"

    app.slider(key="k").set_value(10).run()
    assert len(result_headings()) == 10
    assert {line.split(" · ")[0] for line in result_lines()} == {"ALPHA CORP", "BETA CORP"}

    selector = app.selectbox(key="document")
    assert selector.options == [
        "All documents",
        "ALPHA CORP · 10-K · FY2026",
        "BETA CORP · 10-K · FY2026",
    ]
    selector.set_value(BETA_ACCESSION).run()
    assert len(result_headings()) == 10, "filtering happens before ranking, so still k results"
    assert {line.split(" · ")[0] for line in result_lines()} == {"BETA CORP"}


def test_embedding_page_suggested_question_searches_and_open_search_still_works(
    monkeypatch, two_filings
):
    searched = []
    real_search = search.search

    def recording_search(question, k, collection, doc_id=None):
        searched.append(question)
        return real_search(question, k, collection, doc_id=doc_id)

    monkeypatch.setattr(store, "get_collection", lambda: two_filings)
    monkeypatch.setattr(search, "search", recording_search)

    app = _run(EMBEDDING_PAGE)
    assert not searched, "nothing should be searched until someone asks"

    suggestion = app.button(key="suggestion_3")
    assert suggestion.label == "Can rivals steal their customers?"
    suggestion.click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert app.text_input(key="question").value == suggestion.label
    assert searched == [suggestion.label]
    assert [item.value for item in app.markdown if item.value.startswith("#### ")]

    # Typing over a suggestion must win, and must survive the rerun.
    app.text_input(key="question").set_value("my own question").run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert searched[-1] == "my own question"
    assert app.text_input(key="question").value == "my own question"


def test_embedding_page_sample_buttons_index_and_then_show_as_done(monkeypatch, sec_collection):
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return filing_html("ALPHA CORP", ["word " * 40, "word " * 40])

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)
    apple = ingest.SAMPLE_FILINGS[1]

    app = _run(EMBEDDING_PAGE)
    assert app.text_input(key="url").value == "", "the paste box starts empty"
    assert not app.button(key="sample_Apple").disabled

    app.button(key="sample_Apple").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert fetched == [apple.url]
    assert app.success[0].value == "Indexed 2 pages, skipped 0."
    assert app.button(key="sample_Apple").disabled
    assert app.button(key="sample_Apple").label == "Apple · 2 pages"
    assert not app.button(key="sample_NVIDIA").disabled
    assert list(app.dataframe[0].value["Company"]) == ["ALPHA CORP"]


def test_embedding_page_index_all_fetches_only_whats_missing_and_reports_failures(
    monkeypatch, sec_collection
):
    failing_urls = {ingest.SAMPLE_FILINGS[2].url}
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        if url in failing_urls:
            raise ingest.FetchError("The SEC returned HTTP 500.")
        return filing_html("ALPHA CORP", ["word " * 40, "word " * 40])

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)

    app = _run(EMBEDDING_PAGE)
    app.button(key="sample_NVIDIA").click().run()
    fetched.clear()

    app.button(key="index_all").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert fetched == [sample.url for sample in ingest.SAMPLE_FILINGS[1:]]
    assert "HTTP 500" in app.error[0].value
    assert len(app.success) == 3
    assert not app.button(key="index_all").disabled, "the failed filing is still missing"

    # Retrying fetches only the one that failed, and then nothing is left to index.
    failing_urls.clear()
    fetched.clear()
    app.button(key="index_all").click().run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert fetched == [ingest.SAMPLE_FILINGS[2].url]
    assert app.button(key="index_all").disabled

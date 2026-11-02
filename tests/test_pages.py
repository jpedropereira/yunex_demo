"""Smoke-test every Streamlit page by actually executing its script.

HTTP 200 from the server proves nothing: Streamlit runs page scripts per
session over a websocket, so a typo in a page only surfaces when it renders.
AppTest executes the script and collects any exception.

The Stack Health page talks to Ollama and Chroma, so that one test skips
itself when the stack is not reachable -- `make test` stays green offline,
and verifies for real when the stack is up.
"""

import socket
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
import streamlit as st
from conftest import (
    ALPHA_ACCESSION,
    BETA_ACCESSION,
    fake_latest_10k_url,
    filing_html,
    filing_url,
)
from ollama import ChatResponse, Message
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.testing.v1 import AppTest

from yunex_demo.agent import context_usage, loop, market, tools
from yunex_demo.config import settings
from yunex_demo.sec import ingest, search, store

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "yunex_demo" / "ui"

CHATBOT_PAGE = "pages/1_Bare_Chatbot.py"
ONE_OFF_PAGE = "pages/2_One_Off.py"
WORKFLOW_PAGE = "pages/3_Workflow.py"
EMBEDDING_PAGE = "pages/4_Embedding_Search.py"
AGENT_PAGE = "pages/5_Agent.py"


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch, tmp_path):
    """Drawing a page must never touch Yahoo, a real database or a real Chroma."""

    def chroma_is_down():
        raise ValueError("Could not connect to a Chroma server.")

    def fake_prices(tickers):
        return {ticker: [["2026-10-29", "190.00"], ["2026-10-30", "200.00"]] for ticker in tickers}

    monkeypatch.setattr(store, "get_collection", chroma_is_down)
    monkeypatch.setattr(market, "_download", fake_prices)
    monkeypatch.setattr(
        context_usage,
        "_count_prompt_tokens",
        lambda messages, tool_definitions: 500 if tool_definitions else 200,
    )
    monkeypatch.setattr(settings, "agent_db_path", tmp_path / "agent.db")
    monkeypatch.setattr(settings, "price_cache_path", tmp_path / "prices.json")


def _run(relative: str, timeout: int = 180) -> AppTest:
    app = AppTest.from_file(str(UI_DIR / relative), default_timeout=timeout)
    app.run()
    return app


@pytest.mark.parametrize(
    "page", ["Home.py", CHATBOT_PAGE, ONE_OFF_PAGE, WORKFLOW_PAGE, EMBEDDING_PAGE, AGENT_PAGE]
)
def test_page_renders_without_exception(page):
    app = _run(page)
    assert not app.exception, f"{page} raised: {[e.value for e in app.exception]}"
    assert app.title, f"{page} rendered no title"


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


def test_embedding_page_explains_an_index_built_by_an_older_version(monkeypatch):
    def index_without_tickers(collection):
        raise KeyError("ticker")

    monkeypatch.setattr(store, "get_collection", lambda: None)
    monkeypatch.setattr(search, "list_documents", index_without_tickers)

    app = _run(EMBEDDING_PAGE)

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "make reindex" in app.error[0].value


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


def test_embedding_page_buttons_index_the_latest_10k_and_show_as_done(monkeypatch, sec_collection):
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return filing_html("ALPHA CORP", ["word " * 40, "word " * 40])

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "latest_10k_url", fake_latest_10k_url)
    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)

    app = _run(EMBEDDING_PAGE)
    assert app.text_input(key="url").value == "", "the paste box starts empty"
    assert app.button(key="latest_AAPL").label == "Apple"

    app.button(key="latest_AAPL").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert fetched == [fake_latest_10k_url(320193)]
    assert app.success[0].value == "Indexed 2 pages, skipped 0."
    assert app.button(key="latest_AAPL").label == "Apple · 2 pages"
    assert app.button(key="latest_NVDA").label == "NVIDIA"
    assert list(app.dataframe[0].value["Company"]) == ["ALPHA CORP"]

    # Still clickable, because the company may have filed since. Here it hasn't.
    fetched.clear()
    app.button(key="latest_AAPL").click().run()
    assert not fetched
    assert app.success[0].value == "Already indexed: 2 pages"


def test_embedding_page_index_all_reports_failures_and_retrying_fetches_only_whats_missing(
    monkeypatch, sec_collection
):
    microsoft_url = fake_latest_10k_url(789019)
    failing_urls = {microsoft_url}
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        if url in failing_urls:
            raise ingest.FetchError("The SEC returned HTTP 500.")
        return filing_html("ALPHA CORP", ["word " * 40, "word " * 40])

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "latest_10k_url", fake_latest_10k_url)
    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)

    app = _run(EMBEDDING_PAGE)
    app.button(key="index_all").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert len(fetched) == len(settings.agent_filings)
    assert "HTTP 500" in app.error[0].value
    assert len(app.success) == 4

    failing_urls.clear()
    fetched.clear()
    app.button(key="index_all").click().run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert fetched == [microsoft_url]


def test_embedding_page_rejects_a_company_outside_the_five(monkeypatch, sec_collection):
    def must_not_fetch(url):
        raise AssertionError("a rejected company must never be fetched")

    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr(ingest, "fetch_filing", must_not_fetch)

    app = _run(EMBEDDING_PAGE)
    app.text_input(key="url").set_value(filing_url(ALPHA_ACCESSION, cik=2488))
    app.button(key="index").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "Only filings from these companies" in app.error[0].value


# --- The agent page -----------------------------------------------------------


def _reply(text: str = "", tool_calls: list | None = None) -> ChatResponse:
    calls = [
        Message.ToolCall(function=Message.ToolCall.Function(name=name, arguments=arguments))
        for name, arguments in tool_calls or []
    ]
    return ChatResponse(
        message=Message(role="assistant", content=text, tool_calls=calls or None),
        done_reason="stop",
        prompt_eval_count=650,
        eval_count=40,
    )


def _script_model(monkeypatch, replies: list[ChatResponse]) -> None:
    remaining = iter(replies)
    monkeypatch.setattr(loop, "call_model", lambda request: next(remaining))


def _positions(app: AppTest) -> dict[str, int]:
    rows = _table_with_column(app, "Shares")
    return {row["Ticker"]: row["Shares"] for row in rows}


def _trade_statuses(app: AppTest) -> list[str]:
    return [row["Status"] for row in _table_with_column(app, "Status")]


def test_agent_page_shows_the_portfolio_and_opens_with_no_calls():
    app = _run(AGENT_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert _positions(app) == {"NVDA": 120, "AAPL": 150, "MSFT": 80, "GOOGL": 140, "AMZN": 160}
    assert [tab.label for tab in app.tabs] == ["Chat", "Tools", "Context", "How it works"]
    assert any("Could not reach ChromaDB" in warning.value for warning in app.warning)


def test_agent_page_lists_every_tool_the_model_can_ask_for():
    app = _run(AGENT_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    tools_tab = app.tabs[1]
    table = tools_tab.table[0].value
    assert set(table["Tool"]) == {f"`{name}`" for name in tools.TOOL_NAMES}
    effects = dict(zip(table["Tool"], table["Effect"], strict=True))
    assert effects.pop("`propose_trade`") == "Waits for a human"
    assert set(effects.values()) == {"Only reads"}
    assert "These 5 tools are all the agent can ask for" in tools_tab.markdown[0].value


def test_agent_page_approving_a_trade_changes_the_portfolio_once(monkeypatch):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Buy 10 NVDA shares.").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert _positions(app)["NVDA"] == 120, "proposing must change nothing"
    assert _trade_statuses(app) == ["pending"]

    app.button(key="approve_1").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert _positions(app)["NVDA"] == 130
    assert _trade_statuses(app) == ["executed"]
    assert not any(button.key == "approve_1" for button in app.button), "the card should be gone"
    assert any("Trade #1 approved and executed" in text.value for text in app.text)


def test_agent_page_rejecting_a_trade_changes_nothing(monkeypatch):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Buy 10 NVDA shares.").run()
    app.button(key="reject_1").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert _positions(app)["NVDA"] == 120
    assert _trade_statuses(app) == ["rejected"]


def test_agent_page_shows_the_guards_decision_and_the_full_context(monkeypatch):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "TSLA", "side": "buy", "quantity": 10})]
            ),
            _reply("TSLA is not on the allowlist."),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Buy 10 TSLA.").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any(item.value.startswith("**⛔ deny**") for item in app.markdown)
    assert any("allowlist_hook: TSLA is not on the allowlist" in t.value for t in app.text)
    assert _trade_statuses(app) == ["denied"]
    assert app.selectbox[0].options == [
        "Call 1 · turn 1 · 650 prompt tokens · asked for propose_trade",
        "Call 2 · turn 1 · 650 prompt tokens · answered",
    ]
    assert app.selectbox[0].value == 1, "the newest call is shown first"
    assert any("You are a research and trading assistant" in block.value for block in app.code)


def test_agent_page_flags_a_citation_that_was_never_retrieved(monkeypatch):
    _script_model(monkeypatch, [_reply("Made in Cupertino (Apple 10-K, p. 99).")])
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Where does Apple make its products?").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any(text.value == "Apple 10-K, p. 99" for text in app.text)
    assert any("never read this page" in block.value for block in app.markdown)


def test_agent_page_a_failed_model_call_leaves_no_half_finished_question(monkeypatch):
    def model_is_down(request):
        raise ConnectionError("Ollama went away")

    monkeypatch.setattr(loop, "call_model", model_is_down)
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Hello").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "Model call failed" in app.error[0].value
    assert app.session_state.agent_chat == []
    assert len(app.session_state.agent_messages) == 1, "only the system prompt should remain"


def test_agent_page_without_prices_disables_trading(monkeypatch):
    def yahoo_is_down(tickers):
        raise ValueError("rate limited")

    monkeypatch.setattr(market, "_download", yahoo_is_down)
    app = _run(AGENT_PAGE)

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any("Trading is disabled" in error.value for error in app.error)


def test_agent_page_says_when_prices_come_from_the_cache(monkeypatch):
    _run(AGENT_PAGE)

    def yahoo_is_down(tickers):
        raise ValueError("rate limited")

    monkeypatch.setattr(market, "_download", yahoo_is_down)
    app = _run(AGENT_PAGE)

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any(text.value.startswith("Prices cached at ") for text in app.text)


def _context_summaries(app: AppTest) -> list[str]:
    return [item.value for item in app.caption if "tokens used" in item.value]


def test_agent_page_shows_how_full_the_context_is_and_it_grows(monkeypatch):
    app = _run(AGENT_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    # Measured before anything is asked: the scripted system prompt and tool definitions.
    assert _context_summaries(app)[0].startswith("500 of 8,192 tokens used")
    assert any('class="context-bar"' in item.value for item in app.markdown)

    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("get_portfolio", {})]), _reply("It holds five companies.")],
    )
    app.chat_input[0].set_value("What is in the portfolio?").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    # The newest call: 650 prompt tokens and 40 of reply, from the scripted model.
    summary = _context_summaries(app)[0]
    assert summary.startswith("690 of 8,192 tokens used · 7,502 free")
    assert "estimated" not in summary
    assert app.session_state.agent_fixed_costs.tool_definitions == 300

    breakdown = _table_with_column(app, "Tokens")
    assert sum(row["Tokens"] for row in breakdown) == 8192
    assert [row["Part"] for row in breakdown] == [
        "System prompt",
        "Tool definitions",
        "User",
        "Tool results",
        "Model",
        "Free",
    ]


def test_agent_page_says_when_the_context_is_nearly_full(monkeypatch):
    nearly_full = _reply("A long answer.")
    nearly_full.prompt_eval_count = 6000
    nearly_full.eval_count = 1500
    _script_model(monkeypatch, [nearly_full])
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Hello").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any("Nearly full" in item.value and "6,692" in item.value for item in app.caption)
    assert any(text.value.startswith("Total:") for text in app.text), "the sidebar still draws"


def test_agent_page_still_opens_when_ollama_cannot_be_measured(monkeypatch):
    def ollama_is_down(messages, tool_definitions):
        raise ConnectionError("Ollama went away")

    monkeypatch.setattr(context_usage, "_count_prompt_tokens", ollama_is_down)
    app = _run(AGENT_PAGE)

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert "estimated, because Ollama" in _context_summaries(app)[0]


def test_agent_page_warns_when_a_reply_was_cut_short(monkeypatch):
    cut_short = _reply("Half an answ")
    cut_short.done_reason = "length"
    _script_model(monkeypatch, [cut_short])
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Hello").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any("cut short" in warning.value for warning in app.warning)


def test_agent_page_new_conversation_keeps_the_portfolio(monkeypatch):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Buy 10 NVDA shares.").run()
    app.button(key="approve_1").click().run()

    next(b for b in app.button if b.label == "New conversation").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert not app.chat_message
    assert _positions(app)["NVDA"] == 130

    next(b for b in app.button if b.label == "Reset portfolio").click().run()
    assert _positions(app)["NVDA"] == 120


def _render_as_growing_text(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "render_prompt",
        lambda request: "".join(f"<{m['role']}>{m['content']}" for m in request["messages"]),
    )


def _code_after(app: AppTest, heading: str) -> str:
    """The code block drawn right after a heading in the selected call's first tab."""
    tab = next(tab for tab in app.tabs if tab.label == "Prompt as the model reads it")
    blocks = list(tab.children.values())
    position = next(i for i, block in enumerate(blocks) if getattr(block, "value", "") == heading)
    return blocks[position + 1].value


def test_agent_page_shows_each_calls_literal_prompt_split_into_resent_and_new(monkeypatch):
    _render_as_growing_text(monkeypatch)
    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("get_portfolio", {})]), _reply("It holds five companies.")],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("What is in the portfolio?").run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"

    # The newest call is shown first, and the two parts join back into the whole prompt.
    resent = _code_after(app, "##### Resent from the last call")
    new = _code_after(app, "##### New in this call")
    whole = "".join(
        f"<{m['role']}>{m['content']}" for m in app.session_state.agent_calls[1].request["messages"]
    )
    assert resent + new == whole
    assert resent.endswith("What is in the portfolio?")
    assert new.startswith("<assistant><tool>")

    app.selectbox[0].select_index(0).run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    new = _code_after(app, "##### New in this call")
    assert new.startswith("<system>You are a research and trading assistant")
    assert new.endswith("What is in the portfolio?")


def test_agent_page_says_when_ollama_did_not_return_the_prompt(monkeypatch):
    _script_model(monkeypatch, [_reply("Hello.")])
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Hello").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any("didn't return the final text" in info.value for info in app.info)


def test_agent_page_shows_quotes_unescaped_page_by_page(monkeypatch):
    page = search.SearchHit(
        text="Line one\nLine two",
        page=14,
        part=1,
        similarity=0.71,
        fiscal_year="2025",
        form="10-K",
        company="APPLE INC",
        source_url="https://example.test",
        ticker="AAPL",
        printed_page=12,
    )
    monkeypatch.setattr(store, "get_collection", lambda: None)
    monkeypatch.setattr(search, "search", lambda *args, **kwargs: [page])
    monkeypatch.setattr(search, "list_documents", lambda collection: [])
    _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "Where?"})]),
            _reply("In many places (Apple 10-K, p. 12)."),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Where does Apple make its products?").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any(text.value == "Apple 10-K, p. 12" for text in app.text)
    assert any(block.value == "Line one\nLine two" for block in app.code)
    assert any(text.value == "Apple 10-K, p. 12: kept 1 sentence" for text in app.text)
    assert not any(tab.label == "Page checks" for tab in app.tabs), "the newest call only answered"

    app.selectbox[0].select_index(0).run()
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    checks = next(tab for tab in app.tabs if tab.label == "Page checks")
    assert any("Copy, word for word" in block.value for block in checks.code)
    assert any(text.value == "Apple 10-K, p. 12 · similarity 0.71 · 0.0s" for text in checks.text)


FILING_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
FILING_PAGE = (
    "The Company depends on component suppliers. Substantially all hardware products are "
    "manufactured by outsourcing partners located primarily in China mainland and India. "
    "Some of the 10 sites are **bold** and <b>marked up</b> in the filing."
)


def _ask_about_filing(monkeypatch, answers: list[str], page_text: str = FILING_PAGE) -> AppTest:
    """Each question searches Apple's filing once, and then gets the next of these answers."""
    page = search.SearchHit(
        text=page_text,
        page=14,
        part=1,
        similarity=0.71,
        fiscal_year="2025",
        form="10-K",
        company="APPLE INC",
        source_url=FILING_URL,
        ticker="AAPL",
        printed_page=12,
    )
    monkeypatch.setattr(store, "get_collection", lambda: None)
    monkeypatch.setattr(search, "search", lambda *args, **kwargs: [page])
    monkeypatch.setattr(search, "list_documents", lambda collection: [])
    replies = []
    for answer in answers:
        replies.append(
            _reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "Where?"})])
        )
        replies.append(_reply(answer))
    _script_model(monkeypatch, replies)

    app = _run(AGENT_PAGE)
    for _ in answers:
        app.chat_input[0].set_value("Where does Apple make its products?").run()
        assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    return app


CLOSE_ANSWER = (
    "Substantially all hardware products are manufactured by outsourcing partners located "
    "primarily in China mainland and India (Apple 10-K, p. 12)."
)


def test_agent_page_shows_a_claim_beside_its_closest_passage(monkeypatch):
    app = _ask_about_filing(monkeypatch, [CLOSE_ANSWER])

    assert any(text.value == CLOSE_ANSWER.split(" (")[0] for text in app.text)
    assert any(block.value.startswith("Substantially all hardware") for block in app.code)
    assert any(block.value == FILING_PAGE for block in app.code), "the whole page, one tab along"
    assert any("Closest passage" in block.value for block in app.markdown)
    chat_tab = app.tabs[0]
    assert not any("verified" in block.value.lower() for block in chat_tab.markdown)
    assert any("1 of 1 citations have a close passage" in c.value for c in app.caption)


def test_agent_page_links_to_the_passage_in_the_filing(monkeypatch):
    app = _ask_about_filing(monkeypatch, [CLOSE_ANSWER])
    assert app.get("link_button")[0].proto.url.startswith(f"{FILING_URL}#:~:text=")


def test_agent_page_flags_a_page_that_does_not_back_the_claim(monkeypatch):
    app = _ask_about_filing(monkeypatch, ["Rockets launch from Florida (Apple 10-K, p. 12)."])

    assert any("No close passage" in block.value for block in app.markdown)
    assert any("0 of 1 citations have a close passage" in c.value for c in app.caption)


def test_agent_page_flags_a_number_that_is_not_on_the_page(monkeypatch):
    answer = CLOSE_ANSWER.replace("Substantially all", "All 75")
    app = _ask_about_filing(monkeypatch, [answer])
    assert any("says 75, and that number isn't" in text.value for text in app.text)


def test_agent_page_shows_filing_text_as_text_never_as_markup(monkeypatch):
    app = _ask_about_filing(monkeypatch, ["Some sites are bold and marked up (Apple 10-K, p. 12)."])
    assert any("<b>marked up</b>" in block.value for block in app.code)
    assert not any("<b>marked up</b>" in block.value for block in app.markdown)


def test_agent_page_lists_the_pages_read_when_an_answer_cites_none(monkeypatch):
    app = _ask_about_filing(monkeypatch, ["They use outsourcing partners."])

    assert any("cites no page" in warning.value for warning in app.warning)
    assert any(text.value == "Apple 10-K, p. 12" for text in app.text)
    assert any(block.value == FILING_PAGE for block in app.code)


def test_agent_page_answers_citing_the_same_page_do_not_collide(monkeypatch):
    """Identical widgets with no key of their own make Streamlit raise an error."""
    two_claims = f"{CLOSE_ANSWER} {CLOSE_ANSWER.replace('primarily ', '')}"
    app = _ask_about_filing(monkeypatch, [two_claims, CLOSE_ANSWER, "No citation at all."])
    assert len(app.session_state.agent_chat[1]["evidence"]) == 2


def _how_it_works_tab(app: AppTest):
    # By label, because each model call adds tabs of its own to the list.
    return next(tab for tab in app.tabs if tab.label == "How it works")


def test_agent_page_explains_how_it_works_before_any_question():
    app = _run(AGENT_PAGE)
    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"

    how_it_works = _how_it_works_tab(app)
    assert any("Copy, word for word" in block.value for block in how_it_works.code)
    assert how_it_works.table[0].value["Rule"].tolist()[0] == "Stay on topic"
    assert not any(t.value.startswith("Last tool call") for t in how_it_works.text)


def test_agent_page_how_it_works_shows_what_the_hooks_said_last(monkeypatch):
    _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("<b>**made_up**</b>", {})]),
            _reply("That tool doesn't exist."),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Use a tool that doesn't exist.").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    how_it_works = _how_it_works_tab(app)
    assert any(t.value == "Last tool call: <b>**made_up**</b>" for t in how_it_works.text)
    said = how_it_works.dataframe[0].value.set_index("Hook")["Last tool call"]
    assert said["known_tool_hook"] == "⛔ deny"
    assert said["approval_hook"] == "—"


def test_agent_page_how_it_works_shows_the_last_search_in_numbers(monkeypatch):
    page = search.SearchHit(
        text="<script>alert(1)</script> Apple makes things in China. **Bold** and more words.",
        page=14,
        part=1,
        similarity=0.71,
        fiscal_year="2025",
        form="10-K",
        company="APPLE INC",
        source_url="https://example.test",
        ticker="AAPL",
        printed_page=12,
    )
    monkeypatch.setattr(store, "get_collection", lambda: None)
    monkeypatch.setattr(search, "search", lambda *args, **kwargs: [page])
    monkeypatch.setattr(search, "list_documents", lambda collection: [])
    _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "Where?"})]),
            _reply("In China (Apple 10-K, p. 12)."),
        ],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Where does Apple make its products?").run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    how_it_works = _how_it_works_tab(app)
    funnel = next(t.value for t in how_it_works.text if t.value.startswith("Front pages"))
    assert "pages read: 1" in funnel
    assert "script" not in funnel


def test_agent_page_answer_cant_load_an_image_or_hide_a_link(monkeypatch):
    _script_model(
        monkeypatch,
        [_reply("Costs $5 ![x](http://attacker.test/?q=1) and [safe](http://attacker.test) <b>")],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("How much?").run()

    answer = next(block.value for block in app.markdown if "attacker.test" in block.value)
    assert answer == (
        r"Costs \$5 !\[x](http://attacker.test/?q=1) and \[safe](http://attacker.test) \<b>"
    )


def test_agent_page_keeps_the_answer_when_a_click_interrupts_the_script(monkeypatch):
    """A click stops the script at its next st.* call, which here is the spinner closing."""
    real_spinner = st.spinner

    @contextmanager
    def spinner_interrupted_by_a_click(text, **kwargs):
        with real_spinner(text, **kwargs):
            yield
        if text == "Thinking...":
            raise StopException()

    _script_model(monkeypatch, [_reply("The answer survives.")])
    app = _run(AGENT_PAGE)
    monkeypatch.setattr(st, "spinner", spinner_interrupted_by_a_click)
    app.chat_input[0].set_value("A question?").run()

    app.run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    assert any(block.value == "A question?" for block in app.markdown)
    assert any(block.value == "The answer survives." for block in app.markdown)
    assert len(app.session_state.agent_calls) == 1
    assert app.session_state.agent_messages[-1]["content"] == "The answer survives."


def test_agent_page_failed_model_call_leaves_no_trace_of_the_question(monkeypatch):
    def model_is_down(request):
        raise ConnectionError("Ollama is down")

    monkeypatch.setattr(loop, "call_model", model_is_down)
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("A question?").run()

    assert "Model call failed" in app.error[0].value
    assert app.session_state.agent_chat == []
    assert [message["role"] for message in app.session_state.agent_messages] == ["system"]


def test_agent_page_card_shows_the_refreshed_price_approve_would_use(monkeypatch):
    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})])],
    )
    app = _run(AGENT_PAGE)
    app.chat_input[0].set_value("Buy 10 NVDA shares.").run()
    assert any("at USD 200.00 = USD 2,000.00" in text.value for text in app.text)

    monkeypatch.setattr(
        market,
        "_download",
        lambda tickers: {
            ticker: [["2026-10-30", "200.00"], ["2026-10-31", "210.00"]] for ticker in tickers
        },
    )
    next(button for button in app.button if button.label == "Refresh prices").click().run()

    assert not app.exception, f"page raised: {[e.value for e in app.exception]}"
    card = next(text.value for text in app.text if text.value.startswith("buy 10 NVDA"))
    assert "at USD 210.00 = USD 2,100.00" in card
    assert "Proposed at USD 200.00" in card

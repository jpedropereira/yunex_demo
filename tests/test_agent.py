"""The agent: guards one by one and together, tools, the loop, approval, grounding.

No network, no Ollama, no Chroma server. The model is scripted, so these tests pin down
what code does with any answer the model could give.
"""

import json
import re
from datetime import datetime
from decimal import Decimal

import httpx
import pytest
from ollama import ChatResponse, Message

from yunex_demo.agent import (
    approval,
    citations,
    context_usage,
    explain,
    hooks,
    loop,
    market,
    portfolio,
    quotes,
    tools,
)
from yunex_demo.agent.hooks import HookState, ToolCall, run_pre_hooks
from yunex_demo.agent.tools import AgentContext
from yunex_demo.config import settings
from yunex_demo.sec import search, store

# Kept before the autouse fixture in conftest replaces it, so it can be tested itself.
REAL_RENDER_PROMPT = loop.render_prompt

CLOSES = {
    "NVDA": [["2026-08-03", "160.00"], ["2026-10-29", "190.00"], ["2026-10-30", "200.00"]],
    "AAPL": [["2026-08-03", "250.00"], ["2026-10-29", "310.00"], ["2026-10-30", "300.00"]],
    "MSFT": [["2026-10-30", "500.00"]],
    "GOOGL": [["2026-10-30", "300.00"]],
    "AMZN": [["2026-10-30", "200.00"]],
}
# 120*200 + 150*300 + 80*500 + 140*300 + 160*200
TOTAL_VALUE = Decimal(183000)


def _snapshot(from_cache: bool = False) -> market.PriceSnapshot:
    return market._build_snapshot(CLOSES, datetime(2026, 10, 30, 16, 5), from_cache)


@pytest.fixture
def conn(tmp_path):
    return portfolio.connect(tmp_path / "agent.db")


@pytest.fixture
def context(conn):
    return AgentContext(conn, _snapshot())


def _state(user_messages: tuple[str, ...] = ()) -> HookState:
    return HookState(
        positions=dict(portfolio.SEED_POSITIONS),
        prices=_snapshot().latest_prices(),
        user_messages=user_messages,
    )


def _trade(ticker: str = "NVDA", side: str = "buy", **size) -> ToolCall:
    return ToolCall("propose_trade", {"ticker": ticker, "side": side, **size})


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


def _script_model(monkeypatch, replies: list[ChatResponse]) -> list[dict]:
    """Answer with these replies in order, and keep a copy of what each call received."""
    received = []
    remaining = iter(replies)

    def fake_call_model(request):
        received.append(json.loads(json.dumps(request)))
        return next(remaining)

    monkeypatch.setattr(loop, "call_model", fake_call_model)
    return received


def _turn(question: str, context: AgentContext) -> tuple[loop.TurnResult, list[dict]]:
    messages = loop.new_conversation()
    messages.append({"role": "user", "content": question})
    return loop.run_turn(messages, context), messages


# --- Hooks, one by one --------------------------------------------------------


def test_known_tool_hook():
    assert hooks.known_tool_hook(ToolCall("get_quote", {}), _state()) is None
    assert hooks.known_tool_hook(ToolCall("wire_money", {}), _state()).decision == "deny"


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("get_quote", {}),
        ToolCall("get_quote", {"ticker": 7}),
        ToolCall("search_filing", {"ticker": "AAPL", "question": " "}),
        ToolCall("get_price_history", {"ticker": "AAPL", "period": "5y"}),
        ToolCall("propose_trade", {"ticker": "AAPL", "side": "short", "quantity": 1}),
        ToolCall("propose_trade", {"ticker": "AAPL", "side": "buy"}),
        ToolCall(
            "propose_trade", {"ticker": "AAPL", "side": "buy", "quantity": 1, "portfolio_pct": 1}
        ),
        ToolCall("propose_trade", {"ticker": "AAPL", "side": "buy", "portfolio_pct": 0}),
        ToolCall("propose_trade", {"ticker": "AAPL", "side": "buy", "portfolio_pct": 101}),
        ToolCall("propose_trade", {"ticker": "AAPL", "side": "buy", "portfolio_pct": "lots"}),
    ],
)
def test_arguments_hook_denies_the_wrong_shape(call):
    assert hooks.arguments_hook(call, _state()).decision == "deny"


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("get_portfolio", {}),
        ToolCall("search_filing", {"ticker": "AAPL", "question": "q"}),
        ToolCall("get_price_history", {"ticker": "AAPL", "period": "3mo"}),
        _trade(quantity=1),
        _trade(portfolio_pct=30),
        # Which ticker is allowed is a separate question, for the next hook to answer.
        ToolCall("get_quote", {"ticker": "TSLA"}),
    ],
)
def test_arguments_hook_has_no_opinion_on_the_right_shape(call):
    assert hooks.arguments_hook(call, _state()) is None


def test_allowlist_hook():
    denied = hooks.allowlist_hook(ToolCall("get_quote", {"ticker": "TSLA"}), _state())
    assert (denied.decision, denied.reason) == ("deny", "TSLA is not on the allowlist")
    assert hooks.allowlist_hook(ToolCall("get_quote", {"ticker": "AAPL"}), _state()) is None
    assert hooks.allowlist_hook(ToolCall("get_portfolio", {}), _state()) is None


def _search(ticker: str) -> ToolCall:
    return ToolCall("search_filing", {"ticker": ticker, "question": "where"})


def test_mention_hook_warns_on_the_wrong_company():
    state = _state(("Where does Apple make its products?",))
    result = hooks.ticker_mention_hook(_search("NVDA"), state)
    assert result.decision == "warn"
    assert "AAPL" in result.reason


def test_mention_hook_accepts_the_right_company():
    state = _state(("Where does Apple make its products?",))
    assert hooks.ticker_mention_hook(_search("AAPL"), state) is None


def test_mention_hook_ignores_a_follow_up_that_names_no_company():
    assert hooks.ticker_mention_hook(_search("NVDA"), _state(("and their risks?",))) is None


def test_mention_hook_reads_the_newest_message_that_names_a_company():
    state = _state(("Tell me about Apple.", "Now NVIDIA.", "and their risks?"))
    assert hooks.ticker_mention_hook(_search("NVDA"), state) is None
    assert hooks.ticker_mention_hook(_search("AAPL"), state).decision == "warn"


def test_mention_hook_knows_both_of_alphabets_names():
    assert hooks.ticker_mention_hook(_search("GOOGL"), _state(("What about Google?",))) is None
    assert hooks.ticker_mention_hook(_search("GOOGL"), _state(("And Alphabet?",))) is None


def test_mention_hook_accepts_each_company_in_a_comparison():
    state = _state(("Compare NVIDIA's and Apple's supply-chain risks",))
    assert hooks.ticker_mention_hook(_search("NVDA"), state) is None
    assert hooks.ticker_mention_hook(_search("AAPL"), state) is None


def test_mention_hook_only_applies_to_search():
    call = ToolCall("get_quote", {"ticker": "NVDA"})
    assert hooks.ticker_mention_hook(call, _state(("Apple?",))) is None


@pytest.mark.parametrize("quantity", [0, -5, 1.5, "10", True])
def test_quantity_hook_denies_anything_but_a_positive_whole_number(quantity):
    assert hooks.quantity_hook(_trade(quantity=quantity), _state()).decision == "deny"


def test_quantity_hook_denies_a_percentage_too_small_for_one_share():
    assert hooks.quantity_hook(_trade(portfolio_pct=0.0001), _state()).decision == "deny"


def test_holdings_hook():
    result = hooks.holdings_hook(_trade("AAPL", "sell", quantity=500), _state())
    assert (result.decision, result.reason) == ("deny", "can't sell 500 AAPL, holding 150")
    assert hooks.holdings_hook(_trade("AAPL", "sell", quantity=150), _state()) is None
    assert hooks.holdings_hook(_trade("AAPL", "buy", quantity=500), _state()) is None


def test_size_limit_hook_explains_with_the_numbers():
    result = hooks.size_limit_hook(_trade(quantity=46), _state())
    assert result.decision == "deny"
    assert result.reason == "trade value USD 9,200.00 is above the 5% limit of USD 9,150.00"


def test_size_limit_hook_does_not_apply_to_sells():
    """Selling 100 AAPL is 16% of the portfolio, and it is capped by holdings instead."""
    assert hooks.size_limit_hook(_trade("AAPL", "sell", quantity=100), _state()) is None


def test_approval_hook_always_asks_for_a_trade():
    assert hooks.approval_hook(_trade(quantity=1), _state()).decision == "ask"
    assert hooks.approval_hook(ToolCall("get_portfolio", {}), _state()) is None


# --- Hooks together -----------------------------------------------------------

_EXACT_LIMIT_STATE = HookState(positions={"NVDA": 1000}, prices={"NVDA": Decimal(10)})


@pytest.mark.parametrize(
    ("call", "state", "decision", "deciding_hook"),
    [
        (ToolCall("get_quote", {"ticker": "AAPL"}), _state(), "allow", ""),
        (ToolCall("get_portfolio", {}), _state(), "allow", ""),
        (ToolCall("get_quote", {"ticker": "TSLA"}), _state(), "deny", "allowlist_hook"),
        (_trade("TSLA", quantity=10), _state(), "deny", "allowlist_hook"),
        (_trade(quantity=0), _state(), "deny", "quantity_hook"),
        (_trade(quantity=-3), _state(), "deny", "quantity_hook"),
        (_trade("AAPL", "sell", quantity=500), _state(), "deny", "holdings_hook"),
        (_trade(quantity=50), _EXACT_LIMIT_STATE, "ask", "approval_hook"),
        (_trade(quantity=51), _EXACT_LIMIT_STATE, "deny", "size_limit_hook"),
        (_trade(portfolio_pct=30), _state(), "deny", "size_limit_hook"),
        (_trade(quantity=10), _state(), "ask", "approval_hook"),
        (_trade("AAPL", "sell", quantity=100), _state(), "ask", "approval_hook"),
        (ToolCall("wire_money", {}), _state(), "deny", "known_tool_hook"),
    ],
)
def test_run_pre_hooks(call, state, decision, deciding_hook):
    verdict = run_pre_hooks(call, state)
    assert (verdict.decision, verdict.hook) == (decision, deciding_hook)


def test_a_deny_beats_an_ask_whichever_comes_first():
    def asks(call, state):
        return hooks.HookResult("ask", "ask", "asks")

    def denies(call, state):
        return hooks.HookResult("deny", "no", "denies")

    verdict = run_pre_hooks(ToolCall("x", {}), _state(), hooks=[asks, denies])
    assert (verdict.decision, verdict.hook) == ("deny", "denies")


def test_a_warning_changes_nothing_but_is_kept():
    verdict = run_pre_hooks(_search("NVDA"), _state(("Apple?",)))
    assert verdict.decision == "allow"
    assert [warning.hook for warning in verdict.warnings] == ["ticker_mention_hook"]


def test_every_hook_that_ran_is_on_record():
    verdict = run_pre_hooks(_trade("AAPL", "sell", quantity=500), _state())
    names = [name for name, _ in verdict.checks]
    assert names[-1] == "holdings_hook", "a deny stops the list there"
    assert names[:3] == ["known_tool_hook", "arguments_hook", "allowlist_hook"]


def test_without_prices_trading_is_denied_but_reading_is_not():
    state = HookState(positions=dict(portfolio.SEED_POSITIONS), prices={})
    assert run_pre_hooks(_trade(quantity=1), state).hook == "prices_hook"
    assert run_pre_hooks(ToolCall("get_portfolio", {}), state).decision == "allow"


def test_a_percentage_becomes_whole_shares_rounding_down():
    # 30% of 183,000 is 54,900, and at 200 a share that is 274.5.
    assert (
        tools.trade_quantity(
            {"ticker": "NVDA", "portfolio_pct": 30}, TOTAL_VALUE, {"NVDA": Decimal(200)}
        )
        == 274
    )


# --- Tools --------------------------------------------------------------------


def test_tickers_are_plain_strings_so_a_bad_one_reaches_the_guard():
    """With an enum, the model refuses TSLA itself, and nothing is logged or shown."""
    for tool in tools.TOOLS:
        ticker = tool["function"]["parameters"]["properties"].get("ticker")
        if ticker:
            assert "enum" not in ticker


def test_get_price_history_computes_the_change(context):
    result = tools.get_price_history({"ticker": "NVDA", "period": "3mo"}, context)
    assert result["start_date"] == "2026-08-03"
    assert (result["start_price"], result["end_price"]) == ("160.00", "200.00")
    assert result["change_pct"] == "25.00"


def test_get_quote_computes_the_day_change_and_range(context):
    result = tools.get_quote({"ticker": "AAPL"}, context)
    assert result["price"] == "300.00"
    assert result["day_change_pct"] == "-3.23"
    assert (result["low_52w"], result["high_52w"]) == ("250.00", "310.00")
    assert result["as_of"] == "2026-10-30"
    assert result["cached"] is False


def test_get_portfolio_weights_sum_to_100(context):
    result = tools.get_portfolio({}, context)
    assert result["total_value"] == "183000.00"
    assert result["max_buy_value"] == "9150.00"
    weights = [Decimal(position["weight_pct"]) for position in result["positions"]]
    assert abs(sum(weights) - 100) <= Decimal("0.05")


def test_read_tools_without_prices_say_so(conn):
    context = AgentContext(conn, None)
    assert tools.get_quote({"ticker": "AAPL"}, context)["status"] == "no_prices"
    assert tools.get_portfolio({}, context)["status"] == "no_prices"


def test_search_filing_returns_only_that_company_even_if_another_is_more_similar(
    monkeypatch, sec_collection, context
):
    def metadata(ticker: str, page: int) -> dict:
        return {
            "company": ticker,
            "ticker": ticker,
            "form": "10-K",
            "fiscal_year": "2026",
            "period_end": "January 25, 2026",
            "source_url": "",
            "doc_id": ticker,
            "page": page,
            "part": 1,
            # Unlike the position above, so a test can tell which of the two reaches a citation.
            "printed_page": page - 2,
        }

    sec_collection.upsert(
        ids=["nvda", "aapl"],
        documents=["a perfect match", "a poor match"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[metadata("NVDA", 1), metadata("AAPL", 7)],
    )
    monkeypatch.setattr(store, "embed_texts", lambda texts: [[1.0, 0.0]])
    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)

    # Without the filter, NVIDIA's page really is the closest.
    assert search.search("q", 1, sec_collection)[0].ticker == "NVDA"

    result = tools.search_filing({"ticker": "AAPL", "question": "q"}, context)

    assert result["results"][0]["cite"] == "Apple 10-K, p. 5", "the number printed in the filing"
    assert result["results"][0]["quotes"] == ["a poor match"]
    assert list(context.retrieved_pages) == [("AAPL", 5)]


def test_the_model_cannot_choose_how_many_pages_come_back(monkeypatch, context):
    """Offered k, it asked for five, and comparing two companies overflowed the context."""
    asked_for = []

    def fake_search(question, k, collection, ticker):
        asked_for.append(k)
        return []

    monkeypatch.setattr(tools.search, "search", fake_search)
    monkeypatch.setattr(store, "get_collection", lambda: None)
    tools.search_filing({"ticker": "AAPL", "question": "q", "k": 10}, context)

    assert asked_for == [tools.SEARCH_RESULTS + tools.MOST_FRONT_MATTER_HITS]
    schema = next(t for t in tools.TOOLS if t["function"]["name"] == "search_filing")
    assert "k" not in schema["function"]["parameters"]["properties"]


def _hit(page: int, text: str = "A sentence on the page.", printed_page: int = -1):
    printed = page if printed_page == -1 else printed_page
    return search.SearchHit(0.7, text, "APPLE INC", "AAPL", "10-K", "2025", page, 1, "", printed)


def _search_finds(monkeypatch, hits: list) -> None:
    monkeypatch.setattr(tools.search, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(store, "get_collection", lambda: None)


def _search_apple(context) -> dict:
    return tools.search_filing({"ticker": "AAPL", "question": "q"}, context)


def _model_quotes(monkeypatch, *sentences: str) -> None:
    reply = json.dumps({"quotes": list(sentences)})
    monkeypatch.setattr(quotes, "ask_model", lambda prompt: reply)


def test_a_search_gives_the_model_only_sentences_really_on_the_page(monkeypatch, context):
    page = "Partners make the phones.\nThey   are in Vietnam. Other text the model never sees."
    _search_finds(monkeypatch, [_hit(9, page)])
    _model_quotes(monkeypatch, "They are in Vietnam.", "They are on the Moon.", " ")

    result = _search_apple(context)

    assert result == {"results": [{"cite": "Apple 10-K, p. 9", "quotes": ["They are in Vietnam."]}]}
    [review] = context.page_reviews
    assert review.kept == ["They are in Vietnam."]
    assert review.discarded == ["They are on the Moon.", " "]
    assert review.outcome == "kept 1 sentence, discarded 2 not word for word on the page"
    assert page.replace("<", "&lt;") in review.prompt


def test_a_sentence_with_an_escaped_bracket_still_counts_as_on_the_page(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9, "Margins were <5% last year.")])
    _model_quotes(monkeypatch, "Margins were &lt;5% last year.")
    assert _search_apple(context)["results"][0]["quotes"] == ["Margins were &lt;5% last year."]


def test_a_page_with_no_helpful_sentence_is_dropped_and_cannot_be_cited(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9, "Useful."), _hit(10, "Not useful.")])
    monkeypatch.setattr(
        quotes,
        "ask_model",
        lambda prompt: json.dumps({"quotes": ["Useful."] if "Useful." in prompt else []}),
    )

    result = _search_apple(context)

    assert [row["cite"] for row in result["results"]] == ["Apple 10-K, p. 9"]
    assert list(context.retrieved_pages) == [("AAPL", 9)]
    [evidence] = citations.check_answer("Useful (Apple 10-K, p. 10).", context.retrieved_pages)
    assert evidence.status == citations.NOT_RETRIEVED


def test_a_search_says_so_when_no_page_helps(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9), _hit(10)])
    _model_quotes(monkeypatch)
    assert _search_apple(context)["status"] == "no_relevant_pages"
    assert context.retrieved_pages == {}


def test_a_reply_that_is_not_valid_loses_that_page_and_says_why(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9)])
    monkeypatch.setattr(quotes, "ask_model", lambda prompt: '{"quotes": ["cut off')
    assert _search_apple(context)["status"] == "no_relevant_pages"
    assert context.page_reviews[0].outcome == "unusable reply"


def test_the_cover_and_contents_are_skipped_and_the_next_page_takes_their_place(
    monkeypatch, context
):
    _search_finds(
        monkeypatch,
        [_hit(1), _hit(2, printed_page=0), *[_hit(page) for page in range(5, 15)]],
    )
    result = _search_apple(context)

    assert [row["cite"] for row in result["results"]] == [
        f"Apple 10-K, p. {page}" for page in range(5, 5 + tools.SEARCH_RESULTS)
    ]
    assert [review.outcome for review in context.page_reviews[:2]] == [
        "cover or contents, skipped",
        "cover or contents, skipped",
    ]


def test_tool_results_keep_their_characters_unescaped(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9, "The Company’s phones.")])
    result = tools.run_tool("search_filing", {"ticker": "AAPL", "question": "q"}, context)
    assert "The Company’s phones." in result


def test_the_loop_records_page_checks_on_the_step_and_the_call_that_asked(monkeypatch, context):
    _search_finds(monkeypatch, [_hit(9)])
    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "q"})]), _reply("Ok")],
    )
    result, _ = _turn("Where?", context)

    assert [review.cite for review in result.steps[0].page_reviews] == ["Apple 10-K, p. 9"]
    assert result.calls[0].page_reviews == result.steps[0].page_reviews
    assert result.calls[1].page_reviews == []
    assert context.page_reviews == [], "collected, so the next step doesn't report them again"


def test_two_parts_of_one_page_are_kept_as_one_page(monkeypatch, context):
    def part(number: int, text: str) -> search.SearchHit:
        return search.SearchHit(0.9, text, "APPLE INC", "AAPL", "10-K", "2025", 9, number, "", 7)

    monkeypatch.setattr(store, "get_collection", lambda: None)
    for hits in ([part(1, "first part")], [part(2, "second part")], [part(1, "first part")]):
        monkeypatch.setattr(tools.search, "search", lambda *args, hits=hits, **kwargs: hits)
        tools.search_filing({"ticker": "AAPL", "question": "q"}, context)

    assert list(context.retrieved_pages) == [("AAPL", 7)]
    assert context.retrieved_pages[("AAPL", 7)].text == "first part second part"


def test_search_filing_with_nothing_indexed_says_so(monkeypatch, sec_collection, context):
    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    result = tools.search_filing({"ticker": "AAPL", "question": "q"}, context)
    assert result["status"] == "not_indexed"


# --- Market data --------------------------------------------------------------


def test_prices_fall_back_to_the_cache_when_yahoo_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "price_cache_path", tmp_path / "nested" / "prices.json")
    monkeypatch.setattr(market, "_download", lambda tickers: CLOSES)

    fresh = market.load_snapshot()
    assert not fresh.from_cache
    assert fresh.latest_prices()["NVDA"] == Decimal("200.00")

    def yahoo_is_down(tickers):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(market, "_download", yahoo_is_down)
    cached = market.load_snapshot()

    assert cached.from_cache
    assert cached.latest_prices() == fresh.latest_prices()
    assert cached.fetched_at == fresh.fetched_at


def test_no_prices_and_no_cache_is_none(monkeypatch, tmp_path):
    def yahoo_is_down(tickers):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(settings, "price_cache_path", tmp_path / "prices.json")
    monkeypatch.setattr(market, "_download", yahoo_is_down)
    assert market.load_snapshot() is None


def test_a_corrupt_cache_counts_as_no_cache(monkeypatch, tmp_path):
    def yahoo_is_down(tickers):
        raise RuntimeError("rate limited")

    cache = tmp_path / "prices.json"
    cache.write_text("{not json")
    monkeypatch.setattr(settings, "price_cache_path", cache)
    monkeypatch.setattr(market, "_download", yahoo_is_down)
    assert market.load_snapshot() is None


# --- The loop -----------------------------------------------------------------


def test_two_tool_calls_then_an_answer(monkeypatch, context):
    _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("get_quote", {"ticker": "NVDA"})]),
            _reply(tool_calls=[("get_portfolio", {})]),
            _reply("NVDA is 200."),
        ],
    )
    result, messages = _turn("What is NVDA at?", context)

    assert result.answer == "NVDA is 200."
    assert [(step.tool, step.decision) for step in result.steps] == [
        ("get_quote", "allow"),
        ("get_portfolio", "allow"),
    ]
    tool_message = messages[3]
    assert (tool_message["role"], tool_message["tool_name"]) == ("tool", "get_quote")
    assert json.loads(tool_message["content"])["price"] == "200.00"


def test_several_tool_calls_in_one_reply_each_run(monkeypatch, context):
    both = [("get_quote", {"ticker": "NVDA"}), ("get_quote", {"ticker": "AAPL"})]
    _script_model(monkeypatch, [_reply(tool_calls=both), _reply("Done.")])
    result, _ = _turn("Compare them.", context)
    assert [step.arguments["ticker"] for step in result.steps] == ["NVDA", "AAPL"]


def test_a_model_that_never_stops_is_stopped(monkeypatch, context):
    received = _script_model(
        monkeypatch, [_reply(tool_calls=[("get_portfolio", {})])] * (settings.agent_max_turns + 5)
    )
    result, messages = _turn("Keep going.", context)

    assert len(received) == settings.agent_max_turns
    assert result.answer == f"I stopped after {settings.agent_max_turns} steps without an answer."
    assert messages[-1]["role"] == "assistant", "the next question needs a well-formed history"


def test_a_denied_trade_reaches_the_model_and_the_log(monkeypatch, context):
    received = _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "AAPL", "side": "sell", "quantity": 500})]
            ),
            _reply("You only hold 150."),
        ],
    )
    result, _ = _turn("Sell 500 AAPL.", context)

    assert (result.steps[0].decision, result.steps[0].hook) == ("deny", "holdings_hook")
    told = json.loads(received[1]["messages"][-1]["content"])
    assert told == {"status": "denied", "reason": "can't sell 500 AAPL, holding 150"}

    trade = portfolio.read_trades(context.conn)[0]
    assert (trade["status"], trade["reason"]) == ("denied", "can't sell 500 AAPL, holding 150")
    assert portfolio.read_positions(context.conn)["AAPL"] == 150


@pytest.mark.parametrize(
    "arguments",
    [
        {"ticker": "TSLA", "side": "buy", "quantity": 10},
        {"ticker": ["NVDA"], "side": 7, "quantity": "lots"},
        {},
    ],
)
def test_a_denied_trade_is_logged_whatever_the_model_sent(monkeypatch, context, arguments):
    _script_model(monkeypatch, [_reply(tool_calls=[("propose_trade", arguments)]), _reply("No.")])
    result, _ = _turn("Trade.", context)

    assert result.steps[0].decision == "deny"
    assert portfolio.read_trades(context.conn)[0]["status"] == "denied"


def test_a_denied_read_is_not_a_trade(monkeypatch, context):
    _script_model(
        monkeypatch, [_reply(tool_calls=[("get_quote", {"ticker": "TSLA"})]), _reply("No.")]
    )
    _turn("Quote TSLA.", context)
    assert portfolio.read_trades(context.conn) == []


def test_an_asked_trade_is_pending_and_changes_nothing(monkeypatch, context):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    result, messages = _turn("Buy 10 NVDA.", context)

    assert result.steps[0].decision == "ask"
    told = json.loads(messages[-2]["content"])
    assert told == {"status": "pending_approval", "trade_id": 1}

    trade = portfolio.read_trade(context.conn, 1)
    assert (trade["status"], trade["quantity"], trade["price"]) == ("pending", 10, "200.00")
    assert trade["value"] == "2000.00"
    assert portfolio.read_positions(context.conn)["NVDA"] == 120


def test_a_pending_trade_ends_the_turn_however_keen_the_model_is(monkeypatch, context):
    """The real model proposed the same trade six times in one turn, once for each step."""
    proposal = ("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})
    received = _script_model(monkeypatch, [_reply(tool_calls=[proposal])] * 6)

    result, messages = _turn("Buy 10 NVDA.", context)

    assert len(received) == 1, "the model isn't called again once a trade is pending"
    assert len(portfolio.read_pending_trades(context.conn)) == 1
    assert result.answer == "Trade #1 is waiting for your approval."
    assert messages[-1] == {"role": "assistant", "content": result.answer}


def test_every_trade_in_one_reply_is_proposed_and_reported(monkeypatch, context):
    both = [
        ("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10}),
        ("propose_trade", {"ticker": "AAPL", "side": "sell", "quantity": 5}),
    ]
    _script_model(monkeypatch, [_reply(tool_calls=both)])

    result, _ = _turn("Buy 10 NVDA and sell 5 AAPL.", context)

    assert len(portfolio.read_pending_trades(context.conn)) == 2
    assert "Trade #1" in result.answer
    assert "Trade #2" in result.answer


def test_a_denied_trade_beside_a_pending_one_still_gives_its_reason(monkeypatch, context):
    both = [
        ("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10}),
        ("propose_trade", {"ticker": "AAPL", "side": "sell", "quantity": 500}),
    ]
    _script_model(monkeypatch, [_reply(tool_calls=both)])

    result, _ = _turn("Buy 10 NVDA and sell 500 AAPL.", context)

    assert "Trade #1 is waiting for your approval." in result.answer
    assert "can't sell 500 AAPL, holding 150" in result.answer


def test_research_before_a_trade_still_happens(monkeypatch, context):
    received = _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("get_quote", {"ticker": "NVDA"})]),
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    result, _ = _turn("Check NVDA, then buy 10.", context)

    assert len(received) == 2
    assert [step.tool for step in result.steps] == ["get_quote", "propose_trade"]


def test_a_percentage_trade_is_denied_with_the_limit_and_logged_in_shares(monkeypatch, context):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[
                    ("propose_trade", {"ticker": "NVDA", "side": "buy", "portfolio_pct": 30})
                ]
            ),
            _reply("Too large."),
        ],
    )
    result, _ = _turn("Put 30% of the portfolio into NVDA.", context)

    assert result.steps[0].hook == "size_limit_hook"
    assert "above the 5% limit" in result.steps[0].reason
    assert portfolio.read_trades(context.conn)[0]["quantity"] == 274


def test_a_tool_that_fails_tells_the_model_rather_than_crashing(monkeypatch, context):
    def chroma_is_down():
        raise ValueError("Could not connect to a Chroma server.")

    monkeypatch.setattr(store, "get_collection", chroma_is_down)
    received = _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "q"})]),
            _reply("Search is down."),
        ],
    )
    result, _ = _turn("Search Apple.", context)
    assert result.answer == "Search is down."
    assert json.loads(received[1]["messages"][-1]["content"])["status"] == "error"


def test_an_empty_reply_still_gives_an_answer(monkeypatch, context):
    _script_model(monkeypatch, [_reply("")])
    result, messages = _turn("Hello", context)
    assert result.answer == "The model returned no answer."
    assert messages[-1]["content"] == result.answer


def test_the_approval_note_is_not_read_as_a_question_about_that_company(monkeypatch, context):
    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "q"})]), _reply("Ok")],
    )
    monkeypatch.setattr(tools.search, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(store, "get_collection", lambda: None)
    messages = loop.new_conversation()
    messages += [
        {"role": "user", "content": "Tell me about Apple."},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": f"{approval.APPROVAL_NOTE_PREFIX} bought 10 NVDA."},
        {"role": "user", "content": "and their risks?"},
    ]
    result = loop.run_turn(messages, context)
    assert result.steps[0].warnings == []


# --- The call log -------------------------------------------------------------


def test_a_turn_with_two_tool_calls_logs_three_model_calls_exactly(monkeypatch, context):
    received = _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[("get_quote", {"ticker": "NVDA"})]),
            _reply(tool_calls=[("get_portfolio", {})]),
            _reply("Done."),
        ],
    )
    result, _ = _turn("Go.", context)

    assert [call.number for call in result.calls] == [1, 2, 3]
    for call, what_the_model_saw in zip(result.calls, received, strict=True):
        assert call.request["messages"] == what_the_model_saw["messages"]
        assert call.request["messages"][0]["role"] == "system"
        assert call.request["tools"] == tools.TOOLS
        assert call.request["options"]["seed"] == settings.seed
        assert call.response["prompt_eval_count"] == 650

    assert result.calls[0].new_messages == [0, 1], "everything is new on the first call"

    second = result.calls[1]
    assert [second.request["messages"][i]["role"] for i in second.new_messages] == [
        "assistant",
        "tool",
    ]
    assert result.calls[0].response["message"]["tool_calls"][0]["function"]["name"] == "get_quote"


def test_a_logged_call_does_not_change_as_the_conversation_grows(monkeypatch, context):
    _script_model(monkeypatch, [_reply(tool_calls=[("get_portfolio", {})]), _reply("Done.")])
    result, messages = _turn("Go.", context)
    assert len(result.calls[0].request["messages"]) == 2
    assert len(messages) == 5


def test_the_next_turn_marks_only_what_it_added_as_new(monkeypatch, context):
    _script_model(monkeypatch, [_reply("First."), _reply("Second.")])
    first, messages = _turn("One.", context)
    messages.append({"role": "user", "content": "Two."})

    second = loop.run_turn(
        messages,
        context,
        turn_number=2,
        previous_messages=first.calls[-1].request["messages"],
        first_call_number=2,
    )

    call = second.calls[0]
    assert (call.number, call.turn) == (2, 2)
    assert [call.request["messages"][i]["content"] for i in call.new_messages] == ["First.", "Two."]


# --- Keeping the history inside the context -----------------------------------


def _history_of(turns: int, words_per_turn: int = 1) -> list[dict]:
    messages = loop.new_conversation()
    for number in range(turns):
        messages.append({"role": "user", "content": f"question {number}"})
        messages.append({"role": "assistant", "content": "word " * words_per_turn})
    return messages


def test_a_short_history_is_left_alone():
    messages = _history_of(3)
    assert loop.trim_history(messages) == 0
    assert len(messages) == 7


def test_a_long_history_loses_its_oldest_whole_turns_but_never_the_system_prompt():
    messages = _history_of(30)
    dropped = loop.trim_history(messages)

    assert dropped == 11
    assert len(messages) <= loop.MAX_HISTORY_MESSAGES
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "question 11"}


def test_a_few_large_turns_are_trimmed_by_tokens_well_before_the_message_limit():
    messages = _history_of(4, words_per_turn=2500)
    assert loop.trim_history(messages) > 0
    assert len(messages) < loop.MAX_HISTORY_MESSAGES
    assert loop.estimate_context_tokens(messages) <= settings.ollama_num_ctx


def test_a_tool_result_never_outlives_its_call():
    messages = loop.new_conversation()
    for number in range(3):
        messages += [
            {"role": "user", "content": f"question {number}"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]},
            {"role": "tool", "tool_name": "x", "content": "word " * 2500},
            {"role": "assistant", "content": "answer"},
        ]
    loop.trim_history(messages)
    assert messages[1]["role"] == "user"


def test_the_turn_in_progress_is_never_dropped_and_too_much_stops_the_loop(monkeypatch, context):
    """Sending this would make Ollama drop messages silently, so it must not be sent."""
    received = _script_model(
        monkeypatch, [_reply(tool_calls=[("get_portfolio", {})]), _reply("Never sent.")]
    )
    monkeypatch.setattr(tools, "run_tool", lambda *args: "word " * 9000)

    result, messages = _turn("Go.", context)

    assert len(received) == 1
    assert "no longer fit" in result.answer
    assert messages[1] == {"role": "user", "content": "Go."}


def test_results_too_big_for_the_history_push_out_older_turns_mid_turn(monkeypatch, context):
    received = _script_model(
        monkeypatch, [_reply(tool_calls=[("get_portfolio", {})]), _reply("Fits now.")]
    )
    monkeypatch.setattr(tools, "run_tool", lambda *args: "word " * 3000)
    messages = _history_of(1, words_per_turn=3500)
    messages.append({"role": "user", "content": "Go."})

    result = loop.run_turn(messages, context)

    assert result.answer == "Fits now."
    assert result.turns_dropped == 1
    assert received[1]["messages"][1] == {"role": "user", "content": "Go."}


# --- Human approval -----------------------------------------------------------


def _pending(conn, ticker="AAPL", side="sell", quantity=100, price=Decimal("300.00")) -> int:
    return portfolio.log_trade(
        conn, ticker=ticker, side=side, quantity=quantity, price=price, status="pending", reason=""
    )


def test_approving_executes_and_approving_again_changes_nothing(conn):
    trade_id = _pending(conn, "NVDA", "buy", 10, Decimal("200.00"))

    first = approval.approve_trade(conn, trade_id, _snapshot())
    assert first.executed
    assert first.note == (
        "[Approval note] Trade #1 approved and executed: bought 10 NVDA at USD 200.00."
    )
    assert portfolio.read_positions(conn)["NVDA"] == 130
    assert portfolio.read_trade(conn, trade_id)["status"] == "executed"

    second = approval.approve_trade(conn, trade_id, _snapshot())
    assert second.note == ""
    assert portfolio.read_positions(conn)["NVDA"] == 130


def test_rejecting_changes_nothing_and_cannot_undo_an_execution(conn):
    rejected = _pending(conn)
    assert "rejected" in approval.reject_trade(conn, rejected).note
    assert portfolio.read_trade(conn, rejected)["status"] == "rejected"
    assert portfolio.read_positions(conn)["AAPL"] == 150
    assert approval.approve_trade(conn, rejected, _snapshot()).note == ""
    assert portfolio.read_positions(conn)["AAPL"] == 150

    executed = _pending(conn)
    approval.approve_trade(conn, executed, _snapshot())
    assert approval.reject_trade(conn, executed).note == ""
    assert portfolio.read_trade(conn, executed)["status"] == "executed"


def test_approval_rechecks_against_the_positions_as_they_are_now(conn):
    """Each was fine when proposed. Together they sell 200 of the 150 held."""
    first, second = _pending(conn), _pending(conn)

    assert approval.approve_trade(conn, first, _snapshot()).executed
    assert portfolio.read_positions(conn)["AAPL"] == 50

    outcome = approval.approve_trade(conn, second, _snapshot())

    assert not outcome.executed
    assert "can't sell 100 AAPL, holding 50" in outcome.note
    trade = portfolio.read_trade(conn, second)
    assert (trade["status"], trade["reason"]) == ("denied", "can't sell 100 AAPL, holding 50")
    assert portfolio.read_positions(conn)["AAPL"] == 50


def test_approval_rechecks_the_size_limit_against_todays_prices(conn):
    trade_id = _pending(conn, "NVDA", "buy", 40, Decimal("200.00"))
    expensive = {**CLOSES, "NVDA": [["2026-10-30", "2000.00"]]}
    snapshot = market._build_snapshot(expensive, datetime(2026, 10, 30), False)

    outcome = approval.approve_trade(conn, trade_id, snapshot)

    assert not outcome.executed
    assert portfolio.read_trade(conn, trade_id)["status"] == "denied"
    assert portfolio.read_positions(conn)["NVDA"] == 120


def test_approval_executes_at_the_current_price(conn):
    trade_id = _pending(conn, "NVDA", "buy", 10, Decimal("100.00"))
    approval.approve_trade(conn, trade_id, _snapshot())
    trade = portfolio.read_trade(conn, trade_id)
    assert (trade["price"], trade["value"]) == ("200.00", "2000.00")


def test_approval_without_prices_is_denied(conn):
    trade_id = _pending(conn, "NVDA", "buy", 10)
    outcome = approval.approve_trade(conn, trade_id, None)
    assert not outcome.executed
    assert portfolio.read_trade(conn, trade_id)["status"] == "denied"


def test_the_whole_trade_from_the_model_to_the_button(monkeypatch, context):
    _script_model(
        monkeypatch,
        [
            _reply(
                tool_calls=[("propose_trade", {"ticker": "NVDA", "side": "buy", "quantity": 10})]
            ),
        ],
    )
    _turn("Buy 10 NVDA.", context)
    trade = portfolio.read_pending_trades(context.conn)[0]

    assert approval.approve_trade(context.conn, trade["id"], context.snapshot).executed

    assert portfolio.read_positions(context.conn)["NVDA"] == 130
    assert portfolio.read_pending_trades(context.conn) == []


# --- The database -------------------------------------------------------------


def test_the_database_survives_a_reconnect_and_reset_reseeds_it(tmp_path):
    path = tmp_path / "data" / "agent.db"
    conn = portfolio.connect(path)
    trade_id = _pending(conn, "NVDA", "buy", 10)
    approval.approve_trade(conn, trade_id, _snapshot())
    conn.close()

    conn = portfolio.connect(path)
    assert portfolio.read_positions(conn)["NVDA"] == 130

    portfolio.reset(conn)
    assert portfolio.read_positions(conn) == portfolio.SEED_POSITIONS
    assert portfolio.read_trades(conn) == []
    assert _pending(conn) == 1, "trade numbers start again"


# --- Grounding ---------------------------------------------------------------

# Real sentences from the filings, and real answers the model gave about them.
APPLE_PAGE = (
    "The Company depends on component suppliers. Substantially all of the Company's hardware "
    "products are manufactured by outsourcing partners that are located primarily in China "
    "mainland, India, Japan, South Korea, Taiwan and Vietnam. Shipments can be delayed by "
    "events outside the control of the Company."
)
APPLE_CLAIM = (
    "Substantially all of Apple's hardware products are manufactured by outsourcing partners "
    "located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam"
)
APPLE_SENTENCE = APPLE_PAGE.split(". ")[1] + "."
NVIDIA_P28 = (
    "Export controls have already encouraged and may in the future encourage overseas "
    "governments to request that our customers purchase from our competitors rather than NVIDIA "
    "or other U.S. firms, harming our business, market position, and financial results. The "
    "Chinese government has encouraged customers to purchase from our China-based competitors "
    "and discouraged customers from purchasing, importing, or using our data center products."
)
NVIDIA_P10 = (
    "Under the current rules and geopolitical landscape, we are unable to create and deliver "
    "a competitive product for China's data center market that receives approval from both "
    "the USG and the Chinese government."
)
MICROSOFT_PAGE = (
    "HUMAN CAPITAL RESOURCES As of June 30, 2026, we employed approximately 223,000 people on "
    "a full-time basis, 121,000 in the U.S. and 102,000 internationally."
)
EDGAR_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def _pages() -> dict:
    return {
        ("AAPL", 48): tools.RetrievedPage("Apple 10-K, p. 48", APPLE_PAGE, EDGAR_URL),
        ("NVDA", 28): tools.RetrievedPage("NVIDIA 10-K, p. 28", NVIDIA_P28, ""),
        ("NVDA", 10): tools.RetrievedPage("NVIDIA 10-K, p. 10", NVIDIA_P10, ""),
        ("MSFT", 9): tools.RetrievedPage("Microsoft 10-K, p. 9", MICROSOFT_PAGE, ""),
    }


def _not_retrieved(answer: str) -> list[str]:
    """What the check found no retrieved page for, which is all the older check could see."""
    return [
        evidence.cite
        for evidence in citations.check_answer(answer, _pages())
        if evidence.status == citations.NOT_RETRIEVED
    ]


def test_a_close_paraphrase_finds_the_sentence_it_came_from():
    [evidence] = citations.check_answer(f"{APPLE_CLAIM} (Apple 10-K, p. 48).", _pages())

    assert evidence.status == citations.PASSAGE_FOUND
    assert evidence.passage == APPLE_SENTENCE, "just that sentence, not its neighbours too"
    assert evidence.claim == APPLE_CLAIM
    assert evidence.page_text == APPLE_PAGE
    assert evidence.source_url == EDGAR_URL
    assert not evidence.needs_attention


def test_a_real_page_that_does_not_say_it_is_flagged_with_where_the_wording_is():
    """The model really did this: the page was retrieved, so the older check passed it."""
    answer = (
        "The Chinese government has encouraged customers to purchase from China-based competitors "
        "and discouraged customers from purchasing, importing, or using NVIDIA's data center "
        "products (NVIDIA 10-K, p. 10)."
    )
    [evidence] = citations.check_answer(answer, _pages())

    assert evidence.status == citations.NO_CLOSE_PASSAGE
    assert evidence.needs_attention
    assert evidence.better_cite == "NVIDIA 10-K, p. 28"
    assert evidence.better_passage.startswith("The Chinese government has encouraged")


def test_an_invented_claim_has_no_close_passage_and_nowhere_better():
    answer = "Apple builds all of its hardware in its own factories in Germany (Apple 10-K, p. 48)."
    [evidence] = citations.check_answer(answer, _pages())

    assert evidence.status == citations.NO_CLOSE_PASSAGE
    assert evidence.better_cite == ""


def test_a_changed_number_is_flagged_even_though_the_words_match():
    answer = (
        "As of June 30, 2026, Microsoft employed approximately 323,000 people on a full-time "
        "basis, with 121,000 in the U.S. and 102,000 internationally (MSFT 10-K, p. 9)."
    )
    [evidence] = citations.check_answer(answer, _pages())

    assert evidence.status == citations.PASSAGE_FOUND
    assert evidence.missing_numbers == ["323,000"]
    assert evidence.needs_attention


def test_numbers_that_are_on_the_page_pass_however_the_sentence_ends():
    answer = "As of June 30, 2026, there were 223,000 people (Microsoft 10-K, p. 9)."
    assert citations.check_answer(answer, _pages())[0].missing_numbers == []


def test_a_misquote_is_flagged_even_though_the_words_match():
    """The model's real quote, which drops one word from the filing's sentence."""
    misquote = "have already and may in the future encourage overseas governments"
    answer = f'Export controls "{misquote}" to buy elsewhere (NVIDIA 10-K, p. 28).'
    assert citations.check_answer(answer, _pages())[0].misquotes == [misquote]


def test_a_real_quote_passes_with_either_kind_of_quotation_mark():
    answer = (
        "It says “request that our customers purchase from our competitors” (NVIDIA 10-K, p. 28)."
    )
    assert citations.check_answer(answer, _pages())[0].misquotes == []


def test_a_claim_that_means_the_opposite_still_matches():
    """The limit of this check, on purpose: it's why the page says "read them", not "verified"."""
    answer = APPLE_CLAIM.replace("Substantially all", "Substantially none")
    [evidence] = citations.check_answer(f"{answer} (Apple 10-K, p. 48).", _pages())
    assert evidence.status == citations.PASSAGE_FOUND
    assert not evidence.needs_attention


def test_each_claim_is_only_the_sentence_before_its_citation():
    answer = (
        "An opening remark.\n"
        "* **Supply:** Outsourcing partners make the products (Apple 10-K, p. 48). Export controls "
        "harm the business (NVIDIA 10-K, p. 28)."
    )
    assert [evidence.claim for evidence in citations.check_answer(answer, _pages())] == [
        "Outsourcing partners make the products",
        "Export controls harm the business",
    ]


def test_a_citation_after_the_full_stop_still_finds_its_claim():
    [evidence] = citations.check_answer("First. Partners make them. (Apple 10-K, p. 48)", _pages())
    assert evidence.claim == "Partners make them"


def test_a_citation_of_a_retrieved_page_is_not_flagged_as_unread():
    assert _not_retrieved("They outsource (Apple 10-K, p. 48).") == []


def test_a_page_that_was_not_retrieved_is_flagged():
    assert _not_retrieved("They outsource (Apple 10-K, p. 12).") == ["Apple 10-K, p. 12"]


def test_a_page_that_was_not_retrieved_still_says_where_the_wording_is():
    [evidence] = citations.check_answer(f"{APPLE_CLAIM} (Apple 10-K, p. 12).", _pages())
    assert evidence.status == citations.NOT_RETRIEVED
    assert evidence.better_cite == "Apple 10-K, p. 48"


def test_a_page_cited_under_the_wrong_company_is_flagged():
    assert _not_retrieved("They outsource (Amazon 10-K, p. 48).") == ["Amazon 10-K, p. 48"]


def test_a_ticker_works_as_the_company_in_a_citation():
    assert _not_retrieved("(AAPL 10-K, p. 48)") == []


def test_several_citations_in_one_pair_of_brackets_are_each_checked():
    answer = "Both do (Apple 10-K, p. 48; NVIDIA 10-K, p. 3)."
    assert _not_retrieved(answer) == ["NVIDIA 10-K, p. 3"]


@pytest.mark.parametrize(
    "cited", ["NVIDIA 10-K, p. 10, p. 26", "NVIDIA 10-K, pp. 10, 26", "NVIDIA 10-K, p. 10, 26"]
)
def test_every_page_in_a_list_is_checked(cited):
    assert _not_retrieved(f"({cited})") == ["NVIDIA 10-K, p. 26"]


def test_an_unknown_company_is_flagged():
    assert _not_retrieved("(Tesla 10-K, p. 48)") == ["Tesla 10-K, p. 48"]


def test_the_same_claim_and_citation_is_checked_once():
    answer = "One thing (Apple 10-K, p. 8). One thing (Apple 10-K, p. 8)."
    assert _not_retrieved(answer) == ["Apple 10-K, p. 8"]


def test_two_claims_citing_one_page_are_each_checked():
    answer = "One thing (Apple 10-K, p. 8). Another thing (Apple 10-K, p. 8)."
    assert len(citations.check_answer(answer, _pages())) == 2


def test_an_answer_with_no_citations_has_nothing_to_check():
    assert citations.check_answer("NVDA is at 200.", _pages()) == []


def test_a_source_can_still_be_checked_after_its_turn_leaves_the_history(monkeypatch, context):
    hit = search.SearchHit(0.9, APPLE_PAGE, "APPLE INC", "AAPL", "10-K", "2025", 50, 1, "", 48)
    monkeypatch.setattr(tools.search, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(store, "get_collection", lambda: None)
    _script_model(
        monkeypatch,
        [_reply(tool_calls=[("search_filing", {"ticker": "AAPL", "question": "q"})]), _reply("Ok")],
    )
    _, messages = _turn("Where?", context)
    messages.append({"role": "user", "content": "Next question"})
    messages[3]["content"] = "word " * 9000

    assert loop.trim_history(messages) == 1
    assert not any(message["role"] == "tool" for message in messages), "the page left the history"

    [evidence] = citations.check_answer(
        f"{APPLE_CLAIM} (Apple 10-K, p. 48).", context.retrieved_pages
    )
    assert evidence.status == citations.PASSAGE_FOUND


def test_pages_read_lists_this_turns_pages_once_and_skips_other_tools(monkeypatch, context):
    hit = search.SearchHit(0.9, APPLE_PAGE, "APPLE INC", "AAPL", "10-K", "2025", 50, 1, "", 48)
    monkeypatch.setattr(tools.search, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(store, "get_collection", lambda: None)
    context.retrieved_pages[("NVDA", 1)] = tools.RetrievedPage("From an earlier turn", "x", "")
    search_call = ("search_filing", {"ticker": "AAPL", "question": "q"})
    _script_model(
        monkeypatch,
        [
            _reply(tool_calls=[search_call, ("get_quote", {"ticker": "AAPL"}), search_call]),
            _reply("No citations here."),
        ],
    )
    result, _ = _turn("Where?", context)

    pages = citations.pages_read(result.steps, context.retrieved_pages)
    assert [page.cite for page in pages] == ["Apple 10-K, p. 48"]


def test_a_denied_search_is_not_a_page_read():
    step = loop.Step("search_filing", {}, "deny", "no", "hook", [], [], '{"status": "denied"}', 0.0)
    assert citations.pages_read([step], {}) == []


def test_the_filing_link_finds_the_passage_with_its_punctuation_encoded():
    link = citations.filing_link(EDGAR_URL, "China-based firms, and R&D")
    assert link == f"{EDGAR_URL}#:~:text=China%2Dbased%20firms%2C%20and%20R%26D"


def test_the_filing_link_uses_words_from_the_middle_of_a_long_passage():
    passage = "HEADING " + " ".join(str(number) for number in range(30))
    target = citations.filing_link(EDGAR_URL, passage).split("#:~:text=")[1]
    assert "HEADING" not in target
    assert len(target.split("%20")) == citations.LINK_WORDS


def test_without_a_passage_the_link_opens_the_filing():
    assert citations.filing_link(EDGAR_URL) == EDGAR_URL


@pytest.mark.parametrize("url", ["", "https://evil.example/filing.htm", "javascript:alert(1)"])
def test_only_an_edgar_url_becomes_a_link(url):
    assert citations.filing_link(url, "some passage") == ""


# --- Context usage ------------------------------------------------------------

_FIXED = context_usage.FixedCosts(system_prompt=200, tool_definitions=500)
_SYSTEM = {"role": "system", "content": "word " * 200}


def _logged_call(
    messages: list[dict], new: list[int], prompt: int, reply: int = 0
) -> loop.ModelCall:
    return loop.ModelCall(
        number=1,
        turn=1,
        request={"messages": messages, "options": {"num_ctx": 8192}},
        response={"prompt_eval_count": prompt, "eval_count": reply, "done_reason": "stop"},
        new_messages=new,
    )


def _user(tokens: int) -> dict:
    return {"role": "user", "content": "word " * tokens}


def _tool_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_portfolio", "arguments": {}}}],
        },
        {"role": "tool", "tool_name": "get_portfolio", "content": "word " * 1000},
    ]


def test_fixed_costs_come_from_subtracting_two_real_counts(monkeypatch):
    def fake_count(messages, tool_definitions):
        return 716 if tool_definitions else 227

    monkeypatch.setattr(context_usage, "_count_prompt_tokens", fake_count)
    fixed = context_usage.measure_fixed_costs(_SYSTEM)
    assert fixed == context_usage.FixedCosts(system_prompt=227, tool_definitions=489)


def test_before_any_call_the_bar_shows_the_measured_fixed_costs():
    usage = context_usage.usage_before_any_call([_SYSTEM], _FIXED)
    assert usage.measured
    assert usage.parts[context_usage.SYSTEM] == 200
    assert usage.parts[context_usage.TOOL_DEFINITIONS] == 500
    assert usage.used == 700


def test_without_ollama_the_bar_estimates_and_says_so():
    usage = context_usage.usage_before_any_call(loop.new_conversation(), None)
    assert not usage.measured
    # Measured with Ollama: system prompt and tool definitions took about 700 tokens.
    assert 500 < usage.used < 1000


def test_each_message_costs_what_the_prompt_grew_by_when_it_arrived():
    first = [_SYSTEM, _user(5)]
    second = [*first, _tool_turn()[0]]
    calls = [
        _logged_call(first, [0, 1], prompt=720),
        _logged_call(second, [2], prompt=750),
    ]
    usages = context_usage.usage_of_calls(calls, _FIXED)

    assert usages[0].parts[context_usage.USER] == 20
    assert usages[1].parts[context_usage.USER] == 20, "a resent message costs the same"
    assert usages[1].parts[context_usage.MODEL] == 30
    assert usages[1].used == 750


def test_the_parts_always_add_up_to_what_ollama_measured_plus_the_reply():
    first = [_SYSTEM, _user(5)]
    second = [*first, *_tool_turn()]
    calls = [
        _logged_call(first, [0, 1], prompt=720, reply=300),
        _logged_call(second, [2, 3], prompt=2500, reply=400),
    ]
    usages = context_usage.usage_of_calls(calls, _FIXED)

    assert usages[0].used == 1020
    assert usages[1].used == 2900
    assert usages[1].reply_tokens == 400
    assert usages[1].parts[context_usage.FREE] == 8192 - 2900


def test_messages_arriving_together_are_divided_by_estimate():
    first = [_SYSTEM, _user(5)]
    second = [*first, *_tool_turn()]
    calls = [
        _logged_call(first, [0, 1], prompt=720),
        _logged_call(second, [2, 3], prompt=2720),
    ]
    parts = context_usage.usage_of_calls(calls, _FIXED)[1].parts

    assert parts[context_usage.TOOL_RESULTS] + parts[context_usage.MODEL] == 2000
    assert parts[context_usage.TOOL_RESULTS] > 1900, "the tool call itself is only a few tokens"


def test_dropped_turns_take_their_measured_cost_with_them():
    old_turn = [_user(5), {"role": "assistant", "content": "word " * 100}]
    new_question = _user(7)
    long_history = [_SYSTEM, *old_turn]
    calls = [
        _logged_call(long_history[:2], [0, 1], prompt=720),
        _logged_call([*long_history, new_question], [2, 3], prompt=900),
        # The first turn was dropped, so this prompt is smaller than the last one.
        _logged_call([_SYSTEM, new_question, _tool_turn()[0]], [2], prompt=760),
    ]
    usages = context_usage.usage_of_calls(calls, _FIXED)

    assert usages[2].parts[context_usage.USER] < usages[1].parts[context_usage.USER]
    assert usages[2].used == 760


def test_the_real_loop_logs_what_the_measurement_needs(monkeypatch, context):
    first_reply = _reply(tool_calls=[("get_portfolio", {})])
    first_reply.prompt_eval_count = 720
    second_reply = _reply("Done.")
    second_reply.prompt_eval_count = 1500
    _script_model(monkeypatch, [first_reply, second_reply])

    result, _ = _turn("What is in it?", context)
    usages = context_usage.usage_of_calls(result.calls, _FIXED)

    assert usages[0].parts[context_usage.USER] == 20
    assert usages[1].parts[context_usage.TOOL_RESULTS] > 0
    assert usages[1].used == 1500 + result.calls[1].response["eval_count"]


def test_free_space_never_goes_negative():
    calls = [_logged_call([_SYSTEM, _user(5)], [0, 1], prompt=8000, reply=900)]
    assert context_usage.usage_of_calls(calls, _FIXED)[0].parts[context_usage.FREE] == 0


def test_a_call_with_no_count_makes_the_whole_bar_an_estimate():
    call = _logged_call([_SYSTEM, _user(5)], [0, 1], prompt=0)
    usage = context_usage.usage_of_calls([call], _FIXED)[0]
    assert not usage.measured
    assert usage.used > 0


def test_with_no_fixed_costs_calls_are_estimated():
    call = _logged_call([_SYSTEM, _user(5)], [0, 1], prompt=720)
    assert not context_usage.usage_of_calls([call], None)[0].measured


# --- The final prompt text ---


def test_render_prompt_sends_the_same_request_and_asks_only_for_the_text(monkeypatch):
    posted = {}

    def fake_post(url, json, timeout):
        posted.update(json)
        return httpx.Response(
            200,
            json={"_debug_info": {"rendered_template": "<prompt>"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    request = {"model": "m", "messages": [], "options": {"num_ctx": 8192}}

    assert REAL_RENDER_PROMPT(request) == "<prompt>"
    assert posted == {**request, "stream": False, "_debug_render_only": True}


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"message": {}}),
        httpx.Response(200, text="not json"),
        httpx.Response(500, json={"error": "no"}),
    ],
)
def test_render_prompt_gives_nothing_when_ollama_wont_say(monkeypatch, answer):
    answer.request = httpx.Request("POST", "http://test")
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: answer)
    assert REAL_RENDER_PROMPT({"model": "m"}) == ""


def test_render_prompt_gives_nothing_when_ollama_is_down(monkeypatch):
    def down(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", down)
    assert REAL_RENDER_PROMPT({"model": "m"}) == ""


def test_each_call_logs_the_prompt_rendered_from_what_it_sent(monkeypatch, context):
    monkeypatch.setattr(loop, "render_prompt", lambda request: f"{len(request['messages'])} sent")
    _script_model(monkeypatch, [_reply(tool_calls=[("get_portfolio", {})]), _reply("Done.")])
    messages = [*loop.new_conversation(), {"role": "user", "content": "Portfolio?"}]

    result = loop.run_turn(messages, context)

    assert [call.rendered_prompt for call in result.calls] == ["2 sent", "4 sent"]


# --- The "How it works" explanation ---------------------------------------------


def test_every_hook_has_an_explanation_and_nothing_explains_a_removed_hook():
    assert set(explain.HOOK_EXPLANATIONS) == {hook.__name__ for hook in hooks.PRE_TOOL_HOOKS}


def test_hook_rows_list_the_hooks_in_the_order_they_run():
    rows = explain.hook_rows()
    assert [row["Hook"] for row in rows] == [hook.__name__ for hook in hooks.PRE_TOOL_HOOKS]
    assert "Last tool call" not in rows[0], "no column before any tool call has happened"


def test_hook_rows_show_a_deny_and_that_the_hooks_after_it_never_ran(context):
    call = ToolCall("propose_trade", {"ticker": "TSLA", "side": "buy", "quantity": 10})
    verdict = run_pre_hooks(call, _state())
    checks = [
        {"hook": name, "decision": said.decision if said else None} for name, said in verdict.checks
    ]

    said = {row["Hook"]: row["Last tool call"] for row in explain.hook_rows(checks)}

    assert said["known_tool_hook"] == "no objection"
    assert said["allowlist_hook"] == "⛔ deny"
    assert said["approval_hook"] == "—", "the first deny ends the list"


def test_search_funnel_counts_skipped_pages_and_discarded_sentences():
    reviews = [
        quotes.PageReview("Apple 10-K, front pages", 0.75, "cover or contents, skipped"),
        quotes.PageReview("Apple 10-K, p. 9", 0.72, "kept 1", ["One two three"], [], "a b c d e f"),
        quotes.PageReview("Apple 10-K, p. 8", 0.71, "kept 1", ["Four"], ["Made up"], "g h"),
    ]

    funnel = explain.search_funnel(reviews)

    assert "Front pages skipped: 1" in funnel
    assert "pages read: 2" in funnel
    assert "sentences kept: 2" in funnel
    assert "discarded: 1" in funnel
    assert "One two three" not in funnel, "numbers only, so no filing text gets through"


def test_flowchart_is_fixed_text_with_every_edge_between_declared_nodes():
    dot = explain.flow_dot()
    declared = set(re.findall(r"^(\w+) \[label=", dot, flags=re.MULTILINE))
    edge_lines = [line for line in dot.splitlines() if "->" in line]
    used = {
        name.strip() for line in edge_lines for name in line.split("[")[0].rstrip(" ;").split("->")
    }
    assert used == declared
    assert dot.count("{") == dot.count("}")


def test_explanations_quote_the_limits_the_code_really_uses():
    assert str(settings.agent_max_turns) in explain.LOOP.text
    assert f"{loop.context_budget():,}" in explain.FIT_CONTEXT.text
    assert f"{tools.SEARCH_RESULTS} other pages" in explain.SHORT_RESULTS.text
    assert str(citations.MATCH_THRESHOLD) in explain.CHECK_ANSWER.text


def test_the_prompt_on_show_is_the_real_one():
    example = explain.relevance_prompt_example()
    assert example == quotes.review_prompt("The user's question", "The whole page goes here.")

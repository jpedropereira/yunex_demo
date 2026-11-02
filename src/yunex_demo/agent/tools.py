"""What the model can ask for: the schemas it reads, and the code that answers.

All arithmetic happens here and never in the model. Results are compact JSON strings.

Tickers are plain strings in the schemas on purpose. Given an enum, the model refused
"Buy 10 TSLA" in plain text: no tool call, so no guard, no audit row, nothing to show. A
schema only suggests; the allowlist hook is what enforces.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from yunex_demo.agent import portfolio, quotes
from yunex_demo.agent.market import PriceSnapshot
from yunex_demo.config import settings
from yunex_demo.sec import search, store

CURRENCY = "USD"

PERIOD_DAYS = {"1mo": 30, "3mo": 91, "6mo": 182, "1y": 365}

# Whole pages go to the relevance step, because the answer is often far down the page. Set
# here and not by the model: offered a choice, it asked for five. Of 17 known facts, 5 pages
# got 30 of 34 answers right, 3 pages 26 and 8 pages 31, at about 2.5 seconds more per step up.
SEARCH_RESULTS = 5

# The most chunks a filing's cover and contents can take of the results (Apple's take three).
# A cover ranks well for "where" questions because of its address, and helps none of them.
MOST_FRONT_MATTER_HITS = 3


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_TICKER = {"type": "string", "description": "Stock ticker, e.g. NVDA"}

TOOLS = [
    _tool(
        "search_filing",
        "Search one company's latest 10-K. Use for anything a 10-K would say. "
        "Call once per company.",
        {
            "ticker": _TICKER,
            "question": {"type": "string", "description": "The user's question, in their words."},
        },
        ["ticker", "question"],
    ),
    _tool(
        "get_quote",
        "Current price, day change and 52-week range for one ticker.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_price_history",
        "How one ticker's price changed over a period, with the % change.",
        {"ticker": _TICKER, "period": {"type": "string", "enum": list(PERIOD_DAYS)}},
        ["ticker", "period"],
    ),
    _tool(
        "get_portfolio",
        "Ana's portfolio: each position's shares, price, value and weight, the total value, "
        "and the largest buy allowed.",
        {},
        [],
    ),
    _tool(
        "propose_trade",
        "Propose a trade. A guard checks it and a human approves it. It never executes by itself.",
        {
            "ticker": _TICKER,
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "quantity": {
                "type": "integer",
                "description": "Number of shares, when the user gives a share count.",
            },
            "portfolio_pct": {
                "type": "number",
                "description": "Trade size as a percent of portfolio value, when the user gives "
                "a percentage. Code converts it to shares. Send this instead of quantity.",
            },
        },
        ["ticker", "side"],
    ),
]

TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


@dataclass
class RetrievedPage:
    """A whole page the model got sentences from. Its citations are checked against this."""

    cite: str
    text: str
    source_url: str


@dataclass
class AgentContext:
    """What the tools read. ``retrieved_pages`` lasts the whole conversation.

    It is keyed by (ticker, printed page), and kept apart from the message history so that
    a source can still be checked once its turn has been dropped from the context.
    """

    conn: sqlite3.Connection
    snapshot: PriceSnapshot | None
    retrieved_pages: dict[tuple[str, int], RetrievedPage] = field(default_factory=dict)
    # What the relevance step did since the agent loop last collected this.
    page_reviews: list[quotes.PageReview] = field(default_factory=list)


def trade_quantity(arguments: dict, total_value: Decimal, prices: dict[str, Decimal]) -> int:
    """The shares a trade asks for, converting a percentage in code, rounding down."""
    if arguments.get("quantity") is not None:
        return arguments["quantity"]
    target_value = Decimal(str(arguments["portfolio_pct"])) * total_value / 100
    return int(target_value // prices[arguments["ticker"]])


def _money(value: Decimal) -> str:
    return f"{value:.2f}"


def _percent_change(start: Decimal, end: Decimal) -> str:
    return _money((end - start) / start * 100)


def _no_prices() -> dict:
    return {"status": "no_prices", "reason": "Market data is unavailable right now."}


def company_name(ticker: str) -> str:
    return next(f.company for f in settings.agent_filings if f.ticker == ticker)


def _remember_page(hit: search.SearchHit, cite: str, context: AgentContext) -> None:
    key = (hit.ticker, hit.printed_page)
    page = context.retrieved_pages.get(key)
    if page is None:
        context.retrieved_pages[key] = RetrievedPage(cite, hit.text, hit.source_url)
    # A long page is stored in parts, and another search can return a different part of it.
    elif hit.text not in page.text:
        page.text += f" {hit.text}"


def search_filing(arguments: dict, context: AgentContext) -> dict:
    ticker = arguments["ticker"]
    # The model only chooses the company. This filter is what guarantees that the pages
    # come from that company, however similar another filing's text may be.
    question = arguments["question"]
    hits = search.search(
        question, SEARCH_RESULTS + MOST_FRONT_MATTER_HITS, store.get_collection(), ticker=ticker
    )
    if not hits:
        return {"status": "not_indexed", "reason": f"No 10-K is indexed for {ticker}."}

    company = company_name(ticker)
    results = []
    quotes_kept = []
    pages_reviewed = 0
    for hit in hits:
        if pages_reviewed == SEARCH_RESULTS:
            break
        if hit.page == 1 or hit.printed_page == 0:
            context.page_reviews.append(
                quotes.PageReview(
                    f"{company} 10-K, front pages", hit.similarity, "cover or contents, skipped"
                )
            )
            continue

        pages_reviewed += 1
        cite = f"{company} 10-K, p. {hit.printed_page}"
        review = quotes.review_page(question, hit.text, cite, hit.similarity)
        quotes.set_aside_repeats(review, quotes_kept)
        context.page_reviews.append(review)
        if review.kept:
            _remember_page(hit, cite, context)
            results.append({"cite": cite, "quotes": review.kept})
            quotes_kept.extend(review.kept)

    if not results:
        return {
            "status": "no_relevant_pages",
            "reason": f"No page found in {company}'s 10-K helps answer this question.",
        }
    return {"results": results}


def get_quote(arguments: dict, context: AgentContext) -> dict:
    if context.snapshot is None:
        return _no_prices()
    ticker = arguments["ticker"]
    closes = context.snapshot.closes[ticker]
    prices = [price for _, price in closes]
    return {
        "ticker": ticker,
        "price": _money(prices[-1]),
        "day_change_pct": _percent_change(prices[-2], prices[-1]) if len(prices) > 1 else "0.00",
        "low_52w": _money(min(prices)),
        "high_52w": _money(max(prices)),
        "currency": CURRENCY,
        "as_of": closes[-1][0],
        "cached": context.snapshot.from_cache,
    }


def get_price_history(arguments: dict, context: AgentContext) -> dict:
    if context.snapshot is None:
        return _no_prices()
    closes = context.snapshot.closes[arguments["ticker"]]
    end_date, end_price = closes[-1]

    cutoff = date.fromisoformat(end_date) - timedelta(days=PERIOD_DAYS[arguments["period"]])
    start_date, start_price = next(
        (day, price) for day, price in closes if date.fromisoformat(day) >= cutoff
    )
    return {
        "ticker": arguments["ticker"],
        "period": arguments["period"],
        "start_date": start_date,
        "start_price": _money(start_price),
        "end_date": end_date,
        "end_price": _money(end_price),
        "change_pct": _percent_change(start_price, end_price),
        "currency": CURRENCY,
    }


def get_portfolio(arguments: dict, context: AgentContext) -> dict:
    if context.snapshot is None:
        return _no_prices()
    positions = portfolio.read_positions(context.conn)
    prices = context.snapshot.latest_prices()
    total = portfolio.total_value(positions, prices)
    return {
        "name": portfolio.PORTFOLIO_NAME,
        "positions": [
            {
                "ticker": ticker,
                "shares": quantity,
                "price": _money(prices[ticker]),
                "value": _money(quantity * prices[ticker]),
                "weight_pct": _money(quantity * prices[ticker] / total * 100),
            }
            for ticker, quantity in positions.items()
        ],
        "total_value": _money(total),
        # Named for what it limits. As "max_trade_value", the model applied it to sells too.
        "max_buy_value": _money(portfolio.max_trade_value(total)),
        "currency": CURRENCY,
    }


# propose_trade isn't here on purpose: approval_hook always stops it, so no code path
# from the model can reach anything that changes a position.
READ_TOOLS = {
    "search_filing": search_filing,
    "get_quote": get_quote,
    "get_price_history": get_price_history,
    "get_portfolio": get_portfolio,
}


def run_tool(name: str, arguments: dict, context: AgentContext) -> str:
    # Not escaped: a curly quote costs the model about five tokens as "\\u2019" (measured).
    return json.dumps(READ_TOOLS[name](arguments, context), ensure_ascii=False)

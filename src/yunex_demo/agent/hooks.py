"""The guards: small functions that run in code before every tool call.

Modelled on Claude Code's PreToolUse hooks. A hook sees the call and the current state,
and returns a decision or None for no opinion. The model can ask for anything; only
what gets through this list ever happens.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from yunex_demo.agent import portfolio
from yunex_demo.agent.tools import PERIOD_DAYS, TOOL_NAMES, trade_quantity
from yunex_demo.config import settings

# How people refer to each company in a question, for ticker_mention_hook.
COMPANY_ALIASES = {
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "amazon": "AMZN",
    "amzn": "AMZN",
}
COMPANY_PATTERN = re.compile(rf"\b({'|'.join(COMPANY_ALIASES)})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class HookState:
    """The facts a decision can rest on: all from code, none from the model."""

    positions: dict[str, int]
    prices: dict[str, Decimal]
    # What the user has said in this conversation, oldest first.
    user_messages: tuple[str, ...] = ()

    @property
    def total_value(self) -> Decimal:
        return portfolio.total_value(self.positions, self.prices)


@dataclass(frozen=True)
class HookResult:
    decision: Literal["allow", "deny", "ask", "warn"]
    reason: str
    hook: str


def is_whole_number(value: object) -> bool:
    # bool is a subclass of int in Python, and "true" is not a quantity.
    return isinstance(value, int) and not isinstance(value, bool)


def is_valid_percent(value: object) -> bool:
    # Infinity and NaN fail too: every comparison with NaN is false.
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 < value <= 100


def _quantity(call: ToolCall, state: HookState) -> int:
    return trade_quantity(call.arguments, state.total_value, state.prices)


def known_tool_hook(call: ToolCall, state: HookState) -> HookResult | None:
    if call.name not in TOOL_NAMES:
        return HookResult("deny", f"{call.name} is not a known tool", "known_tool_hook")
    return None


def _argument_problem(call: ToolCall) -> str | None:
    """What is wrong with the shape of these arguments. Which ticker is allowed comes later."""
    arguments = call.arguments
    if call.name == "get_portfolio":
        return None

    if not isinstance(arguments.get("ticker"), str):
        return "ticker must be a ticker symbol"

    question = arguments.get("question")
    if call.name == "search_filing" and not (isinstance(question, str) and question.strip()):
        return "question must not be empty"

    # A string first: a list from the model can't be looked up in a dict, and would raise.
    period = arguments.get("period")
    if call.name == "get_price_history" and not (isinstance(period, str) and period in PERIOD_DAYS):
        return f"period must be one of {', '.join(PERIOD_DAYS)}"

    if call.name == "propose_trade":
        if arguments.get("side") not in ("buy", "sell"):
            return "side must be buy or sell"
        has_quantity = arguments.get("quantity") is not None
        has_percent = arguments.get("portfolio_pct") is not None
        if has_quantity == has_percent:
            return "send exactly one of quantity or portfolio_pct"
        if has_percent and not is_valid_percent(arguments["portfolio_pct"]):
            return "portfolio_pct must be above 0 and at most 100"
    return None


def arguments_hook(call: ToolCall, state: HookState) -> HookResult | None:
    problem = _argument_problem(call)
    return HookResult("deny", problem, "arguments_hook") if problem else None


def allowlist_hook(call: ToolCall, state: HookState) -> HookResult | None:
    ticker = call.arguments.get("ticker")
    if ticker is not None and ticker not in settings.agent_tickers:
        return HookResult("deny", f"{ticker} is not on the allowlist", "allowlist_hook")
    return None


def ticker_mention_hook(call: ToolCall, state: HookState) -> HookResult | None:
    """Warn, never deny: "and what about their risks?" names no company, and is fine."""
    if call.name != "search_filing":
        return None

    # The newest message that names a company is the one the question is about.
    for message in reversed(state.user_messages):
        mentioned = {COMPANY_ALIASES[word.lower()] for word in COMPANY_PATTERN.findall(message)}
        if mentioned:
            ticker = call.arguments["ticker"]
            if ticker in mentioned:
                return None
            names = ", ".join(sorted(mentioned))
            reason = f"searched {ticker}, but the user asked about {names}"
            return HookResult("warn", reason, "ticker_mention_hook")
    return None


def prices_hook(call: ToolCall, state: HookState) -> HookResult | None:
    if call.name == "propose_trade" and not state.prices:
        return HookResult("deny", "trading is disabled: no market data", "prices_hook")
    return None


def quantity_hook(call: ToolCall, state: HookState) -> HookResult | None:
    if call.name != "propose_trade":
        return None
    quantity = _quantity(call, state)
    if not is_whole_number(quantity) or quantity <= 0:
        reason = f"quantity must be a positive whole number of shares, not {quantity}"
        return HookResult("deny", reason, "quantity_hook")
    return None


def holdings_hook(call: ToolCall, state: HookState) -> HookResult | None:
    if call.name != "propose_trade" or call.arguments["side"] != "sell":
        return None
    ticker = call.arguments["ticker"]
    quantity = _quantity(call, state)
    held = state.positions.get(ticker, 0)
    if quantity > held:
        return HookResult(
            "deny", f"can't sell {quantity} {ticker}, holding {held}", "holdings_hook"
        )
    return None


def size_limit_hook(call: ToolCall, state: HookState) -> HookResult | None:
    """Buys only: a sell is already capped by what is held."""
    if call.name != "propose_trade" or call.arguments["side"] != "buy":
        return None
    value = _quantity(call, state) * state.prices[call.arguments["ticker"]]
    limit = portfolio.max_trade_value(state.total_value)
    if value > limit:
        reason = (
            f"trade value USD {value:,.2f} is above the {settings.trade_max_pct}% limit "
            f"of USD {limit:,.2f}"
        )
        return HookResult("deny", reason, "size_limit_hook")
    return None


def approval_hook(call: ToolCall, state: HookState) -> HookResult | None:
    if call.name == "propose_trade":
        return HookResult("ask", "every trade needs a human to approve it", "approval_hook")
    return None


# The order matters: each hook can rely on what the ones above it have checked.
PRE_TOOL_HOOKS = [
    known_tool_hook,
    arguments_hook,
    allowlist_hook,
    ticker_mention_hook,
    prices_hook,
    quantity_hook,
    holdings_hook,
    size_limit_hook,
    approval_hook,
]

# At approval the click is the approval, and everything else is checked again.
APPROVAL_HOOKS = [hook for hook in PRE_TOOL_HOOKS if hook is not approval_hook]


@dataclass(frozen=True)
class Verdict:
    decision: Literal["allow", "deny", "ask"]
    reason: str
    # Which hook decided. Empty when nothing objected.
    hook: str
    warnings: tuple[HookResult, ...]
    # Every hook that ran, by name, with what it said: None for no opinion.
    checks: tuple[tuple[str, HookResult | None], ...]


def run_pre_hooks(call: ToolCall, state: HookState, hooks: list = PRE_TOOL_HOOKS) -> Verdict:
    """The first deny wins. Otherwise any ask. Otherwise allow. A warn changes nothing."""
    checks = []
    ask = None
    for hook in hooks:
        result = hook(call, state)
        checks.append((hook.__name__, result))
        if result is None:
            continue
        if result.decision == "deny":
            return Verdict("deny", result.reason, result.hook, _warnings(checks), tuple(checks))
        if result.decision == "ask" and ask is None:
            ask = result

    if ask:
        return Verdict("ask", ask.reason, ask.hook, _warnings(checks), tuple(checks))
    return Verdict("allow", "", "", _warnings(checks), tuple(checks))


def _warnings(checks: list[tuple[str, HookResult | None]]) -> tuple[HookResult, ...]:
    return tuple(result for _, result in checks if result and result.decision == "warn")

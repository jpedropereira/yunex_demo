"""The human's decision. This is the only code that changes a position.

Things may have changed since the trade was proposed, so approving checks it again
against the positions and prices as they are at the moment of the click.
"""

import sqlite3
from dataclasses import dataclass

from yunex_demo.agent import portfolio
from yunex_demo.agent.hooks import APPROVAL_HOOKS, HookState, ToolCall, run_pre_hooks
from yunex_demo.agent.market import PriceSnapshot

# Added to the conversation so the model knows the outcome on its next turn. Marked,
# because it is the app speaking in the user's slot, and hooks must not read it as a question.
APPROVAL_NOTE_PREFIX = "[Approval note]"


@dataclass(frozen=True)
class ApprovalOutcome:
    # Empty when the click changed nothing, because the trade was already decided.
    note: str
    executed: bool = False


def _note(text: str) -> str:
    return f"{APPROVAL_NOTE_PREFIX} {text}"


def approve_trade(
    conn: sqlite3.Connection, trade_id: int, snapshot: PriceSnapshot | None
) -> ApprovalOutcome:
    trade = portfolio.read_trade(conn, trade_id)
    if trade is None or trade["status"] != "pending":
        return ApprovalOutcome(note="")

    prices = snapshot.latest_prices() if snapshot else {}
    state = HookState(positions=portfolio.read_positions(conn), prices=prices)
    call = ToolCall(
        "propose_trade",
        {"ticker": trade["ticker"], "side": trade["side"], "quantity": trade["quantity"]},
    )
    verdict = run_pre_hooks(call, state, hooks=APPROVAL_HOOKS)

    if verdict.decision == "deny":
        if not portfolio.close_trade(conn, trade_id, "denied", verdict.reason):
            return ApprovalOutcome(note="")
        return ApprovalOutcome(note=_note(f"Trade #{trade_id} denied: {verdict.reason}."))

    price = prices[trade["ticker"]]
    if not portfolio.execute_trade(conn, trade_id, price):
        return ApprovalOutcome(note="")
    verb = "bought" if trade["side"] == "buy" else "sold"
    return ApprovalOutcome(
        note=_note(
            f"Trade #{trade_id} approved and executed: "
            f"{verb} {trade['quantity']} {trade['ticker']} at USD {price:,.2f}."
        ),
        executed=True,
    )


def reject_trade(conn: sqlite3.Connection, trade_id: int) -> ApprovalOutcome:
    if not portfolio.close_trade(conn, trade_id, "rejected", "rejected by a human"):
        return ApprovalOutcome(note="")
    return ApprovalOutcome(note=_note(f"Trade #{trade_id} was rejected by the human."))

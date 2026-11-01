"""The rules a document must pass before it may change the database.

A schema proves an answer has the right shape, not that it is true. These checks
are plain code: they test whether the numbers add up, and whether each value
really appears in the document. Each one returns a Check, so the page can show
every rule that was applied, not only the ones that failed.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from yunex_demo.workflow.schemas import CapitalCall, TradeConfirmation

# Allows for rounding to the cent on the document itself.
TOLERANCE = Decimal("0.01")

# A number as a document writes it: "200", "42.50", "2,000,000.00".
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _amounts_match(first: Decimal, second: Decimal) -> bool:
    return abs(first - second) <= TOLERANCE


def _sum_check(name: str, label: str, computed: Decimal, stated_label: str, stated: Decimal):
    """Compare a computed amount with the one the document states."""
    if _amounts_match(computed, stated):
        return Check(name, True, f"{label} = {computed:,.2f}, as the document says")
    detail = f"{label} = {computed:,.2f}, but the document says {stated_label} {stated:,.2f}"
    return Check(name, False, detail)


def _date_order_check(name: str, earlier: date, later: date) -> Check:
    return Check(name, later > earlier, f"{earlier.isoformat()} then {later.isoformat()}")


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks in the PDF don't matter."""
    return " ".join(text.lower().split())


def _numbers_in(text: str) -> set[Decimal]:
    """Every number written in the text: "2,000,000.00" -> Decimal("2000000.00").

    Decimals compare by value, so the model's 42.5 matches the document's 42.50.
    """
    return {Decimal(match.replace(",", "")) for match in _NUMBER.findall(text)}


def _grounding_check(
    document_text: str,
    numbers: dict[str, Decimal | int],
    strings: dict[str, str],
    dates: dict[str, date],
) -> Check:
    """Confirm every extracted value appears in the document.

    This stops the model inventing a value that happens to add up. It proves a
    value is *somewhere* in the document, not that it is the right one: a ticker
    of "SOL" would pass, because it is part of "SOLR". The PDF reader's page
    marker also makes a 1 appear in every document.
    """
    text = _normalise(document_text)
    written_numbers = _numbers_in(document_text)
    missing = []

    for field, number in numbers.items():
        if Decimal(number) not in written_numbers:
            missing.append(field)

    for field, string in strings.items():
        # An empty string would count as "contained" in any text.
        if not string.strip() or _normalise(string) not in text:
            missing.append(field)

    for field, value in dates.items():
        written_out = _normalise(f"{value.day} {value:%B %Y}")  # 3 november 2026
        if written_out not in text and value.isoformat() not in text:
            missing.append(field)

    if missing:
        return Check("Grounding", False, f"not found in the document: {', '.join(missing)}")
    count = len(numbers) + len(strings) + len(dates)
    return Check("Grounding", True, f"all {count} values appear in the document")


def validate_trade(trade: TradeConfirmation, document_text: str) -> list[Check]:
    # A buyer pays the commission on top; a seller has it taken off the proceeds.
    if trade.side == "buy":
        net_check = _sum_check(
            "gross + commission = total",
            "gross + commission",
            trade.gross_amount + trade.commission,
            "total",
            trade.total_amount,
        )
    else:
        net_check = _sum_check(
            "gross - commission = total",
            "gross - commission",
            trade.gross_amount - trade.commission,
            "total",
            trade.total_amount,
        )

    return [
        _sum_check(
            "quantity x price = gross",
            f"{trade.quantity:,} x {trade.price:,.2f}",
            trade.quantity * trade.price,
            "gross",
            trade.gross_amount,
        ),
        net_check,
        _date_order_check(
            "settlement after trade date",
            trade.trade_date,
            trade.settlement_date,
        ),
        Check(
            "quantity and price above 0, commission not negative",
            trade.quantity > 0 and trade.price > 0 and trade.commission >= 0,
            f"quantity {trade.quantity:,}, price {trade.price:,.2f}, "
            f"commission {trade.commission:,.2f}",
        ),
        _grounding_check(
            document_text,
            numbers={
                "quantity": trade.quantity,
                "price": trade.price,
                "gross_amount": trade.gross_amount,
                "commission": trade.commission,
                "total_amount": trade.total_amount,
            },
            strings={
                "reference": trade.reference,
                "side": trade.side,
                "security_name": trade.security_name,
                "ticker": trade.ticker,
                "currency": trade.currency,
            },
            dates={
                "trade_date": trade.trade_date,
                "settlement_date": trade.settlement_date,
            },
        ),
    ]


def validate_capital_call(call: CapitalCall, document_text: str) -> list[Check]:
    return [
        _sum_check(
            "commitment x call % = amount due",
            f"{call.total_commitment:,.2f} x {call.call_pct}%",
            call.total_commitment * call.call_pct / 100,
            "amount due",
            call.amount_due,
        ),
        _sum_check(
            "called + due + unfunded = commitment",
            "called before + amount due + unfunded after",
            call.called_before + call.amount_due + call.unfunded_after,
            "commitment",
            call.total_commitment,
        ),
        _date_order_check(
            "due date after notice date",
            call.notice_date,
            call.due_date,
        ),
        _grounding_check(
            document_text,
            numbers={
                "call_number": call.call_number,
                "total_commitment": call.total_commitment,
                "call_pct": call.call_pct,
                "amount_due": call.amount_due,
                "called_before": call.called_before,
                "unfunded_after": call.unfunded_after,
            },
            strings={
                "fund_name": call.fund_name,
                "currency": call.currency,
            },
            dates={
                "notice_date": call.notice_date,
                "due_date": call.due_date,
            },
        ),
    ]

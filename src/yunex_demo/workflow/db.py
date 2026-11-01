"""The small database the workflow acts on, kept in memory and seeded fresh.

Only validated documents reach the load functions, and each load is idempotent:
pressing the button twice must not double-count a trade. Every query is
parameterised, because the values originally came from a document and a model.
"""

import sqlite3
from dataclasses import dataclass
from typing import Literal

from yunex_demo.workflow.schemas import CapitalCall, TradeConfirmation

# The demo inbox belongs to one portfolio, so trades don't need to name theirs.
# In a real system this would be extracted from the document and checked.
PORTFOLIO = "Balanced Growth"

SEED_POSITIONS = [
    {
        "portfolio": PORTFOLIO,
        "ticker": "SOLR",
        "security_name": "Solaris Parks Ltd",
        "quantity": 1000,
    },
]

_SCHEMA = """
DROP TABLE IF EXISTS positions;
DROP TABLE IF EXISTS loaded_trades;
DROP TABLE IF EXISTS cash_calls;
DROP TABLE IF EXISTS review_queue;

CREATE TABLE positions (
    portfolio TEXT NOT NULL,
    ticker TEXT NOT NULL,
    security_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    PRIMARY KEY (portfolio, ticker)
);

-- Which trades were already applied. A position is only a running total, so it
-- can't tell us this itself.
CREATE TABLE loaded_trades (
    reference TEXT PRIMARY KEY
);

-- Money is stored as TEXT: SQLite has no decimal type, and REAL would be a float.
CREATE TABLE cash_calls (
    fund_name TEXT NOT NULL,
    call_number INTEGER NOT NULL,
    amount_due TEXT NOT NULL,
    currency TEXT NOT NULL,
    due_date TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (fund_name, call_number)
);

CREATE TABLE review_queue (
    file_name TEXT PRIMARY KEY,
    doc_type TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""


@dataclass
class LoadResult:
    status: Literal["loaded", "already_loaded", "rejected"]
    detail: str


def seed(conn: sqlite3.Connection) -> None:
    """Rebuild every table, so the demo always starts from the same state."""
    conn.executescript(_SCHEMA)
    with conn:
        conn.executemany(
            "INSERT INTO positions VALUES (:portfolio, :ticker, :security_name, :quantity)",
            SEED_POSITIONS,
        )


def connect() -> sqlite3.Connection:
    """Open a new in-memory database, seeded and ready to use."""
    # Streamlit can run each rerun of a page on a different thread.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    seed(conn)
    return conn


def load_trade(conn: sqlite3.Connection, trade: TradeConfirmation) -> LoadResult:
    """Apply a validated trade to its position, once per trade reference."""
    # One transaction: the position and the record of the trade change together.
    with conn:
        already_loaded = conn.execute(
            "SELECT 1 FROM loaded_trades WHERE reference = ?", (trade.reference,)
        ).fetchone()
        if already_loaded:
            return LoadResult("already_loaded", f"trade {trade.reference} was already loaded")

        held = conn.execute(
            "SELECT quantity FROM positions WHERE portfolio = ? AND ticker = ?",
            (PORTFOLIO, trade.ticker),
        ).fetchone()
        held_quantity = held["quantity"] if held else 0

        if trade.side == "sell" and trade.quantity > held_quantity:
            return LoadResult(
                "rejected",
                f"selling {trade.quantity:,} {trade.ticker} but only {held_quantity:,} held",
            )

        change = trade.quantity if trade.side == "buy" else -trade.quantity
        conn.execute(
            """
            INSERT INTO positions (portfolio, ticker, security_name, quantity)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (portfolio, ticker) DO UPDATE SET quantity = quantity + excluded.quantity
            """,
            (PORTFOLIO, trade.ticker, trade.security_name, change),
        )
        conn.execute("INSERT INTO loaded_trades VALUES (?)", (trade.reference,))

    return LoadResult(
        "loaded", f"{trade.ticker}: {held_quantity:,} -> {held_quantity + change:,} shares"
    )


def load_capital_call(conn: sqlite3.Connection, call: CapitalCall) -> LoadResult:
    """Record a validated capital call as pending, once per fund and call number."""
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO cash_calls VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                call.fund_name,
                call.call_number,
                f"{call.amount_due:.2f}",
                call.currency,
                call.due_date.isoformat(),
            ),
        )

    if cursor.rowcount == 0:
        return LoadResult("already_loaded", f"call no. {call.call_number} was already loaded")
    return LoadResult(
        "loaded", f"pending cash call: {call.currency} {call.amount_due:,.2f} due {call.due_date}"
    )


def add_to_review(conn: sqlite3.Connection, file_name: str, doc_type: str, reason: str) -> None:
    """Queue a document for a person to look at. Re-running a file replaces its entry."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO review_queue VALUES (?, ?, ?)",
            (file_name, doc_type, reason),
        )


def read_positions(conn: sqlite3.Connection) -> list[dict]:
    """Current positions, each with the quantity it was seeded with (0 if it is new)."""
    seeded = {position["ticker"]: position["quantity"] for position in SEED_POSITIONS}
    rows = conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
    return [
        {
            "portfolio": row["portfolio"],
            "ticker": row["ticker"],
            "security_name": row["security_name"],
            "before": seeded.get(row["ticker"], 0),
            "after": row["quantity"],
        }
        for row in rows
    ]


def read_cash_calls(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM cash_calls ORDER BY fund_name, call_number").fetchall()
    return [dict(row) for row in rows]


def read_review_queue(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM review_queue ORDER BY file_name").fetchall()
    return [dict(row) for row in rows]

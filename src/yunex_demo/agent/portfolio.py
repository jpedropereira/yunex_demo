"""The demo portfolio and its audit trail: a small SQLite database for this page only.

Every query is parameterised, because tickers and quantities originally came from a
model. Trades that were denied are stored too: that is what makes it an audit trail.
"""

import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from yunex_demo.config import settings

PORTFOLIO_NAME = "Ana — Tech Growth"

SEED_POSITIONS = {"NVDA": 120, "AAPL": 150, "MSFT": 80, "GOOGL": 140, "AMZN": 160}

# Money is stored as TEXT: SQLite has no decimal type, and REAL would be a float. Quantity,
# price and value are empty on a denied trade that never had a usable one.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER,
    price TEXT,
    value TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'executed', 'rejected', 'denied')),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def reset(conn: sqlite3.Connection) -> None:
    """Rebuild every table, so trade numbers start from 1 again too."""
    conn.executescript("DROP TABLE IF EXISTS positions; DROP TABLE IF EXISTS trades;" + _SCHEMA)
    with conn:
        conn.executemany("INSERT INTO positions VALUES (?, ?)", SEED_POSITIONS.items())


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database, creating and seeding it on first use."""
    path = path or settings.agent_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit can run each rerun of a page on a different thread.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    if not read_positions(conn):
        reset(conn)
    return conn


def read_positions(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT ticker, quantity FROM positions").fetchall()
    return {row["ticker"]: row["quantity"] for row in rows}


def total_value(positions: dict[str, int], prices: dict[str, Decimal]) -> Decimal:
    if not prices:
        return Decimal(0)
    return sum((quantity * prices[ticker] for ticker, quantity in positions.items()), Decimal(0))


def max_trade_value(total: Decimal) -> Decimal:
    return total * settings.trade_max_pct / 100


def log_trade(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    side: str,
    quantity: int | None,
    price: Decimal | None,
    status: str,
    reason: str,
) -> int:
    """Record a request, whatever became of it, and return its trade id."""
    value = quantity * price if quantity is not None and price is not None else None
    with conn:
        cursor = conn.execute(
            "INSERT INTO trades (ticker, side, quantity, price, value, status, reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                side,
                quantity,
                None if price is None else str(price),
                None if value is None else str(value),
                status,
                reason,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
    return cursor.lastrowid


def read_trade(conn: sqlite3.Connection, trade_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()


def read_trades(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every trade, newest first."""
    return conn.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()


def read_pending_trades(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM trades WHERE status = 'pending' ORDER BY id").fetchall()


def close_trade(conn: sqlite3.Connection, trade_id: int, status: str, reason: str) -> bool:
    """Move a pending trade to a final status. False if it already had one."""
    with conn:
        cursor = conn.execute(
            "UPDATE trades SET status = ?, reason = ? WHERE id = ? AND status = 'pending'",
            (status, reason, trade_id),
        )
    return cursor.rowcount == 1


def execute_trade(conn: sqlite3.Connection, trade_id: int, price: Decimal) -> bool:
    """Apply a pending trade to its position, once. False if it was no longer pending."""
    trade = read_trade(conn, trade_id)
    # One transaction: the position and the trade's status change together, and the
    # WHERE clause is what makes a second click do nothing.
    with conn:
        cursor = conn.execute(
            "UPDATE trades SET status = 'executed', reason = 'approved', price = ?, value = ?"
            " WHERE id = ? AND status = 'pending'",
            (str(price), str(trade["quantity"] * price), trade_id),
        )
        if cursor.rowcount != 1:
            return False

        change = trade["quantity"] if trade["side"] == "buy" else -trade["quantity"]
        conn.execute(
            "UPDATE positions SET quantity = quantity + ? WHERE ticker = ?",
            (change, trade["ticker"]),
        )
    return True

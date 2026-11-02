"""Market data, fetched once in a batch and then read from memory.

Every tool and guard reads the same snapshot, so a guard's decision can always be
explained with the numbers on screen. Yahoo Finance is unofficial and can fail or
rate-limit, so each good response is saved to disk as the fallback for a bad one, and
no chat turn ever depends on it.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import yfinance

from yunex_demo.config import settings

logger = logging.getLogger(__name__)

HISTORY_PERIOD = "1y"


@dataclass(frozen=True)
class PriceSnapshot:
    # Per ticker, the daily closes oldest first, as (ISO date, price).
    closes: dict[str, list[tuple[str, Decimal]]]
    fetched_at: datetime
    from_cache: bool = False

    def latest_prices(self) -> dict[str, Decimal]:
        return {ticker: closes[-1][1] for ticker, closes in self.closes.items()}


def _download(tickers: list[str]) -> dict[str, list[list[str]]]:
    """One batched call for all tickers, as JSON-ready [date, price] pairs."""
    with warnings.catch_warnings():
        # yfinance's use of pandas prints deprecation warnings that are no use to us.
        warnings.simplefilter("ignore")
        frame = yfinance.download(
            tickers,
            period=HISTORY_PERIOD,
            interval="1d",
            # Unadjusted, so prices match what a quote screen shows.
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    closes = {}
    for ticker in tickers:
        # yfinance reports a failed ticker with an empty column rather than an exception.
        column = frame["Close"][ticker].dropna() if not frame.empty else []
        if len(column) == 0:
            raise ValueError(f"No prices returned for {ticker}.")
        closes[ticker] = [[day.date().isoformat(), f"{price:.2f}"] for day, price in column.items()]
    return closes


def _build_snapshot(
    closes: dict[str, list[list[str]]], fetched_at: datetime, from_cache: bool
) -> PriceSnapshot:
    return PriceSnapshot(
        closes={
            ticker: [(day, Decimal(price)) for day, price in rows]
            for ticker, rows in closes.items()
        },
        fetched_at=fetched_at,
        from_cache=from_cache,
    )


def _read_cache() -> PriceSnapshot | None:
    try:
        cached = json.loads(settings.price_cache_path.read_text())
        return _build_snapshot(
            cached["closes"], datetime.fromisoformat(cached["fetched_at"]), from_cache=True
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("No usable price cache: %s", exc)
        return None


def load_snapshot() -> PriceSnapshot | None:
    """Fresh prices if Yahoo answers, else the last good ones, else None."""
    # Deliberately broad: yfinance raises many types, and any failure means the same thing.
    try:
        closes = _download(settings.agent_tickers)
    except Exception as exc:
        logger.warning("Could not fetch prices: %s", exc)
        return _read_cache()

    fetched_at = datetime.now().astimezone()
    settings.price_cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.price_cache_path.write_text(
        json.dumps({"fetched_at": fetched_at.isoformat(), "closes": closes})
    )
    return _build_snapshot(closes, fetched_at, from_cache=False)

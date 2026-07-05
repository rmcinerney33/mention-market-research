"""Liquidity / slippage modeling.

To buy ``size`` contracts we consume the ask side of the book from the best
price up. The realized average price (VWAP) is worse than the top of book once
the order walks into deeper levels — that difference is slippage, and it eats
directly into edge, so we model it explicitly.

Two modes:

- :func:`walk_book` — the real thing, given order-book levels. Requires the
  authenticated Kalshi order-book endpoint (see README); used when credentials
  are configured.
- :func:`fallback_fill` — when only top-of-book is available (Phase-1 snapshots,
  no auth), we can't see depth. We add a configurable buffer to the top price
  and mark ``book_available=False`` so the caller can refuse to flag on unknown
  liquidity rather than pretend the fill is clean.

Prices are Kalshi cents on input, dollars on output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BookLevel:
    price: int  # cents
    size: int   # contracts available at this level


@dataclass(frozen=True)
class FillResult:
    avg_price: float      # VWAP, dollars per contract
    filled: int           # contracts fillable
    requested: int
    fully_filled: bool
    top_price: float      # best level, dollars
    slippage: float       # avg_price - top_price, dollars (>= 0)
    book_available: bool


def walk_book(levels: list[BookLevel], size: int) -> FillResult:
    """Walk ``levels`` (best price first) to fill ``size`` contracts."""
    if not levels or size <= 0:
        raise ValueError("walk_book needs at least one level and size > 0")
    remaining = size
    cost_cents = 0
    filled = 0
    for lvl in levels:
        take = min(remaining, lvl.size)
        cost_cents += take * lvl.price
        filled += take
        remaining -= take
        if remaining == 0:
            break
    avg = (cost_cents / filled) / 100.0
    top = levels[0].price / 100.0
    return FillResult(
        avg_price=avg,
        filled=filled,
        requested=size,
        fully_filled=remaining == 0,
        top_price=top,
        slippage=avg - top,
        book_available=True,
    )


def fallback_fill(top_ask_cents: int, size: int, buffer_cents: float = 1.0) -> FillResult:
    """No depth visible: assume the top price plus a conservative buffer.

    ``book_available=False`` signals to the caller that fill quality is a guess,
    not a measurement — flagging policy may require a real book (see EdgeConfig).
    """
    top = top_ask_cents / 100.0
    avg = (top_ask_cents + buffer_cents) / 100.0
    return FillResult(
        avg_price=avg,
        filled=size,
        requested=size,
        fully_filled=True,
        top_price=top,
        slippage=avg - top,
        book_available=False,
    )

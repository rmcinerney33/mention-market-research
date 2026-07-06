"""Flags — the permanent, auditable record of what the system surfaced.

A :class:`Flag` is a post-portfolio decision to show a human an opportunity. It
captures *everything* needed to reproduce or audit it later: the model version
and hash, the exact features fed to the model, the market price/book state at
flag time, the chosen side, the fee/slippage-adjusted EV, and the suggested
size. Flags are append-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .portfolio import Allocation
from .signal import Signal


@dataclass
class Flag:
    flagged_at: datetime
    scan_id: int | None
    ticker: str
    event_ticker: str | None
    category: str
    side: str | None
    model_prob: float | None
    ci_lo: float | None
    ci_hi: float | None
    market_price: float | None
    raw_edge: float | None
    ev_per_contract: float | None
    contracts: int
    notional: float
    slippage: float
    book_available: bool
    model_version: str | None
    model_hash: str | None
    features_json: str | None
    event_time: datetime | None
    reason: str


def build_flags(
    allocations: list[Allocation],
    signals: list[Signal],
    *,
    flagged_at: datetime,
    scan_id: int | None = None,
) -> list[Flag]:
    """Assemble Flag records for allocations that survived portfolio sizing.

    Only positions with at least one contract after the exposure caps become
    flags — a flaggable edge scaled to zero by the budget is not surfaced.
    """
    by_ticker = {s.ticker: s for s in signals}
    flags: list[Flag] = []
    for a in allocations:
        if a.reason == "not_flaggable" or a.contracts < 1:
            continue
        e = a.edge
        s = by_ticker.get(e.ticker)
        flags.append(
            Flag(
                flagged_at=flagged_at,
                scan_id=scan_id,
                ticker=e.ticker,
                event_ticker=e.event_ticker,
                category=e.category,
                side=e.side,
                model_prob=e.model_prob,
                ci_lo=e.ci_lo,
                ci_hi=e.ci_hi,
                market_price=e.market_price,
                raw_edge=e.raw_edge,
                ev_per_contract=e.ev_per_contract,
                contracts=a.contracts,
                notional=a.notional,
                slippage=e.slippage,
                book_available=e.book_available,
                model_version=s.model_version if s else None,
                model_hash=s.model_hash if s else None,
                features_json=s.features_json if s else None,
                event_time=s.event_time if s else None,
                reason="flaggable",
            )
        )
    return flags

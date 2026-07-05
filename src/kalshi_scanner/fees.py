"""Kalshi trading fees.

Kalshi's published general trading-fee formula (fees charged on execution):

    fee = roundup( coefficient * C * P * (1 - P) )   # rounded UP to the cent

where ``C`` is the number of contracts and ``P`` is the per-contract price in
dollars (0 < P < 1). The default coefficient is 0.07.

IMPORTANT: some series carry a different coefficient, and Kalshi can revise the
schedule. The coefficient is therefore configurable and **must be re-confirmed
against the current published schedule before any real-money use** (the docs
fetch was rate-limited when this was written). We never approximate fees with a
flat percentage — the price-dependent ``P*(1-P)`` term matters, especially near
the middle of the book where mention markets often trade.

All money is handled with :class:`decimal.Decimal` so the round-up-to-cent is
exact, not subject to binary-float drift.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

DEFAULT_FEE_COEFFICIENT = Decimal("0.07")
_CENT = Decimal("0.01")


def _dec(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def trading_fee(contracts: int, price, coefficient=DEFAULT_FEE_COEFFICIENT) -> Decimal:
    """Total trading fee in dollars, rounded up to the next cent.

    ``price`` is the per-contract price in dollars, in [0, 1].
    """
    if contracts < 0:
        raise ValueError("contracts must be non-negative")
    p = _dec(price)
    if not (Decimal(0) <= p <= Decimal(1)):
        raise ValueError(f"price must be in [0, 1] dollars; got {price}")
    raw = _dec(coefficient) * Decimal(int(contracts)) * p * (Decimal(1) - p)
    return raw.quantize(_CENT, rounding=ROUND_CEILING)


def fee_rate_per_contract(price, coefficient=DEFAULT_FEE_COEFFICIENT) -> float:
    """Unrounded per-contract fee rate ``coef * P * (1-P)`` in dollars.

    Used for expected-value and Kelly math (the actual charged fee rounds the
    *whole* trade, so per-contract EV uses this smooth rate rather than the
    rounded total divided by C).
    """
    p = _dec(price)
    return float(_dec(coefficient) * p * (Decimal(1) - p))

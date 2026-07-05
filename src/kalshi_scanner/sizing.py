"""Fractional-Kelly position sizing.

We buy a binary contract at an all-in cost ``R`` dollars per contract (executable
price + per-contract fee) that pays $1 if the event resolves in our favor. With
win probability ``p``:

    win  (prob p):     net +$(1 - R)   per contract
    lose (prob 1-p):   net -$R         per contract (the whole outlay)

The Kelly fraction of bankroll to *risk* on such a bet is

    f* = p - (1 - p) * R / (1 - R)          (clipped at 0; 0 means "no bet")

We then scale by a configurable Kelly fraction (default 0.25 — quarter Kelly)
and hard-cap the risked fraction at ``max_position_frac`` of bankroll (default
2%). Contracts are floored to a whole number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizeResult:
    kelly_full: float          # f*, the full-Kelly risked fraction
    kelly_fraction: float      # multiplier applied (e.g. 0.25)
    position_fraction: float   # risked fraction after fraction + cap
    contracts: int
    notional: float            # contracts * cost_per_contract (dollars at risk)
    capped: bool               # True if the hard cap bound the size


def kelly_fraction_binary(p: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll to risk. 0 if there's no positive edge."""
    if not (0.0 < cost < 1.0):
        return 0.0
    f = p - (1.0 - p) * cost / (1.0 - cost)
    return max(0.0, f)


def size_position(
    p: float,
    cost: float,
    bankroll: float,
    *,
    kelly_fraction: float = 0.25,
    max_position_frac: float = 0.02,
) -> SizeResult:
    """Suggested contracts for a bet with win prob ``p`` at all-in ``cost``."""
    f_full = kelly_fraction_binary(p, cost)
    desired = kelly_fraction * f_full
    capped = desired > max_position_frac
    frac = min(desired, max_position_frac)
    alloc = frac * bankroll
    # +epsilon so float drift doesn't shave a whole contract (1999.9999 -> 2000).
    contracts = int(alloc / cost + 1e-9) if cost > 0 else 0
    return SizeResult(
        kelly_full=f_full,
        kelly_fraction=kelly_fraction,
        position_fraction=frac,
        contracts=contracts,
        notional=contracts * cost,
        capped=capped and contracts > 0,
    )

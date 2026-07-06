"""Correlation grouping and portfolio exposure caps (Phase 4).

Per-market Kelly (Phase 3) treats every flag as an independent bet. It isn't:
several mention markets on the *same speech* are highly correlated — if the
speaker is "on message" many fire together. Sizing them independently silently
over-bets that shared risk.

We handle this pragmatically (we have no reliable covariance for thin binary
markets): markets are grouped by their Kalshi ``event_ticker`` (the correlation
key — one speech/occasion), and a group shares a single risk budget. Then we
enforce a hierarchy of caps as fractions of bankroll — per event, per category,
overall — scaling positions down proportionally where a cap binds, and
subtracting any exposure already committed by open positions.

None of this places orders; it only decides suggested sizes for a human.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import TradingConfig
from .edge import EdgeResult


def correlation_key(edge: EdgeResult) -> str:
    """Markets sharing this key are treated as correlated (same event/speech)."""
    return edge.event_ticker or edge.ticker


@dataclass
class Exposure:
    """Risk already committed (e.g. by open paper/live positions), as fractions
    of bankroll."""
    by_event: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)
    total: float = 0.0


@dataclass
class Allocation:
    edge: EdgeResult
    original_fraction: float
    allocated_fraction: float
    contracts: int
    notional: float
    scaled: bool
    reason: str  # "ok" | "event_cap" | "category_cap" | "total_cap" | "not_flaggable"


class PortfolioAllocator:
    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    def allocate(
        self, edges: list[EdgeResult], existing: Exposure | None = None
    ) -> list[Allocation]:
        c = self.config
        existing = existing or Exposure()
        bankroll = c.bankroll

        candidates = [e for e in edges if e.flaggable]
        # Risked fraction each position wants (its notional as a share of bankroll).
        frac = {id(e): (e.notional / bankroll if bankroll else 0.0) for e in candidates}
        reason = {id(e): "ok" for e in candidates}

        def scale_group(members: list[EdgeResult], cap_remaining: float, label: str) -> None:
            total = sum(frac[id(e)] for e in members)
            if total <= cap_remaining or total <= 0:
                return
            factor = max(cap_remaining, 0.0) / total
            for e in members:
                frac[id(e)] *= factor
                reason[id(e)] = label

        # 1) Per-event cap (correlated group shares one budget).
        groups: dict[str, list[EdgeResult]] = defaultdict(list)
        for e in candidates:
            groups[correlation_key(e)].append(e)
        for key, members in groups.items():
            scale_group(members, c.max_event_frac - existing.by_event.get(key, 0.0), "event_cap")

        # 2) Per-category cap.
        cats: dict[str, list[EdgeResult]] = defaultdict(list)
        for e in candidates:
            cats[e.category].append(e)
        for cat, members in cats.items():
            scale_group(members, c.max_category_frac - existing.by_category.get(cat, 0.0),
                        "category_cap")

        # 3) Overall cap.
        scale_group(candidates, c.max_total_frac - existing.total, "total_cap")

        allocations: list[Allocation] = []
        for e in edges:
            if not e.flaggable:
                allocations.append(Allocation(e, 0.0, 0.0, 0, 0.0, scaled=False,
                                              reason="not_flaggable"))
                continue
            orig = e.notional / bankroll if bankroll else 0.0
            alloc_frac = frac[id(e)]
            cost = e.effective_cost or 0.0
            contracts = int(alloc_frac * bankroll / cost + 1e-9) if cost > 0 else 0
            allocations.append(
                Allocation(
                    edge=e,
                    original_fraction=orig,
                    allocated_fraction=alloc_frac,
                    contracts=contracts,
                    notional=contracts * cost,
                    scaled=alloc_frac < orig - 1e-12,
                    reason=reason[id(e)],
                )
            )
        return allocations

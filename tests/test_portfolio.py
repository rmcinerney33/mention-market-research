"""Tests for correlation grouping and portfolio exposure caps."""

from __future__ import annotations

import pytest

from kalshi_scanner.config import TradingConfig
from kalshi_scanner.edge import EdgeResult
from kalshi_scanner.portfolio import Exposure, PortfolioAllocator, correlation_key

BANKROLL = 10_000.0


def _edge(ticker, event, notional, *, category="mention", cost=0.40, flaggable=True):
    return EdgeResult(
        scan_id=1, ticker=ticker, event_ticker=event, category=category, validated=True,
        side="YES", model_prob=0.8, ci_lo=0.6, ci_hi=0.95, market_price=cost, raw_edge=0.4,
        fee_per_contract=0.0, effective_cost=cost, ev_per_contract=0.4,
        kelly_fraction=notional / BANKROLL, contracts=int(notional / cost),
        notional=notional, slippage=0.0, book_available=True, gate_pass=True,
        flaggable=flaggable, reason="flaggable" if flaggable else "no_orderbook_depth",
    )


def _cfg(**kw):
    base = dict(bankroll=BANKROLL, max_event_frac=1.0, max_category_frac=1.0, max_total_frac=1.0)
    base.update(kw)
    return TradingConfig(**base)


def test_correlation_key_groups_by_event():
    assert correlation_key(_edge("A-Y", "EVT1", 300)) == "EVT1"
    assert correlation_key(_edge("A-N", "EVT1", 300)) == correlation_key(_edge("A-Y", "EVT1", 300))
    assert correlation_key(_edge("B-Y", "EVT2", 300)) == "EVT2"


def test_correlated_group_shares_event_budget():
    # Two markets on the SAME event each want 3% -> 6% > the 5% per-event cap.
    alloc = PortfolioAllocator(_cfg(max_event_frac=0.05))
    out = alloc.allocate([_edge("A", "EVT1", 300), _edge("B", "EVT1", 300)])
    assert sum(a.allocated_fraction for a in out) == pytest.approx(0.05)
    assert all(a.reason == "event_cap" and a.scaled for a in out)
    assert out[0].allocated_fraction == pytest.approx(0.025)  # split proportionally


def test_within_caps_is_unchanged():
    alloc = PortfolioAllocator(_cfg(max_event_frac=0.05, max_category_frac=0.10, max_total_frac=0.25))
    out = alloc.allocate([_edge("A", "EVT1", 300), _edge("B", "EVT2", 300)])
    assert all(not a.scaled and a.reason == "ok" for a in out)
    assert out[0].allocated_fraction == pytest.approx(0.03)


def test_category_cap_binds():
    # Four distinct events (event cap not binding), same category -> 12% > 10%.
    alloc = PortfolioAllocator(_cfg(max_event_frac=0.05, max_category_frac=0.10))
    edges = [_edge(f"T{i}", f"EVT{i}", 300) for i in range(4)]
    out = alloc.allocate(edges)
    assert sum(a.allocated_fraction for a in out) == pytest.approx(0.10)
    assert all(a.reason == "category_cap" for a in out)


def test_total_cap_binds():
    # 6 edges, distinct events, 3 categories x 2 -> event 4.5% (ok), cat 9% (ok), total 27% > 25%.
    alloc = PortfolioAllocator(_cfg(max_event_frac=0.05, max_category_frac=0.10, max_total_frac=0.25))
    edges = [_edge(f"T{i}", f"EVT{i}", 450, category=f"cat{i // 2}") for i in range(6)]
    out = alloc.allocate(edges)
    assert sum(a.allocated_fraction for a in out) == pytest.approx(0.25)
    assert all(a.reason == "total_cap" for a in out)


def test_existing_exposure_reduces_event_budget():
    alloc = PortfolioAllocator(_cfg(max_event_frac=0.05))
    existing = Exposure(by_event={"EVT1": 0.04})  # only 1% of the event budget left
    out = alloc.allocate([_edge("A", "EVT1", 300)], existing=existing)
    assert out[0].allocated_fraction == pytest.approx(0.01)
    assert out[0].reason == "event_cap"


def test_non_flaggable_edges_get_zero():
    alloc = PortfolioAllocator(_cfg())
    out = alloc.allocate([_edge("A", "EVT1", 300, flaggable=False)])
    assert out[0].contracts == 0
    assert out[0].allocated_fraction == 0.0
    assert out[0].reason == "not_flaggable"

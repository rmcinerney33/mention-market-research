"""Tests for flag assembly, persistence, and the HTML dashboard."""

from __future__ import annotations

from datetime import UTC, datetime

from kalshi_scanner.dashboard import render_dashboard
from kalshi_scanner.edge import EdgeResult
from kalshi_scanner.flags import build_flags
from kalshi_scanner.portfolio import Allocation
from kalshi_scanner.signal import Signal
from kalshi_scanner.store import SnapshotStore

TS = datetime(2024, 11, 1, tzinfo=UTC)
RESOLVE = datetime(2024, 11, 6, tzinfo=UTC)


def _edge(ticker, *, flaggable=True, gate_pass=True, ev=0.3, reason="flaggable"):
    return EdgeResult(
        scan_id=1, ticker=ticker, event_ticker="EVT", category="mention", validated=True,
        side="YES", model_prob=0.8, ci_lo=0.6, ci_hi=0.95, market_price=0.40, raw_edge=0.4,
        fee_per_contract=0.01, effective_cost=0.42, ev_per_contract=ev, kelly_fraction=0.02,
        contracts=100, notional=42.0, slippage=0.0, book_available=True,
        gate_pass=gate_pass, flaggable=flaggable, reason=reason,
    )


def _signal(ticker):
    return Signal(
        scan_id=1, ticker=ticker, event_ticker="EVT", scan_ts=TS, category="mention",
        validated=True, event_time=RESOLVE, market_yes_bid=None, market_yes_ask=40,
        market_no_bid=None, market_no_ask=62, market_implied_prob=0.4, model_prob=0.8,
        ci_lo=0.6, ci_hi=0.95, model_version="gbdt@abc", model_hash="abc",
        features_json='{"rr_hl30": 0.5}', reason=None,
    )


def _alloc(edge, contracts, reason="ok"):
    return Allocation(edge, 0.02, 0.02, contracts, contracts * 0.42, scaled=False, reason=reason)


def test_build_flags_only_from_sized_flaggable():
    e_ok = _edge("A")
    e_scaled_out = _edge("B")
    e_not = _edge("C", flaggable=False, reason="no_orderbook_depth")
    allocs = [
        _alloc(e_ok, 100),
        _alloc(e_scaled_out, 0),          # scaled to zero by budget -> not a flag
        Allocation(e_not, 0.0, 0.0, 0, 0.0, scaled=False, reason="not_flaggable"),
    ]
    flags = build_flags(allocs, [_signal("A"), _signal("B"), _signal("C")],
                        flagged_at=TS, scan_id=1)
    assert [f.ticker for f in flags] == ["A"]
    assert flags[0].model_version == "gbdt@abc"
    assert flags[0].features_json == '{"rr_hl30": 0.5}'
    assert flags[0].contracts == 100


def test_flags_store_round_trip(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    flags = build_flags([_alloc(_edge("A"), 100)], [_signal("A")], flagged_at=TS, scan_id=1)
    store.record_flags(flags)
    assert store.count_flags() == 1
    row = store.flags_for_scan(1)[0]
    assert row["ticker"] == "A"
    assert row["model_version"] == "gbdt@abc"
    assert row["reason"] == "flaggable"


def test_dashboard_shows_flag_and_hides_non_gated():
    edges = [
        _edge("FLAGME"),                                  # flaggable, gate passed
        _edge("NEARMISS", flaggable=False, reason="no_orderbook_depth"),  # gate passed, not flaggable
        _edge("NOISE", flaggable=False, gate_pass=False, reason="inside_ci_band"),  # excluded
    ]
    signals = [_signal("FLAGME"), _signal("NEARMISS"), _signal("NOISE")]
    html = render_dashboard(edges, signals, scan_ts=TS, validated_categories=[])

    assert "1 flaggable" in html
    assert "FLAGME" in html and ">FLAG<" in html
    assert "NEARMISS" in html and "no_orderbook_depth" in html
    assert "NOISE" not in html               # did not clear the CI gate
    assert "5d 0h" in html                    # time-to-resolution from TS -> RESOLVE

"""Tests for the edge calculator: side selection, the CI gate, and flaggability."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kalshi_scanner.config import TradingConfig
from kalshi_scanner.edge import EdgeEvaluator
from kalshi_scanner.signal import Signal
from kalshi_scanner.slippage import BookLevel
from kalshi_scanner.store import SnapshotStore

SCAN_TS = datetime(2024, 11, 1, tzinfo=UTC)


def _sig(model_prob=None, ci_lo=None, ci_hi=None, yes_ask=None, no_ask=None,
         validated=True, ticker="T", event_ticker="EVT", reason=None):
    scored = model_prob is not None
    return Signal(
        scan_id=1, ticker=ticker, event_ticker=event_ticker, scan_ts=SCAN_TS,
        category="mention", validated=validated,
        event_time=None, market_yes_bid=None, market_yes_ask=yes_ask,
        market_no_bid=None, market_no_ask=no_ask, market_implied_prob=None,
        model_prob=model_prob, ci_lo=ci_lo, ci_hi=ci_hi,
        model_version="m@x" if scored else None, model_hash="x" if scored else None,
        features_json="{}" if scored else None, reason=reason,
    )


def _cfg(**kw):
    base = dict(require_book_for_flag=False)  # tests exercise the gate without needing depth
    base.update(kw)
    return TradingConfig(**base)


def test_inside_ci_band_is_not_flaggable():
    ev = EdgeEvaluator(_cfg())
    r = ev.evaluate(_sig(model_prob=0.5, ci_lo=0.3, ci_hi=0.7, yes_ask=50, no_ask=50))
    assert r.gate_pass is False
    assert r.side is None
    assert r.reason == "inside_ci_band"
    assert r.flaggable is False


def test_yes_side_flaggable_when_price_below_lower_ci():
    ev = EdgeEvaluator(_cfg())
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40, no_ask=62))
    assert r.side == "YES"
    assert r.gate_pass is True
    assert r.raw_edge == pytest.approx(0.8 - 0.40)
    assert r.ev_per_contract > 0
    assert r.contracts > 0
    assert r.flaggable is True
    assert r.reason == "flaggable"


def test_no_side_selected_when_no_ask_cheap():
    ev = EdgeEvaluator(_cfg())
    # model says YES prob 0.2; NO ask 0.40 implies market YES prob 0.60 > ci_hi 0.40.
    r = ev.evaluate(_sig(model_prob=0.2, ci_lo=0.05, ci_hi=0.4, yes_ask=70, no_ask=40))
    assert r.side == "NO"
    assert r.raw_edge == pytest.approx((1 - 0.2) - 0.40)
    assert r.flaggable is True


def test_unvalidated_category_never_flaggable():
    ev = EdgeEvaluator(_cfg())
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40, validated=False))
    assert r.gate_pass is True          # the economics are real
    assert r.flaggable is False          # but the hard gate blocks it
    assert r.reason == "category_not_validated"


def test_requires_orderbook_when_configured():
    ev = EdgeEvaluator(TradingConfig(require_book_for_flag=True))
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40))
    assert r.gate_pass is True
    assert r.book_available is False
    assert r.flaggable is False
    assert r.reason == "no_orderbook_depth"


def test_ev_below_min_blocks_flag():
    ev = EdgeEvaluator(_cfg(min_ev_per_contract=0.5))
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40))
    assert r.reason == "ev_below_min"
    assert r.flaggable is False


def test_kill_switch_forces_not_flaggable():
    # A would-be flaggable market becomes non-flaggable instantly under the switch.
    ev = EdgeEvaluator(_cfg(), kill_switch=True)
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40, no_ask=62))
    assert r.gate_pass is True          # economics still computed
    assert r.flaggable is False
    assert r.reason == "kill_switch"


def test_unscored_signal_passes_through_reason():
    ev = EdgeEvaluator(_cfg())
    r = ev.evaluate(_sig(reason="no_features"))
    assert r.side is None
    assert r.flaggable is False
    assert r.reason == "no_features"


def test_real_book_enables_flag_and_measures_slippage():
    books = {"T": ([BookLevel(40, 1000)], [BookLevel(60, 1000)])}
    ev = EdgeEvaluator(TradingConfig(require_book_for_flag=True), books=books)
    r = ev.evaluate(_sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40, no_ask=62))
    assert r.book_available is True
    assert r.slippage == 0.0             # fills within the top level
    assert r.flaggable is True


def test_evaluate_all_records_to_store(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    ev = EdgeEvaluator(_cfg())
    sigs = [
        _sig(model_prob=0.8, ci_lo=0.6, ci_hi=0.95, yes_ask=40, no_ask=62, ticker="A"),
        _sig(model_prob=0.5, ci_lo=0.3, ci_hi=0.7, yes_ask=50, no_ask=50, ticker="B"),
    ]
    ev.evaluate_all(sigs, scan_id=1, store=store)
    assert store.count_edges() == 2
    assert store.count_edges(flaggable_only=True) == 1

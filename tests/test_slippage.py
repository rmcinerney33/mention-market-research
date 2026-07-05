"""Tests for the slippage / liquidity model."""

from __future__ import annotations

import pytest

from kalshi_scanner.slippage import BookLevel, fallback_fill, walk_book


def test_fills_within_top_level_no_slippage():
    r = walk_book([BookLevel(45, 100)], size=50)
    assert r.avg_price == 0.45
    assert r.filled == 50 and r.fully_filled
    assert r.slippage == 0.0


def test_walks_multiple_levels():
    # 30 @ 45c + 20 @ 46c = 2270c over 50 -> 45.4c avg
    r = walk_book([BookLevel(45, 30), BookLevel(46, 40), BookLevel(48, 100)], size=50)
    assert r.avg_price == pytest.approx(0.454)
    assert r.fully_filled
    assert r.slippage == pytest.approx(0.004)


def test_size_exceeds_depth_is_partial():
    r = walk_book([BookLevel(45, 10), BookLevel(46, 5)], size=50)
    assert r.filled == 15
    assert not r.fully_filled
    assert r.avg_price == pytest.approx((450 + 230) / 15 / 100)


def test_fallback_adds_buffer_and_flags_no_book():
    r = fallback_fill(top_ask_cents=45, size=100, buffer_cents=2)
    assert r.avg_price == 0.47
    assert r.top_price == 0.45
    assert r.slippage == pytest.approx(0.02)
    assert r.book_available is False


def test_walk_book_rejects_empty():
    with pytest.raises(ValueError):
        walk_book([], size=10)

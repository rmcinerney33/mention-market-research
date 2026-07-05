"""Tests for fractional-Kelly sizing."""

from __future__ import annotations

import pytest

from kalshi_scanner.sizing import kelly_fraction_binary, size_position


def test_no_edge_is_zero_kelly():
    # Fair price: p == cost -> no edge.
    assert kelly_fraction_binary(0.5, 0.5) == 0.0
    assert kelly_fraction_binary(0.3, 0.5) == 0.0  # negative edge clipped to 0


def test_positive_edge_kelly_value():
    # p=0.7, cost=0.5 -> 0.7 - 0.3*0.5/0.5 = 0.4
    assert kelly_fraction_binary(0.7, 0.5) == pytest.approx(0.4)


def test_kelly_monotonic_in_probability():
    assert kelly_fraction_binary(0.8, 0.5) > kelly_fraction_binary(0.6, 0.5)


def test_degenerate_costs_return_zero():
    assert kelly_fraction_binary(0.9, 0.0) == 0.0
    assert kelly_fraction_binary(0.9, 1.0) == 0.0


def test_fractional_kelly_without_cap():
    r = size_position(0.7, 0.5, 10_000, kelly_fraction=0.25, max_position_frac=1.0)
    assert r.kelly_full == pytest.approx(0.4)
    assert r.position_fraction == pytest.approx(0.1)   # 0.25 * 0.4
    assert r.contracts == 2000                          # 0.1*10000 / 0.5
    assert r.notional == pytest.approx(1000.0)
    assert r.capped is False


def test_hard_cap_binds():
    r = size_position(0.7, 0.5, 10_000, kelly_fraction=1.0, max_position_frac=0.02)
    assert r.position_fraction == 0.02          # capped from 0.4
    assert r.contracts == 400                    # 0.02*10000 / 0.5
    assert r.capped is True


def test_no_edge_sizes_zero_contracts():
    r = size_position(0.5, 0.5, 10_000)
    assert r.contracts == 0
    assert r.notional == 0.0
    assert r.capped is False

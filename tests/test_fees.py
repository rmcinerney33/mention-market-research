"""Fee-formula tests — the highest-priority correctness check in this project.

Values follow Kalshi's published general formula fee = roundup(0.07*C*P*(1-P))
to the cent. If the live schedule's coefficient differs, update DEFAULT and
these expectations together.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kalshi_scanner.fees import fee_rate_per_contract, trading_fee


def test_one_contract_at_fifty_cents_rounds_up_to_two_cents():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> round up -> $0.02
    assert trading_fee(1, 0.50) == Decimal("0.02")


def test_hundred_contracts_at_fifty_cents():
    # 0.07 * 100 * 0.25 = 1.75 (already a whole cent)
    assert trading_fee(100, 0.50) == Decimal("1.75")


def test_hundred_contracts_at_twenty_cents():
    # 0.07 * 100 * 0.20 * 0.80 = 1.12
    assert trading_fee(100, 0.20) == Decimal("1.12")


def test_ten_contracts_at_thirty_cents_rounds_up():
    # 0.07 * 10 * 0.30 * 0.70 = 0.147 -> round up -> $0.15
    assert trading_fee(10, 0.30) == Decimal("0.15")


def test_fee_is_symmetric_in_price():
    # P*(1-P) is symmetric about 0.5, so YES@P and NO@(1-P) cost the same.
    for c, p in [(100, 0.20), (37, 0.65), (5, 0.10)]:
        assert trading_fee(c, p) == trading_fee(c, 1 - p)


def test_endpoints_are_free():
    assert trading_fee(1000, 0.0) == Decimal("0.00")
    assert trading_fee(1000, 1.0) == Decimal("0.00")


def test_rounds_up_never_down():
    # Any sub-cent positive fee must round up to at least one cent.
    assert trading_fee(1, 0.01) == Decimal("0.01")  # 0.07*0.01*0.99 = 0.000693


def test_configurable_coefficient():
    # 0.035 * 100 * 0.25 = 0.875 -> round up -> $0.88
    assert trading_fee(100, 0.50, coefficient=Decimal("0.035")) == Decimal("0.88")


def test_negative_contracts_rejected():
    with pytest.raises(ValueError):
        trading_fee(-1, 0.5)


def test_price_out_of_range_rejected():
    with pytest.raises(ValueError):
        trading_fee(1, 1.5)


def test_fee_rate_per_contract_is_unrounded():
    assert fee_rate_per_contract(0.50) == pytest.approx(0.0175)
    assert fee_rate_per_contract(0.20) == pytest.approx(0.0112)

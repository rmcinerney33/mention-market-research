"""Tests for the paper-trading engine and the go/no-go gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kalshi_scanner.paper_trading import PaperPosition, build_report, open_from_flag, settle

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _pos(*, side="YES", model_prob=0.8, entry_cost=0.5, entry_price=0.40, ev=0.3,
         contracts=1, flag_time=BASE):
    return PaperPosition(
        ticker="T", event_ticker="E", side=side, contracts=contracts, entry_cost=entry_cost,
        market_entry_price=entry_price, model_prob=model_prob, expected_ev=ev, flag_time=flag_time,
    )


class _FakeFlag:
    ticker = "T"
    event_ticker = "E"
    side = "YES"
    contracts = 10
    model_prob = 0.8
    ev_per_contract = 0.3
    market_price = 0.40
    flagged_at = BASE
    scan_id = 1


def test_open_from_flag_reconstructs_entry_cost():
    p = open_from_flag(_FakeFlag())
    assert p.side == "YES" and p.contracts == 10
    assert p.entry_cost == pytest.approx(0.5)   # p_side 0.8 - ev 0.3
    assert p.expected_ev == 0.3


def test_settle_yes_win_and_loss_pnl():
    win = settle(_pos(contracts=100), event_label=1)
    assert win.won and win.pnl == pytest.approx(100 * (1 - 0.5))
    loss = settle(_pos(contracts=100), event_label=0)
    assert not loss.won and loss.pnl == pytest.approx(100 * (0 - 0.5))


def test_settle_no_side_wins_when_phrase_absent():
    p = settle(_pos(side="NO"), event_label=0)   # NO wins when phrase NOT used
    assert p.won
    p2 = settle(_pos(side="NO"), event_label=1)
    assert not p2.won


def test_report_basic_metrics():
    positions = [settle(_pos(contracts=10), 1), settle(_pos(contracts=10), 0)]
    r = build_report(positions, min_flags=1, min_weeks=0)
    assert r.n_settled == 2
    assert r.pnl == pytest.approx(10 * 0.5 + 10 * -0.5)   # one win, one loss -> 0
    assert r.hit_rate == pytest.approx(0.5)
    # p_side=0.8; brier = mean((0.8-1)^2, (0.8-0)^2) = mean(0.04, 0.64)
    assert r.realized_brier == pytest.approx((0.04 + 0.64) / 2)


def test_no_go_on_insufficient_sample():
    r = build_report([settle(_pos(), 1)], min_flags=50, min_weeks=8)
    assert r.go is False
    assert any("insufficient sample" in n for n in r.notes)


def test_go_when_consistent_over_meaningful_sample():
    positions = []
    for i in range(60):
        p = _pos(flag_time=BASE + timedelta(days=i))     # 60 days > 8 weeks
        settle(p, 1 if i < 48 else 0)                     # 80% win -> realized ~ expected 0.3
        positions.append(p)
    r = build_report(positions, min_flags=50, min_weeks=8)
    assert r.realized_edge_mean == pytest.approx(0.3, abs=1e-9)
    assert r.go is True
    assert "consistent with backtest" in r.notes[-1]


def test_no_go_when_realized_below_expected():
    positions = []
    for i in range(60):
        p = _pos(flag_time=BASE + timedelta(days=i))
        settle(p, 1 if i < 18 else 0)                     # 30% win -> negative realized edge
        positions.append(p)
    r = build_report(positions, min_flags=50, min_weeks=8)
    assert r.realized_edge_mean < 0
    assert r.go is False


def test_max_drawdown_tracks_peak_to_trough():
    # Three wins then three losses -> peak +1.5, trough 0.0 -> drawdown 1.5.
    positions = []
    for i in range(6):
        p = _pos(contracts=1, flag_time=BASE + timedelta(days=i))
        settle(p, 1 if i < 3 else 0, settled_at=BASE + timedelta(days=i))
        positions.append(p)
    r = build_report(positions, min_flags=1, min_weeks=0)
    assert r.max_drawdown == pytest.approx(1.5)   # lost 0.5 x 3 from the peak


def test_decay_uses_exit_price_when_available():
    p = _pos()
    settle(p, 1, exit_market_price=0.7)   # market rose from 0.40 toward model -> toward model
    r = build_report([p], min_flags=1, min_weeks=0)
    assert r.decay_toward_model == 1.0


def test_positions_store_round_trip(tmp_path):
    from kalshi_scanner.store import SnapshotStore

    store = SnapshotStore(tmp_path / "s.sqlite")
    p_open = _pos(contracts=10)
    p_settled = settle(_pos(contracts=5), 1, settled_at=BASE)
    store.record_positions([p_open, p_settled])

    assert store.count_positions() == 2
    assert store.count_positions(settled_only=True) == 1
    loaded = store.load_positions()
    assert {p.contracts for p in loaded} == {10, 5}
    settled_loaded = [p for p in loaded if p.outcome is not None][0]
    assert settled_loaded.won and settled_loaded.pnl == pytest.approx(5 * 0.5)

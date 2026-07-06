"""Tests for calibration-drift detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_scanner.config import MonitoringConfig
from kalshi_scanner.monitoring import assess_drift
from kalshi_scanner.paper_trading import PaperPosition, settle

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _cfg(**kw):
    base = dict(min_drift_sample=5, drift_window=30)
    base.update(kw)
    return MonitoringConfig(**base)


def _settled(model_prob, label, i):
    p = PaperPosition(
        ticker="T", event_ticker="E", side="YES", contracts=1, entry_cost=0.5,
        market_entry_price=0.4, model_prob=model_prob, expected_ev=0.3,
        flag_time=BASE + timedelta(days=i),
    )
    return settle(p, label, settled_at=BASE + timedelta(days=i))


def test_insufficient_sample_is_not_degraded():
    positions = [_settled(0.8, 1, i) for i in range(3)]
    status = assess_drift(positions, _cfg())
    assert status.degraded is False
    assert "insufficient sample" in status.note


def test_well_calibrated_is_not_degraded():
    # p_side=0.8, 80% win -> Brier 0.16, well under baseline 0.24 (+15%).
    positions = [_settled(0.8, 1 if i < 16 else 0, i) for i in range(20)]
    status = assess_drift(positions, _cfg())
    assert status.rolling_brier < status.baseline_brier
    assert status.degraded is False


def test_miscalibrated_is_degraded():
    # p_side=0.9 but only 30% win -> Brier ~0.57, far above tolerance.
    positions = [_settled(0.9, 1 if i < 6 else 0, i) for i in range(20)]
    status = assess_drift(positions, _cfg())
    assert status.rolling_brier > status.baseline_brier
    assert status.degraded is True
    assert "degraded" in status.note


def test_window_limits_to_recent():
    # 40 good then evaluated with window 30 -> only recent 30 considered.
    positions = [_settled(0.8, 1 if i < 16 else 0, i) for i in range(40)]
    status = assess_drift(positions, _cfg(drift_window=30))
    assert status.window == 30

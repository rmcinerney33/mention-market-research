"""Tests for push alerting (no network — injected transport)."""

from __future__ import annotations

from datetime import UTC, datetime

from kalshi_scanner.alerts import Alerter
from kalshi_scanner.config import AlertConfig
from kalshi_scanner.flags import Flag


def _flag(ev_per_contract, contracts, ticker="T"):
    return Flag(
        flagged_at=datetime(2024, 11, 1, tzinfo=UTC), scan_id=1, ticker=ticker,
        event_ticker="EVT", category="mention", side="YES", model_prob=0.8,
        ci_lo=0.6, ci_hi=0.95, market_price=0.40, raw_edge=0.4,
        ev_per_contract=ev_per_contract, contracts=contracts, notional=contracts * 0.4,
        slippage=0.0, book_available=True, model_version="m@x", model_hash="x",
        features_json="{}", event_time=None, reason="flaggable",
    )


class _FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, data, headers):
        self.calls.append({"url": url, "data": data, "headers": headers})
        return 200


def test_disabled_alerter_sends_nothing():
    t = _FakeTransport()
    a = Alerter(AlertConfig(enabled=False, ntfy_topic="t"), transport=t)
    assert a.notify_flags([_flag(0.1, 100)]) == []
    assert t.calls == []


def test_only_flags_above_ev_threshold_are_sent():
    t = _FakeTransport()
    a = Alerter(AlertConfig(enabled=True, ntfy_topic="mytopic", min_ev_notional=1.0), transport=t)
    big = _flag(0.10, 20, ticker="BIG")   # EV*size = 2.0 -> sent
    small = _flag(0.01, 5, ticker="SMALL")  # EV*size = 0.05 -> skipped
    sent = a.notify_flags([big, small])
    assert [f.ticker for f in sent] == ["BIG"]
    assert len(t.calls) == 1
    assert t.calls[0]["url"] == "https://ntfy.sh/mytopic"
    assert b"BIG" in t.calls[0]["data"] or "BIG" in t.calls[0]["headers"]["Title"]


def test_enabled_without_topic_sends_nothing():
    t = _FakeTransport()
    a = Alerter(AlertConfig(enabled=True, ntfy_topic=""), transport=t)
    assert a.notify_flags([_flag(0.5, 100)]) == []
    assert t.calls == []

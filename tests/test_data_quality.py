"""Tests for data-quality checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_scanner.config import MonitoringConfig, load_scanner_config
from kalshi_scanner.data_quality import (
    check_corpus_staleness,
    check_coverage_gaps,
    check_scan_freshness,
    check_schema,
    run_data_quality,
)
from kalshi_scanner.scanner import Scanner
from kalshi_scanner.store import SnapshotStore
from tests.kalshi_fakes import StubClient, load_fixture

NOW = datetime(2024, 11, 5, 12, 0, tzinfo=UTC)


def test_scan_freshness():
    assert check_scan_freshness(None, NOW, 900).severity == "error"
    stale = check_scan_freshness(NOW - timedelta(hours=1), NOW, 900)
    assert stale.severity == "warn" and "old" in stale.detail
    assert check_scan_freshness(NOW - timedelta(minutes=5), NOW, 900) is None


def test_schema_detects_missing_fields():
    good = [{"ticker": "A", "title": "t", "close_time": "x", "yes_ask": 1, "no_ask": 2}]
    assert check_schema(good, ("ticker", "title", "close_time", "yes_ask", "no_ask")) is None
    bad = [{"ticker": "A", "title": "t"}]  # missing yes_ask/no_ask/close_time
    issue = check_schema(bad, ("ticker", "title", "close_time", "yes_ask", "no_ask"))
    assert issue.severity == "error" and "schema change" in issue.detail


def test_coverage_and_corpus_helpers():
    assert check_coverage_gaps([]) is None
    assert check_coverage_gaps([{"type": "failed_run"}]).severity == "warn"
    assert check_corpus_staleness(None, NOW, 7, "gdelt").severity == "warn"
    assert check_corpus_staleness(NOW - timedelta(days=10), NOW, 7, "gdelt").severity == "warn"
    assert check_corpus_staleness(NOW - timedelta(days=1), NOW, 7, "gdelt") is None


def _seed(tmp_path):
    cfg = load_scanner_config()
    from dataclasses import replace
    cfg = replace(cfg, db_path=tmp_path / "s.sqlite")
    markets = (load_fixture("markets_page1.json")["markets"]
               + load_fixture("markets_page2.json")["markets"])
    store = SnapshotStore(cfg.db_path)
    Scanner(cfg, StubClient(markets=markets), store, clock=lambda: NOW).scan_once()
    return store


def test_run_data_quality_clean_when_fresh_and_complete(tmp_path):
    store = _seed(tmp_path)
    issues = run_data_quality(store, MonitoringConfig(), now=NOW, expected_interval_s=300)
    assert issues == []


def test_run_data_quality_flags_stale_scan(tmp_path):
    store = _seed(tmp_path)
    later = NOW + timedelta(hours=2)
    issues = run_data_quality(store, MonitoringConfig(max_scan_age_s=900),
                              now=later, expected_interval_s=300)
    assert any(i.check == "scan_freshness" for i in issues)

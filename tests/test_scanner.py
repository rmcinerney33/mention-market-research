"""Tests for scan orchestration: matching, persistence, downtime, no-lookahead."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from kalshi_scanner.config import CategoryRule, load_scanner_config
from kalshi_scanner.scanner import Scanner
from kalshi_scanner.store import SnapshotStore
from tests.kalshi_fakes import StubClient, load_fixture

FIXED = datetime(2024, 11, 5, 12, 0, tzinfo=UTC)


def _config(tmp_path, **overrides):
    cfg = load_scanner_config()
    # The default mention category is title-only; enable the (bounded) full-market
    # scan so the stub's markets are examined. Tests can override.
    overrides.setdefault("allow_full_scan", True)
    return replace(cfg, db_path=tmp_path / "snap.sqlite", **overrides)


def _all_markets():
    return (
        load_fixture("markets_page1.json")["markets"]
        + load_fixture("markets_page2.json")["markets"]
    )


def test_scan_once_matches_and_persists(tmp_path):
    cfg = _config(tmp_path)
    client = StubClient(markets=_all_markets())
    store = SnapshotStore(cfg.db_path)
    scanner = Scanner(cfg, client, store, clock=lambda: FIXED)

    result = scanner.scan_once()
    # 2 of 3 markets are mention markets; the CPI market is filtered out.
    assert result.ok
    assert result.n_markets == 2
    assert store.count_snapshots() == 2
    assert store.count_runs("ok") == 1
    assert {s.category for s in result.snapshots} == {"mention"}


def test_no_lookahead_time_to_resolution_uses_scan_clock(tmp_path):
    cfg = _config(tmp_path)
    scanner = Scanner(cfg, StubClient(markets=_all_markets()),
                      SnapshotStore(cfg.db_path), clock=lambda: FIXED)
    snap = scanner.scan_once().snapshots[0]
    # Everything is anchored to the injected scan time — no future data.
    assert snap.scan_ts == FIXED
    assert snap.close_time > FIXED
    assert snap.time_to_resolution_s == (snap.close_time - FIXED).total_seconds()


def test_downtime_records_failed_run_and_no_snapshots(tmp_path):
    cfg = _config(tmp_path)
    client = StubClient(error=ConnectionError("kalshi unreachable"))
    store = SnapshotStore(cfg.db_path)
    scanner = Scanner(cfg, client, store, clock=lambda: FIXED)

    result = scanner.scan_once()
    assert not result.ok
    assert "kalshi unreachable" in result.error
    assert store.count_snapshots() == 0
    assert store.count_runs("failed") == 1
    assert any(g["type"] == "failed_run" for g in store.find_gaps(cfg.poll_interval_s))


def test_disabled_config_records_skipped_run(tmp_path):
    cfg = _config(tmp_path, enabled=False)
    store = SnapshotStore(cfg.db_path)
    scanner = Scanner(cfg, StubClient(markets=_all_markets()), store, clock=lambda: FIXED)

    result = scanner.scan_once()
    assert not result.ok
    assert result.error == "disabled"
    assert store.count_snapshots() == 0
    assert store.count_runs("skipped") == 1


def test_kill_switch_still_scans_but_warns(tmp_path, caplog):
    # Phase 1 has no flagging; the kill switch pauses flags, not scanning.
    cfg = _config(tmp_path, kill_switch=True)
    store = SnapshotStore(cfg.db_path)
    scanner = Scanner(cfg, StubClient(markets=_all_markets()), store, clock=lambda: FIXED)

    with caplog.at_level("WARNING"):
        result = scanner.scan_once()
    assert result.ok and result.n_markets == 2
    assert any("kill_switch" in r.message for r in caplog.records)


def test_run_forever_respects_max_iterations(tmp_path):
    cfg = _config(tmp_path)
    sleeps: list[float] = []
    scanner = Scanner(
        cfg, StubClient(markets=_all_markets()), SnapshotStore(cfg.db_path),
        clock=lambda: FIXED, sleep=sleeps.append,
    )
    scanner.run_forever(max_iterations=3)
    assert scanner.store.count_runs("ok") == 3
    assert len(sleeps) == 2  # sleeps between scans, not after the last


def test_scan_respects_max_markets_cap(tmp_path):
    # 25 matching markets available, but the cap stops examination at 10.
    markets = [
        {"ticker": f"M{i}", "title": f"Will X say word{i}?", "close_time": "2024-12-01T00:00:00Z"}
        for i in range(25)
    ]
    cfg = _config(tmp_path, max_markets_per_scan=10, allow_full_scan=True)
    store = SnapshotStore(cfg.db_path)
    scanner = Scanner(cfg, StubClient(markets=markets), store, clock=lambda: FIXED)

    result = scanner.scan_once()
    assert result.ok
    assert result.n_markets == 10  # capped, even though 25 would have matched


def test_series_scoped_category_queries_server_side(tmp_path):
    # A category with series_tickers is fetched with targeted queries, not a crawl.
    cat = CategoryRule(name="fed", validated=False, series_tickers=["KXFED", "KXCPI"])
    cfg = _config(tmp_path, categories=[cat], allow_full_scan=False)
    client = StubClient(markets=[])
    scanner = Scanner(cfg, client, SnapshotStore(cfg.db_path), clock=lambda: FIXED)

    scanner.scan_once()
    queried = sorted(c["series_ticker"] for c in client.calls)
    assert queried == ["KXCPI", "KXFED"]           # one targeted query per series
    assert all(c["max_pages"] is not None for c in client.calls)  # bounded


def test_title_only_category_skipped_without_full_scan(tmp_path, caplog):
    # The default mention category is title-only; with allow_full_scan off it is
    # skipped (Kalshi can't filter by title), and no full crawl is issued.
    cfg = _config(tmp_path, allow_full_scan=False)
    client = StubClient(markets=_all_markets())
    scanner = Scanner(cfg, client, SnapshotStore(cfg.db_path), clock=lambda: FIXED)

    with caplog.at_level("WARNING"):
        result = scanner.scan_once()
    assert result.ok
    assert result.n_markets == 0
    assert client.calls == []  # never hit the API
    assert any("allow_full_scan" in r.message for r in caplog.records)

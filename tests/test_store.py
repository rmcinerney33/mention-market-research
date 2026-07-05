"""Tests for the SQLite snapshot store and gap logging."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_scanner.market import MarketSnapshot
from kalshi_scanner.store import SnapshotStore
from tests.kalshi_fakes import load_fixture


def _snapshot(scan_ts):
    raw = load_fixture("markets_page1.json")["markets"][0]
    return MarketSnapshot.from_api(raw, scan_ts=scan_ts, category="mention")


def test_ok_run_round_trip(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    scan_ts = datetime(2024, 11, 5, 12, tzinfo=UTC)
    scan_id = store.start_run(scan_ts, ["mention"])
    store.record_snapshots(scan_id, [_snapshot(scan_ts)])
    store.finish_run(scan_id, scan_ts, "ok", 1)

    assert store.count_snapshots() == 1
    assert store.count_runs("ok") == 1
    rows = store.snapshots_for_scan(scan_id)
    assert rows[0]["ticker"] == "MENTIONSPEECH-24NOV05-TARIFF"
    assert rows[0]["yes_ask"] == 45
    assert rows[0]["time_to_resolution_s"] > 0


def test_failed_run_writes_no_snapshots(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    scan_ts = datetime(2024, 11, 5, 12, tzinfo=UTC)
    scan_id = store.start_run(scan_ts, ["mention"])
    store.finish_run(scan_id, scan_ts, "failed", 0, error="ConnectionError: down")

    assert store.count_snapshots() == 0
    assert store.count_runs("failed") == 1


def test_find_gaps_detects_failed_and_missing(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    t0 = datetime(2024, 11, 5, 12, 0, tzinfo=UTC)

    r1 = store.start_run(t0, ["mention"])
    store.finish_run(r1, t0, "ok", 1)

    # A failed run 5 min later.
    r2 = store.start_run(t0 + timedelta(minutes=5), ["mention"])
    store.finish_run(r2, t0 + timedelta(minutes=5), "failed", 0, error="down")

    # Next success is an hour after the first — a missing-scans stretch.
    r3 = store.start_run(t0 + timedelta(hours=1), ["mention"])
    store.finish_run(r3, t0 + timedelta(hours=1), "ok", 1)

    gaps = store.find_gaps(expected_interval_s=300)
    types = {g["type"] for g in gaps}
    assert "failed_run" in types
    assert "missing_scans" in types


def test_gaps_never_interpolate(tmp_path):
    # A failed run must not create snapshot rows — the gap stays a gap.
    store = SnapshotStore(tmp_path / "s.sqlite")
    t0 = datetime(2024, 11, 5, 12, tzinfo=UTC)
    r = store.start_run(t0, ["mention"])
    store.finish_run(r, t0, "failed", 0, error="down")
    assert store.count_snapshots() == 0


def test_load_snapshots_round_trip(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    scan_ts = datetime(2024, 11, 5, 12, tzinfo=UTC)
    scan_id = store.start_run(scan_ts, ["mention"])
    snap = _snapshot(scan_ts)
    store.record_snapshots(scan_id, [snap])
    store.finish_run(scan_id, scan_ts, "ok", 1)

    loaded = store.load_snapshots(scan_id)
    assert len(loaded) == 1
    assert loaded[0].ticker == snap.ticker
    assert loaded[0].yes_ask == snap.yes_ask
    assert loaded[0].category == "mention"
    assert loaded[0].scan_ts == scan_ts


def test_latest_ok_scan_id(tmp_path):
    store = SnapshotStore(tmp_path / "s.sqlite")
    t0 = datetime(2024, 11, 5, 12, tzinfo=UTC)
    assert store.latest_ok_scan_id() is None
    r1 = store.start_run(t0, ["mention"])
    store.finish_run(r1, t0, "ok", 0)
    r2 = store.start_run(t0, ["mention"])
    store.finish_run(r2, t0, "failed", 0, error="down")
    assert store.latest_ok_scan_id() == r1  # r2 failed, so r1 is the latest ok

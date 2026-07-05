"""SQLite persistence for scans and snapshots.

Two tables:

- ``scan_runs`` — one row per scan attempt, success or failure. A failed scan
  (API down, schema change, etc.) is recorded with ``status='failed'`` and
  ``n_markets=0``. We **never** write fabricated snapshots to paper over a gap;
  gaps stay visible as failed or missing runs.
- ``market_snapshots`` — one row per observed market per successful scan.

Every snapshot becomes part of a growing historical dataset keyed by scan time.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .market import MarketSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    scan_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,          -- running | ok | failed | skipped
    n_markets     INTEGER NOT NULL DEFAULT 0,
    pages_fetched INTEGER NOT NULL DEFAULT 0,
    categories    TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id              INTEGER NOT NULL REFERENCES scan_runs(scan_id),
    scan_ts              TEXT NOT NULL,
    ticker               TEXT NOT NULL,
    event_ticker         TEXT,
    series_ticker        TEXT,
    title                TEXT,
    category             TEXT,
    kalshi_category      TEXT,
    status               TEXT,
    yes_bid              INTEGER,
    yes_ask              INTEGER,
    no_bid               INTEGER,
    no_ask               INTEGER,
    last_price           INTEGER,
    volume               INTEGER,
    volume_24h           INTEGER,
    open_interest        INTEGER,
    open_time            TEXT,
    close_time           TEXT,
    time_to_resolution_s REAL,
    raw_json             TEXT
);

CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON market_snapshots(ticker, scan_ts);
CREATE INDEX IF NOT EXISTS idx_snap_scan ON market_snapshots(scan_id);
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


class SnapshotStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- run lifecycle ---------------------------------------------------
    def start_run(self, started_at: datetime, categories: list[str]) -> int:
        cur = self._conn.execute(
            "INSERT INTO scan_runs (started_at, status, categories) VALUES (?, 'running', ?)",
            (_iso(started_at), ",".join(categories)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        scan_id: int,
        finished_at: datetime,
        status: str,
        n_markets: int,
        pages_fetched: int = 0,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE scan_runs SET finished_at=?, status=?, n_markets=?, pages_fetched=?, error=? "
            "WHERE scan_id=?",
            (_iso(finished_at), status, n_markets, pages_fetched, error, scan_id),
        )
        self._conn.commit()

    def record_snapshots(self, scan_id: int, snapshots: list[MarketSnapshot]) -> None:
        self._conn.executemany(
            """INSERT INTO market_snapshots (
                scan_id, scan_ts, ticker, event_ticker, series_ticker, title,
                category, kalshi_category, status, yes_bid, yes_ask, no_bid, no_ask,
                last_price, volume, volume_24h, open_interest, open_time, close_time,
                time_to_resolution_s, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    scan_id, _iso(s.scan_ts), s.ticker, s.event_ticker, s.series_ticker,
                    s.title, s.category, s.kalshi_category, s.status, s.yes_bid, s.yes_ask,
                    s.no_bid, s.no_ask, s.last_price, s.volume, s.volume_24h, s.open_interest,
                    _iso(s.open_time), _iso(s.close_time), s.time_to_resolution_s,
                    json.dumps(s.raw),
                )
                for s in snapshots
            ],
        )
        self._conn.commit()

    # -- reads / introspection ------------------------------------------
    def count_snapshots(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0])

    def count_runs(self, status: str | None = None) -> int:
        if status is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM scan_runs").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM scan_runs WHERE status=?", (status,)
            ).fetchone()[0]
        )

    def latest_run(self) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM scan_runs ORDER BY scan_id DESC LIMIT 1"
        ).fetchone()

    def snapshots_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM market_snapshots WHERE scan_id=? ORDER BY ticker", (scan_id,)
            ).fetchall()
        )

    def find_gaps(self, expected_interval_s: float, slack: float = 1.5) -> list[dict]:
        """Report coverage gaps: failed runs, and stretches between successful
        runs longer than ``expected_interval_s * slack``. Never interpolated —
        this only *reports* what is missing.
        """
        rows = list(
            self._conn.execute(
                "SELECT scan_id, started_at, finished_at, status FROM scan_runs ORDER BY started_at"
            ).fetchall()
        )
        gaps: list[dict] = []
        threshold = expected_interval_s * slack
        prev_ok_time: datetime | None = None
        for r in rows:
            if r["status"] == "failed":
                gaps.append({"type": "failed_run", "scan_id": r["scan_id"], "at": r["started_at"]})
            started = datetime.fromisoformat(r["started_at"])
            if prev_ok_time is not None:
                delta = (started - prev_ok_time).total_seconds()
                if delta > threshold:
                    gaps.append(
                        {"type": "missing_scans", "from": prev_ok_time.isoformat(),
                         "to": r["started_at"], "gap_s": delta}
                    )
            if r["status"] == "ok":
                prev_ok_time = started
        return gaps


def utcnow() -> datetime:
    return datetime.now(UTC)

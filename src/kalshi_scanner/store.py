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

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id             INTEGER,
    ticker              TEXT NOT NULL,
    event_ticker        TEXT,
    scan_ts             TEXT,
    category            TEXT,
    validated           INTEGER,
    event_time          TEXT,
    market_yes_bid      INTEGER,
    market_yes_ask      INTEGER,
    market_no_bid       INTEGER,
    market_no_ask       INTEGER,
    market_implied_prob REAL,
    model_prob          REAL,
    ci_lo               REAL,
    ci_hi               REAL,
    model_version       TEXT,
    model_hash          TEXT,
    features_json       TEXT,
    reason              TEXT
);

CREATE INDEX IF NOT EXISTS idx_signal_scan ON signals(scan_id);
CREATE INDEX IF NOT EXISTS idx_signal_ticker ON signals(ticker);

CREATE TABLE IF NOT EXISTS edges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id          INTEGER,
    ticker           TEXT NOT NULL,
    event_ticker     TEXT,
    category         TEXT,
    validated        INTEGER,
    side             TEXT,
    model_prob       REAL,
    ci_lo            REAL,
    ci_hi            REAL,
    market_price     REAL,
    raw_edge         REAL,
    fee_per_contract REAL,
    effective_cost   REAL,
    ev_per_contract  REAL,
    kelly_fraction   REAL,
    contracts        INTEGER,
    notional         REAL,
    slippage         REAL,
    book_available   INTEGER,
    gate_pass        INTEGER,
    flaggable        INTEGER,
    reason           TEXT
);

CREATE INDEX IF NOT EXISTS idx_edge_scan ON edges(scan_id);
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

    def load_snapshots(self, scan_id: int) -> list[MarketSnapshot]:
        """Rebuild :class:`MarketSnapshot` objects from a stored scan.

        The full raw API market is round-tripped via ``raw_json``, so signals can
        be (re)generated offline from a past scan without re-hitting Kalshi.
        """
        out: list[MarketSnapshot] = []
        for r in self.snapshots_for_scan(scan_id):
            raw = json.loads(r["raw_json"]) if r["raw_json"] else {"ticker": r["ticker"]}
            out.append(
                MarketSnapshot.from_api(
                    raw, scan_ts=datetime.fromisoformat(r["scan_ts"]), category=r["category"]
                )
            )
        return out

    def latest_ok_scan_id(self) -> int | None:
        row = self._conn.execute(
            "SELECT scan_id FROM scan_runs WHERE status='ok' ORDER BY scan_id DESC LIMIT 1"
        ).fetchone()
        return int(row[0]) if row else None

    # -- signals ---------------------------------------------------------
    def record_signals(self, signals: list) -> None:
        self._conn.executemany(
            """INSERT INTO signals (
                scan_id, ticker, event_ticker, scan_ts, category, validated, event_time,
                market_yes_bid, market_yes_ask, market_no_bid, market_no_ask, market_implied_prob,
                model_prob, ci_lo, ci_hi, model_version, model_hash, features_json, reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    s.scan_id, s.ticker, s.event_ticker, _iso(s.scan_ts), s.category,
                    int(s.validated), _iso(s.event_time), s.market_yes_bid, s.market_yes_ask,
                    s.market_no_bid, s.market_no_ask, s.market_implied_prob,
                    s.model_prob, s.ci_lo, s.ci_hi, s.model_version, s.model_hash,
                    s.features_json, s.reason,
                )
                for s in signals
            ],
        )
        self._conn.commit()

    def count_signals(self, scored_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM signals"
        if scored_only:
            sql += " WHERE model_prob IS NOT NULL"
        return int(self._conn.execute(sql).fetchone()[0])

    def signals_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM signals WHERE scan_id=? ORDER BY ticker", (scan_id,)
            ).fetchall()
        )

    # -- edges -----------------------------------------------------------
    def record_edges(self, edges: list) -> None:
        self._conn.executemany(
            """INSERT INTO edges (
                scan_id, ticker, event_ticker, category, validated, side, model_prob, ci_lo, ci_hi,
                market_price, raw_edge, fee_per_contract, effective_cost, ev_per_contract,
                kelly_fraction, contracts, notional, slippage, book_available,
                gate_pass, flaggable, reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    e.scan_id, e.ticker, e.event_ticker, e.category, int(e.validated), e.side,
                    e.model_prob, e.ci_lo, e.ci_hi, e.market_price, e.raw_edge, e.fee_per_contract,
                    e.effective_cost, e.ev_per_contract, e.kelly_fraction, e.contracts,
                    e.notional, e.slippage, int(e.book_available), int(e.gate_pass),
                    int(e.flaggable), e.reason,
                )
                for e in edges
            ],
        )
        self._conn.commit()

    def count_edges(self, flaggable_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM edges"
        if flaggable_only:
            sql += " WHERE flaggable=1"
        return int(self._conn.execute(sql).fetchone()[0])

    def edges_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM edges WHERE scan_id=? ORDER BY flaggable DESC, ticker", (scan_id,)
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

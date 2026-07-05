"""Snapshot data types.

A :class:`MarketSnapshot` captures exactly what was observable about a market at
one scan instant — nothing that requires future information. Prices are kept in
Kalshi-native **cents** (integers 0–100); conversion to probabilities happens
downstream where the fee math lives, so nothing is lost here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def parse_ts(value: str | None) -> datetime | None:
    """Parse a Kalshi RFC3339 timestamp into a tz-aware UTC datetime."""
    if not value:
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _int_or_none(raw: dict[str, Any], key: str) -> int | None:
    v = raw.get(key)
    return int(v) if isinstance(v, (int, float)) else None


@dataclass(frozen=True)
class MarketSnapshot:
    scan_ts: datetime          # tz-aware UTC — the instant of observation
    ticker: str
    event_ticker: str | None
    series_ticker: str | None
    title: str
    category: str              # our matched category name
    kalshi_category: str | None
    status: str | None
    # Book + tape, in cents (0–100). None means the field was absent.
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    last_price: int | None
    volume: int | None
    volume_24h: int | None
    open_interest: int | None
    open_time: datetime | None
    close_time: datetime | None
    time_to_resolution_s: float | None   # close_time - scan_ts, computed at scan
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(
        cls,
        raw: dict[str, Any],
        *,
        scan_ts: datetime,
        category: str,
    ) -> MarketSnapshot:
        if scan_ts.tzinfo is None:
            raise ValueError("scan_ts must be timezone-aware (UTC)")

        close_time = parse_ts(raw.get("close_time")) or parse_ts(raw.get("expiration_time"))
        ttr = (close_time - scan_ts).total_seconds() if close_time is not None else None

        event_ticker = raw.get("event_ticker")
        series = raw.get("series_ticker")
        if not series and event_ticker and "-" in event_ticker:
            series = event_ticker.split("-", 1)[0]

        return cls(
            scan_ts=scan_ts,
            ticker=raw["ticker"],
            event_ticker=event_ticker,
            series_ticker=series,
            title=raw.get("title", "") or "",
            category=category,
            kalshi_category=raw.get("category"),
            status=raw.get("status"),
            yes_bid=_int_or_none(raw, "yes_bid"),
            yes_ask=_int_or_none(raw, "yes_ask"),
            no_bid=_int_or_none(raw, "no_bid"),
            no_ask=_int_or_none(raw, "no_ask"),
            last_price=_int_or_none(raw, "last_price"),
            volume=_int_or_none(raw, "volume"),
            volume_24h=_int_or_none(raw, "volume_24h"),
            open_interest=_int_or_none(raw, "open_interest"),
            open_time=parse_ts(raw.get("open_time")),
            close_time=close_time,
            time_to_resolution_s=ttr,
            raw=raw,
        )


@dataclass
class ScanResult:
    scan_ts: datetime
    ok: bool
    n_markets: int
    snapshots: list[MarketSnapshot] = field(default_factory=list)
    pages_fetched: int = 0
    error: str | None = None

"""Kalshi market-data adapter.

Kalshi's public trade API exposes markets, candlesticks, and trade history.
Real-time L2 order books are available, but **arbitrary historical order-book
snapshots are not** — you cannot ask "what was the book at 2024-09-01T14:00Z".
Our workaround is to reconstruct an implied-probability time series from
candlesticks / trade history and read off the value at each target lead time.
Consequently our "market price at T-24h/T-6h/T-1h" is a *trade-derived mid*,
not a true book mid. This is documented as a limitation.

Auth (for endpoints that require it) uses an API key id + RSA private key,
supplied via env vars ``KALSHI_KEY_ID`` and ``KALSHI_PRIVATE_KEY_PATH``. Public
market-data reads generally do not require signing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class KalshiClient:
    base_url: str = DEFAULT_BASE_URL
    session: requests.Session | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()

    # ---- Discovery -------------------------------------------------------
    def list_markets(self, *, series_ticker: str | None = None, status: str | None = None,
                     limit: int = 200) -> list[dict]:
        """List markets, optionally filtered by series and status.

        Mention markets tend to live under event/series tickers; the caller is
        responsible for identifying which series correspond to phrase-mention
        contracts for the speaker of interest.
        """
        params: dict[str, object] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        markets: list[dict] = []
        cursor: str | None = None
        while True:
            if cursor:
                params["cursor"] = cursor
            resp = self.session.get(f"{self.base_url}/markets", params=params, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
            markets.extend(payload.get("markets", []))
            cursor = payload.get("cursor")
            if not cursor or not payload.get("markets"):
                break
        return markets

    # ---- Price history ---------------------------------------------------
    def market_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        start: datetime,
        end: datetime,
        period_interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """Fetch candlesticks and return a tidy implied-probability series.

        Returns columns: ``observed_time`` (tz-aware UTC), ``implied_prob``
        (mid of the candle's yes bid/ask if present, else the close price /
        100). Empty frame if Kalshi returns no candles for the window.
        """
        params = {
            "start_ts": int(start.replace(tzinfo=UTC).timestamp()),
            "end_ts": int(end.replace(tzinfo=UTC).timestamp()),
            "period_interval": period_interval_minutes,
        }
        url = f"{self.base_url}/series/{series_ticker}/markets/{ticker}/candlesticks"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        candles = resp.json().get("candlesticks", [])
        rows = []
        for c in candles:
            ts = datetime.fromtimestamp(c["end_period_ts"], tz=UTC)
            prob = _candle_implied_prob(c)
            if prob is not None:
                rows.append({"observed_time": ts, "implied_prob": prob})
        return pd.DataFrame(rows, columns=["observed_time", "implied_prob"])


def _candle_implied_prob(candle: dict) -> float | None:
    """Best-effort implied probability from a Kalshi candlestick.

    Prefer the yes bid/ask mid; fall back to the trade-price close. Kalshi
    prices are in cents (0-100), so we divide by 100 to get a probability.
    """
    yes_bid = (candle.get("yes_bid") or {}).get("close")
    yes_ask = (candle.get("yes_ask") or {}).get("close")
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0 / 100.0
    close = (candle.get("price") or {}).get("close")
    if close is not None:
        return close / 100.0
    return None

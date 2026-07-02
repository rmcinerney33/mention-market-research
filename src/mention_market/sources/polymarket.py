"""Polymarket market-data adapter.

Polymarket historical prices are more accessible than Kalshi's: the Gamma API
serves market metadata, and the CLOB ``/prices-history`` endpoint returns a
time series of the traded price for a given token (outcome). This is typically
the primary market source for this project, with Kalshi used for corroboration
where the same mention question trades on both venues.

The CLOB price series is a last-trade/mid series sampled at a chosen fidelity;
like Kalshi it is trade-derived, not a reconstructed book, but the sampling is
denser and the history deeper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

DEFAULT_GAMMA_URL = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"


@dataclass
class PolymarketClient:
    gamma_url: str = DEFAULT_GAMMA_URL
    clob_url: str = DEFAULT_CLOB_URL
    session: requests.Session | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()

    # ---- Discovery -------------------------------------------------------
    def search_markets(self, query: str, *, limit: int = 100, closed: bool | None = None) -> list[dict]:
        """Search Gamma markets by free text (e.g. a phrase or speaker name)."""
        params: dict[str, object] = {"limit": limit}
        if closed is not None:
            params["closed"] = str(closed).lower()
        # Gamma supports a `_query`-style search on some deployments; we filter
        # client-side as a robust fallback so this works across API versions.
        resp = self.session.get(f"{self.gamma_url}/markets", params=params, timeout=self.timeout)
        resp.raise_for_status()
        markets = resp.json()
        if isinstance(markets, dict):
            markets = markets.get("data", [])
        q = query.lower()
        return [m for m in markets if q in (m.get("question", "") + m.get("description", "")).lower()]

    # ---- Price history ---------------------------------------------------
    def prices_history(
        self,
        clob_token_id: str,
        start: datetime,
        end: datetime,
        fidelity_minutes: int = 60,
    ) -> pd.DataFrame:
        """Return implied-probability time series for one outcome token.

        Columns: ``observed_time`` (tz-aware UTC), ``implied_prob`` (the traded
        price, already in [0, 1] for Polymarket).
        """
        params = {
            "market": clob_token_id,
            "startTs": int(start.replace(tzinfo=UTC).timestamp()),
            "endTs": int(end.replace(tzinfo=UTC).timestamp()),
            "fidelity": fidelity_minutes,
        }
        resp = self.session.get(f"{self.clob_url}/prices-history", params=params, timeout=self.timeout)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        rows = [
            {
                "observed_time": datetime.fromtimestamp(pt["t"], tz=UTC),
                "implied_prob": float(pt["p"]),
            }
            for pt in history
        ]
        return pd.DataFrame(rows, columns=["observed_time", "implied_prob"])

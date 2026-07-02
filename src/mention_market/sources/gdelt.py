"""GDELT context-signal adapter.

We use GDELT to build a *topical relevance* signal: how salient is the subject
of a phrase in the news cycle leading up to an event. The public DOC 2.0 API
returns a volume/tone timeline for a query, which we turn into a timestamped
series that Phase 3 samples strictly before each event.

No auth required. The main constraints are rate limits and the API's 3-month
lookback ceiling on the free timeline endpoint; deeper history needs the BigQuery
GKG dataset (documented, not wired here).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests

DEFAULT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class GdeltClient:
    doc_api_url: str = DEFAULT_DOC_API_URL
    session: requests.Session | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()

    def volume_timeline(self, query: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return a normalized coverage-volume timeline for ``query``.

        Columns: ``observed_time`` (tz-aware UTC), ``volume`` (share of global
        coverage, GDELT's ``VolTimeline`` metric). This becomes a topical
        relevance feature after point-in-time sampling.
        """
        params = {
            "query": query,
            "mode": "TimelineVol",
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        resp = self.session.get(self.doc_api_url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        series = payload.get("timeline", [])
        rows: list[dict] = []
        for block in series:
            for pt in block.get("data", []):
                rows.append(
                    {
                        "observed_time": pd.to_datetime(pt["date"], utc=True),
                        "volume": float(pt["value"]),
                    }
                )
        return pd.DataFrame(rows, columns=["observed_time", "volume"])

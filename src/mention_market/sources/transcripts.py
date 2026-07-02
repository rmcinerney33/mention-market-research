"""Transcript ingestion.

Transcripts are the messiest source (see README). This module defines a common
:class:`TranscriptAdapter` protocol and a C-SPAN-oriented implementation
sketch. Each returned :class:`~mention_market.schema.Transcript` must carry a
*verified* ``event_time`` (when the words were spoken) — that timestamp is the
prediction boundary, so getting it right is non-negotiable.

Rev.com (paid) and Roll Call / Factba.se adapters are intentionally left as
documented stubs; wire them when access is available. Nothing here is required
for the synthetic end-to-end run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import requests

from ..schema import Transcript, VenueType


class TranscriptAdapter(Protocol):
    """Common interface for pulling transcripts for a speaker over a window."""

    def fetch(self, speaker_id: str, start: datetime, end: datetime) -> list[Transcript]: ...


@dataclass
class CSpanAdapter:
    """C-SPAN transcript adapter (sketch).

    C-SPAN publishes program transcripts with air dates. Coverage is uneven and
    the publish timestamp can lag the utterance by hours to days, so we treat
    the *program air time* as ``event_time`` and the crawl time as
    ``publish_time``. Real implementation must scrape/parse the transcript page;
    kept as a documented boundary so the pipeline contract is explicit.
    """

    base_url: str = "https://www.c-span.org"
    session: requests.Session | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()

    def fetch(self, speaker_id: str, start: datetime, end: datetime) -> list[Transcript]:
        raise NotImplementedError(
            "CSpanAdapter.fetch requires live scraping/parsing of C-SPAN transcript "
            "pages. Wire this against real pages when running on live data; the "
            "synthetic generator supplies transcripts for development and tests."
        )


def transcript_from_record(
    *,
    transcript_id: str,
    speaker_id: str,
    event_time: datetime,
    publish_time: datetime,
    venue: VenueType | str,
    text: str,
    source: str,
) -> Transcript:
    """Build a validated :class:`Transcript` from loosely-typed inputs.

    Centralizes construction so every adapter goes through the same validation
    (tz-awareness, publish-after-event ordering) enforced in ``schema.py``.
    """
    venue_enum = VenueType(venue) if not isinstance(venue, VenueType) else venue
    return Transcript(
        transcript_id=transcript_id,
        speaker_id=speaker_id,
        event_time=event_time,
        publish_time=publish_time,
        venue=venue_enum,
        text=text,
        source=source,
    )

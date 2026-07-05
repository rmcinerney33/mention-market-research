"""Parse a Kalshi mention market into a prediction unit.

A mention market asks "Will <speaker> say/mention <phrase> during <occasion>?".
To score it with the research models we need to recover the (speaker, phrase,
event_time) tuple those models are keyed on. This is deliberately conservative:
if a title doesn't clearly parse as a mention market, we return ``None`` and the
signal generator skips it (recording why) rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

_MENTION_RE = re.compile(
    r"^\s*will\s+(?P<speaker>.+?)\s+"
    r"(?:say|says|said|mention|mentions|mentioned|utter|utters|uttered)\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]+)[\"“”'‘’]")
# Where the phrase clause ends when it isn't quoted.
_PHRASE_TAIL_RE = re.compile(
    r"\b(during|in|at|on|before|by|this|next|today|tomorrow|when|while)\b", re.IGNORECASE
)

_VENUE_KEYWORDS = [
    ("state of the union", "formal_address"),
    ("inaugural", "formal_address"),
    ("address", "formal_address"),
    ("speech", "formal_address"),
    ("rally", "rally"),
    ("press", "press_conference"),
    ("debate", "debate"),
    ("interview", "interview"),
]


@dataclass(frozen=True)
class MentionQuery:
    ticker: str
    speaker: str          # human-readable, e.g. "Trump"
    phrase: str           # human-readable, e.g. "tariff"
    speaker_id: str       # slug, e.g. "trump"
    phrase_id: str        # slug, e.g. "tariff"
    event_time: datetime  # resolution time used as the prediction boundary
    venue: str            # inferred VenueType value, e.g. "rally" / "other"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _infer_venue(title: str) -> str:
    t = title.lower()
    for kw, venue in _VENUE_KEYWORDS:
        if kw in t:
            return venue
    return "other"


def _extract_phrase(rest: str) -> str | None:
    quoted = _QUOTED_RE.search(rest)
    if quoted:
        return quoted.group(1).strip()
    # Unquoted: take the words right after the verb, up to a clause boundary.
    text = rest.strip().lstrip(":").strip()
    if not text:
        return None
    cut = _PHRASE_TAIL_RE.search(text)
    phrase = text[: cut.start()] if cut else text
    return phrase.strip(" ?.\"'").strip() or None


def parse_mention_market(snapshot) -> MentionQuery | None:
    """Parse ``snapshot`` (anything with ``title``, ``close_time``, ``ticker``).

    Returns a :class:`MentionQuery`, or ``None`` if the title is not a
    recognizable mention market or has no resolution time to anchor it.
    """
    title = getattr(snapshot, "title", "") or ""
    m = _MENTION_RE.match(title)
    if m is None:
        return None
    if snapshot.close_time is None:
        return None  # no prediction boundary to anchor features

    speaker = m.group("speaker").strip(" ?.\"'").strip()
    phrase = _extract_phrase(m.group("rest"))
    if not speaker or not phrase:
        return None

    return MentionQuery(
        ticker=snapshot.ticker,
        speaker=speaker,
        phrase=phrase,
        speaker_id=_slug(speaker),
        phrase_id=_slug(phrase),
        event_time=snapshot.close_time,
        venue=_infer_venue(title),
    )

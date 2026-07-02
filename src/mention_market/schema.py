"""Core data schema.

Every record that enters the pipeline carries a timezone-aware UTC timestamp.
The whole no-lookahead machinery (``lookahead.py``) depends on this being
enforced, so the constructors here reject naive datetimes loudly.

Three primary record types:

- :class:`Transcript` — a speech/press-conference transcript with a *verified*
  publish/utterance timestamp. ``event_time`` is the moment the words were
  spoken (the prediction boundary ``t``); ``publish_time`` is when the
  transcript became available (may lag, relevant for realism but never used as
  a feature cutoff).
- :class:`MarketSnapshot` — an implied-probability observation for a mention
  contract at a specific time before the event.
- :class:`Event` — the unit of prediction: a (speaker, phrase, occasion) tuple
  with a resolution time and, post-hoc, a binary label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class VenueType(StrEnum):
    """Categorical context in which a speech occurs."""

    RALLY = "rally"
    PRESS_CONFERENCE = "press_conference"
    INTERVIEW = "interview"
    DEBATE = "debate"
    FORMAL_ADDRESS = "formal_address"  # SOTU, inaugural, UN, etc.
    SOCIAL_POST = "social_post"
    OTHER = "other"


def ensure_utc(ts: datetime, *, field_name: str = "timestamp") -> datetime:
    """Return ``ts`` as a timezone-aware UTC datetime, or raise.

    Naive datetimes are rejected outright — silently assuming a timezone is
    exactly the kind of bug that produces subtle lookahead leaks.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(
            f"{field_name} must be timezone-aware; got naive datetime {ts!r}. "
            "Attach an explicit tzinfo (UTC preferred)."
        )
    return ts.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Speaker:
    speaker_id: str
    display_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Phrase:
    """A target phrase plus fuzzy variants used for ground-truth labeling."""

    phrase_id: str
    canonical: str
    variants: tuple[str, ...] = ()

    def all_forms(self) -> tuple[str, ...]:
        return (self.canonical, *self.variants)


@dataclass(frozen=True, slots=True)
class Transcript:
    transcript_id: str
    speaker_id: str
    event_time: datetime  # when spoken — the prediction boundary t
    publish_time: datetime  # when the transcript became available
    venue: VenueType
    text: str
    source: str  # e.g. "cspan", "polymarket_gamma", "synthetic"
    word_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", ensure_utc(self.event_time, field_name="event_time"))
        object.__setattr__(
            self, "publish_time", ensure_utc(self.publish_time, field_name="publish_time")
        )
        if self.publish_time < self.event_time:
            # A transcript cannot be published before the words were spoken.
            raise ValueError(
                f"publish_time {self.publish_time} precedes event_time {self.event_time} "
                f"for transcript {self.transcript_id}"
            )
        if not self.word_count:
            object.__setattr__(self, "word_count", len(self.text.split()))


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Implied probability for a mention contract at a point in time."""

    market_id: str
    speaker_id: str
    phrase_id: str
    event_id: str
    observed_time: datetime  # when this price was observed
    implied_prob: float  # in [0, 1], derived from mid / last trade
    source: str  # "kalshi", "polymarket", "synthetic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_time", ensure_utc(self.observed_time, field_name="observed_time")
        )
        if not (0.0 <= self.implied_prob <= 1.0):
            raise ValueError(
                f"implied_prob must be in [0, 1]; got {self.implied_prob} "
                f"for market {self.market_id}"
            )


@dataclass(frozen=True, slots=True)
class Event:
    """A single prediction unit: will ``phrase_id`` appear when ``speaker_id``
    speaks at ``event_time``?

    ``label`` is ``None`` until resolved (never available before ``event_time``).
    """

    event_id: str
    speaker_id: str
    phrase_id: str
    event_time: datetime
    venue: VenueType
    label: int | None = None  # 1 if phrase used, 0 if not, None if unresolved
    transcript_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", ensure_utc(self.event_time, field_name="event_time"))
        if self.label is not None and self.label not in (0, 1):
            raise ValueError(f"label must be 0, 1, or None; got {self.label!r}")


def now_utc() -> datetime:
    """Current time as tz-aware UTC. Centralized so tests can monkeypatch."""
    return datetime.now(UTC)

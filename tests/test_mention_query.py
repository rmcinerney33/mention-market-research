"""Tests for parsing Kalshi mention-market titles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kalshi_scanner.mention_query import parse_mention_market

CLOSE = datetime(2024, 11, 6, tzinfo=UTC)


@dataclass
class FakeSnap:
    title: str
    ticker: str = "TCK"
    close_time: datetime | None = CLOSE


def test_parses_quoted_phrase():
    q = parse_mention_market(FakeSnap('Will Trump say "tariff" during the Nov 5 rally?'))
    assert q is not None
    assert q.speaker == "Trump" and q.phrase == "tariff"
    assert q.speaker_id == "trump" and q.phrase_id == "tariff"
    assert q.venue == "rally"
    assert q.event_time == CLOSE


def test_parses_unquoted_mention():
    q = parse_mention_market(FakeSnap("Will Biden mention inflation in the address?"))
    assert q is not None
    assert q.speaker == "Biden" and q.phrase == "inflation"
    assert q.venue == "formal_address"


def test_parses_multiword_speaker_and_phrase():
    q = parse_mention_market(FakeSnap('Will Kamala Harris say "middle class" during the debate?'))
    assert q.speaker == "Kamala Harris" and q.speaker_id == "kamala_harris"
    assert q.phrase == "middle class" and q.phrase_id == "middle_class"
    assert q.venue == "debate"


def test_non_mention_title_returns_none():
    assert parse_mention_market(FakeSnap("Will CPI be above 3.5% in November?")) is None


def test_missing_close_time_returns_none():
    assert parse_mention_market(FakeSnap('Will Trump say "tariff"?', close_time=None)) is None


def test_empty_phrase_returns_none():
    assert parse_mention_market(FakeSnap("Will Trump say during the rally?")) is None

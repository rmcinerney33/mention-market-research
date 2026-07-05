"""Tests for the point-in-time feature resolver, focused on no-lookahead."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from kalshi_scanner.feature_resolver import ResearchFeatureResolver
from kalshi_scanner.mention_query import MentionQuery


def _dt(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=UTC)


def _corpus():
    events = pd.DataFrame([
        {"event_id": "e1", "speaker_id": "trump", "phrase_id": "tariff",
         "event_time": _dt(1), "venue": "rally", "label": 1},
        {"event_id": "e2", "speaker_id": "trump", "phrase_id": "tariff",
         "event_time": _dt(2), "venue": "rally", "label": 0},
        {"event_id": "e3", "speaker_id": "trump", "phrase_id": "tariff",
         "event_time": _dt(3), "venue": "rally", "label": 1},
    ])
    transcripts = pd.DataFrame([
        {"event_id": "e1", "publish_time": _dt(1, 1), "word_count": 100},
        {"event_id": "e2", "publish_time": _dt(5), "word_count": 100},   # published LATE
        {"event_id": "e3", "publish_time": _dt(3, 1), "word_count": 100},
    ])
    gdelt = pd.DataFrame([
        {"phrase_id": "tariff", "observed_time": _dt(2), "volume": 0.9},
        {"phrase_id": "tariff", "observed_time": _dt(8), "volume": 0.1},  # after scan
    ])
    return events, transcripts, gdelt


def _query(speaker_id="trump", phrase_id="tariff"):
    return MentionQuery(
        ticker="Q", speaker=speaker_id, phrase=phrase_id,
        speaker_id=speaker_id, phrase_id=phrase_id, event_time=_dt(10), venue="rally",
    )


def test_excludes_events_not_yet_published_at_scan_time():
    resolver = ResearchFeatureResolver(*_corpus())
    # Scan on day 4: e1 & e3 are published, e2's transcript won't publish until day 5.
    row = resolver.resolve(_query(), _dt(4))
    assert row is not None
    # e2 is excluded despite its event_time (day 2) being before the scan.
    assert int(row.iloc[0]["n_prior_events"]) == 2


def test_gdelt_uses_only_pre_scan_observations():
    resolver = ResearchFeatureResolver(*_corpus())
    row = resolver.resolve(_query(), _dt(4))
    # Only the day-2 observation (0.9) is visible; the day-8 one (0.1) is future.
    assert float(row.iloc[0]["topical_relevance"]) == 0.9


def test_more_history_becomes_available_later():
    resolver = ResearchFeatureResolver(*_corpus())
    # By day 6, e2's transcript has published -> all three events are available.
    row = resolver.resolve(_query(), _dt(6))
    assert int(row.iloc[0]["n_prior_events"]) == 3


def test_cold_start_unknown_pair_returns_valid_row():
    resolver = ResearchFeatureResolver(*_corpus())
    row = resolver.resolve(_query(speaker_id="biden", phrase_id="unheardphrase"), _dt(4))
    assert row is not None
    assert int(row.iloc[0]["n_prior_events"]) == 0
    assert float(row.iloc[0]["topical_relevance"]) == 0.5  # no GDELT -> neutral prior


def test_returns_single_row_with_feature_columns():
    resolver = ResearchFeatureResolver(*_corpus())
    row = resolver.resolve(_query(), _dt(4))
    assert len(row) == 1
    for col in ("lap_smoothed_rate", "rr_hl30", "topical_relevance", "venue"):
        assert col in row.columns

"""Tests for feature engineering, with emphasis on point-in-time correctness.

The headline test (`test_publish_time_availability_excludes_unpublished_label`)
proves the subtle guarantee: a prior event whose transcript is not yet public
by the current event time must NOT contribute to features under the default
publish-time availability rule.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mention_market.features import build_feature_table
from mention_market.schema import Transcript, VenueType


def _mk_events(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _mk_transcript(event_id, event_time, publish_time, wc=100) -> Transcript:
    return Transcript(
        transcript_id=event_id,
        speaker_id="s1",
        event_time=event_time,
        publish_time=publish_time,
        venue=VenueType.RALLY,
        text="x " * wc,
        source="test",
    )


def _empty_gdelt() -> pd.DataFrame:
    return pd.DataFrame(columns=["phrase_id", "observed_time", "volume"])


def test_basic_shape_and_cold_start():
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    events = _mk_events(
        [
            {"event_id": "e0", "speaker_id": "s1", "phrase_id": "p1",
             "event_time": t0, "venue": "rally", "label": 1},
        ]
    )
    transcripts = [_mk_transcript("e0", t0, t0 + timedelta(hours=2))]
    feats = build_feature_table(events, transcripts, _empty_gdelt())

    assert len(feats) == 1
    row = feats.iloc[0]
    # Cold start: no prior history.
    assert row["n_prior_events"] == 0
    assert row["ever_used_before"] == 0
    assert row["lap_smoothed_rate"] == pytest.approx(0.5)
    assert np.isnan(row["days_since_last_use"])
    assert row["rr_hl30"] == pytest.approx(0.5)


def test_recency_and_time_since_last_use_hand_computed():
    # Three prior uses at known offsets, all published quickly, then a query event.
    base = datetime(2024, 1, 1, 12, tzinfo=UTC)
    times = [base, base + timedelta(days=10), base + timedelta(days=20), base + timedelta(days=30)]
    labels = [1, 0, 1, None]  # last is the query event
    events = _mk_events(
        [
            {"event_id": f"e{i}", "speaker_id": "s1", "phrase_id": "p1",
             "event_time": times[i], "venue": "rally", "label": labels[i]}
            for i in range(4)
        ]
    )
    transcripts = [_mk_transcript(f"e{i}", times[i], times[i] + timedelta(hours=1)) for i in range(4)]
    feats = build_feature_table(events, transcripts, _empty_gdelt()).set_index("event_id")

    q = feats.loc["e3"]
    assert q["n_prior_events"] == 3
    assert q["ever_used_before"] == 1
    # Last use was e2 at day 20; query at day 30 => 10 days.
    assert q["days_since_last_use"] == pytest.approx(10.0, abs=1e-6)
    # Laplace: (2 uses + 0.5) / (3 + 1) = 0.625
    assert q["lap_smoothed_rate"] == pytest.approx(0.625)
    # Recency-weighted rate must lie in [0, 1] and weight recent e2(use) heavily.
    assert 0.0 <= q["rr_hl7"] <= 1.0


def test_publish_time_availability_excludes_unpublished_label():
    """The crux: a prior event whose transcript publishes AFTER the current
    event time must not leak into features under the default 'publish' rule,
    but does appear under the looser 'event' rule."""
    t_prev = datetime(2024, 1, 1, 12, tzinfo=UTC)
    t_cur = t_prev + timedelta(hours=4)           # events only 4h apart
    publish_prev = t_prev + timedelta(hours=10)   # transcript public AFTER t_cur

    events = _mk_events(
        [
            {"event_id": "prev", "speaker_id": "s1", "phrase_id": "p1",
             "event_time": t_prev, "venue": "rally", "label": 1},
            {"event_id": "cur", "speaker_id": "s1", "phrase_id": "p1",
             "event_time": t_cur, "venue": "rally", "label": 0},
        ]
    )
    transcripts = [
        _mk_transcript("prev", t_prev, publish_prev),
        _mk_transcript("cur", t_cur, t_cur + timedelta(hours=2)),
    ]

    strict = build_feature_table(events, transcripts, _empty_gdelt(),
                                 label_availability="publish").set_index("event_id")
    loose = build_feature_table(events, transcripts, _empty_gdelt(),
                                label_availability="event").set_index("event_id")

    # Under the rigorous rule, 'prev' is not yet public => no usable history.
    assert strict.loc["cur", "n_prior_events"] == 0
    assert strict.loc["cur", "ever_used_before"] == 0

    # Under the optimistic rule, 'prev' counts as soon as it occurs.
    assert loose.loc["cur", "n_prior_events"] == 1
    assert loose.loc["cur", "ever_used_before"] == 1


def test_topical_relevance_is_point_in_time():
    t0 = datetime(2024, 2, 1, 12, tzinfo=UTC)
    events = _mk_events(
        [
            {"event_id": "e0", "speaker_id": "s1", "phrase_id": "p1",
             "event_time": t0, "venue": "rally", "label": 0},
        ]
    )
    transcripts = [_mk_transcript("e0", t0, t0 + timedelta(hours=1))]
    gdelt = pd.DataFrame(
        {
            "phrase_id": ["p1", "p1", "p1"],
            "observed_time": [
                t0 - timedelta(days=2),
                t0 - timedelta(hours=1),   # most recent BEFORE event
                t0 + timedelta(days=1),    # FUTURE — must be ignored
            ],
            "volume": [0.2, 0.7, 0.99],
        }
    )
    feats = build_feature_table(events, transcripts, gdelt)
    # Must pick the 0.7 (most recent prior), never the future 0.99.
    assert feats.iloc[0]["topical_relevance"] == pytest.approx(0.7)


def test_end_to_end_on_synthetic_no_leak():
    """Building features over the whole synthetic world must not raise the
    lookahead audit for any row."""
    from mention_market.synthetic import SyntheticParams, generate

    world = generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2,
                                     events_per_pair=30, seed=11))
    feats = build_feature_table(
        world.events, world.transcripts, world.gdelt, audit=True
    )
    assert len(feats) == len(world.events)
    # Labels preserved and binary on resolved events.
    assert set(feats["label"].dropna().unique()) <= {0, 1}

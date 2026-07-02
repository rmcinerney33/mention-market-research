"""Tests for the synthetic world generator.

We check structural integrity and, importantly, self-consistency between the
generated transcripts and the labels (the labeler run on a transcript must
reproduce the planted label), plus that market snapshots sit strictly before
their events (so they are legitimately usable as pre-event prices).
"""

from __future__ import annotations

import pandas as pd
import pytest

from mention_market.labeling import phrase_appears
from mention_market.schema import Event
from mention_market.synthetic import (
    SyntheticParams,
    build_event_objects,
    generate,
)


@pytest.fixture(scope="module")
def world():
    return generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2, events_per_pair=40, seed=7))


def test_shapes_are_consistent(world):
    n_pairs = 2 * 2
    assert len(world.events) == n_pairs * 40
    # one transcript per event
    assert len(world.transcripts) == len(world.events)
    # one snapshot per (event, lead_time)
    assert len(world.market_snapshots) == len(world.events) * len(world.params.lead_times_hours)


def test_determinism(world):
    again = generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2, events_per_pair=40, seed=7))
    pd.testing.assert_frame_equal(world.events, again.events)


def test_labels_match_transcripts(world):
    """The planted label must equal what the labeler extracts from the text."""
    phrase_by_id = {p.phrase_id: p for plist in world.phrases.values() for p in plist}
    by_event = {t.transcript_id: t for t in world.transcripts}
    mismatches = 0
    for row in world.events.itertuples(index=False):
        transcript = by_event[row.event_id]
        phrase = phrase_by_id[row.phrase_id]
        extracted = int(phrase_appears(transcript.text, phrase))
        mismatches += int(extracted != row.label)
    assert mismatches == 0


def test_snapshots_strictly_before_events(world):
    ev_time = world.events.set_index("event_id")["event_time"]
    snaps = world.market_snapshots.copy()
    snaps["event_time"] = snaps["event_id"].map(ev_time)
    assert (snaps["observed_time"] < snaps["event_time"]).all()


def test_labels_are_binary(world):
    assert set(world.events["label"].unique()) <= {0, 1}


def test_efficient_market_tracks_truth_better_than_inefficient():
    """Sanity check on the efficiency knob: an efficient market's implied prob
    should correlate with p_true more tightly than an inefficient one's."""
    eff = generate(SyntheticParams(events_per_pair=60, market_efficiency=1.0, seed=3))
    ineff = generate(SyntheticParams(events_per_pair=60, market_efficiency=0.2, seed=3))

    def corr_with_truth(w):
        # Use the 1h snapshot (sharpest).
        s = w.market_snapshots[w.market_snapshots["lead_time_hours"] == 1]
        merged = s.merge(w.events[["event_id", "p_true"]], on="event_id")
        return merged["implied_prob"].corr(merged["p_true"])

    assert corr_with_truth(eff) > corr_with_truth(ineff)


def test_build_event_objects(world):
    events = build_event_objects(world)
    assert len(events) == len(world.events)
    assert all(isinstance(e, Event) for e in events)
    assert all(e.label in (0, 1) for e in events)

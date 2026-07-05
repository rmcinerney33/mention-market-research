"""Resolve a live market into a point-in-time feature row.

Given a :class:`MentionQuery` and the ``scan_ts`` at which the market was
observed, we build the exact feature row the research models expect — by
appending the query as a new event to that speaker/phrase's history and running
the research feature builder. The no-lookahead guarantee is enforced two ways:

1. The history is pre-filtered to events whose information was *available* before
   ``scan_ts`` (a prior label counts only once its transcript was published), and
   GDELT is filtered to observations before ``scan_ts``.
2. ``build_feature_table(..., audit=True)`` independently asserts every source
   timestamp feeding the row is strictly before the event time.

Because ``scan_ts <= event_time`` (we scan before resolution), anything available
before ``scan_ts`` is automatically available before the event — so the row uses
strictly pre-scan information.

The training corpus here is the research repo's **synthetic** world (all that
exists today). When real transcript/GDELT/usage ingestion is wired, swap the
corpus passed to :class:`ResearchFeatureResolver` and nothing else changes.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from mention_market.features import build_feature_table

from .mention_query import MentionQuery

_EVENT_COLS = ["event_id", "speaker_id", "phrase_id", "event_time", "venue", "label"]


def _transcripts_frame(transcripts) -> pd.DataFrame:
    if isinstance(transcripts, pd.DataFrame):
        df = transcripts.copy()
        if "event_id" not in df.columns and "transcript_id" in df.columns:
            df = df.rename(columns={"transcript_id": "event_id"})
        keep = [c for c in ("event_id", "publish_time", "word_count") if c in df.columns]
        df = df[keep]
    else:
        df = pd.DataFrame(
            [
                {
                    "event_id": t.transcript_id,
                    "publish_time": pd.Timestamp(t.publish_time),
                    "word_count": t.word_count,
                }
                for t in transcripts
            ]
        )
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True)
    return df


class ResearchFeatureResolver:
    def __init__(self, events, transcripts, gdelt, *, label_availability: str = "publish") -> None:
        ev = events.copy()
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        tdf = _transcripts_frame(transcripts)

        merged = ev.merge(tdf, on="event_id", how="left") if "event_id" in tdf.columns else ev.copy()
        if "publish_time" in merged.columns and label_availability == "publish":
            merged["available_time"] = merged["publish_time"].fillna(merged["event_time"])
        else:
            merged["available_time"] = merged["event_time"]

        self._merged = merged
        self._tdf = tdf
        self._gdelt = gdelt.copy()
        self._gdelt["observed_time"] = pd.to_datetime(self._gdelt["observed_time"], utc=True)
        self._label_availability = label_availability

    def resolve(self, query: MentionQuery, scan_ts: datetime) -> pd.DataFrame | None:
        """Return a one-row feature frame for ``query`` as of ``scan_ts``.

        Uses only information available strictly before ``scan_ts``. Unknown
        speaker/phrase pairs are not an error — they yield a valid cold-start row
        (prior rates default to 0.5, ``n_prior_events=0``).
        """
        m = self._merged
        hist = m[
            (m["speaker_id"] == query.speaker_id)
            & (m["phrase_id"] == query.phrase_id)
            & (m["available_time"] < scan_ts)
            & (m["event_time"] < query.event_time)
        ]

        query_event = {
            "event_id": f"QUERY::{query.ticker}",
            "speaker_id": query.speaker_id,
            "phrase_id": query.phrase_id,
            "event_time": query.event_time,
            "venue": query.venue,
            "label": np.nan,
        }
        events = pd.concat(
            [hist[_EVENT_COLS], pd.DataFrame([query_event])], ignore_index=True
        )

        tdf = self._tdf[self._tdf["event_id"].isin(hist["event_id"])] if "event_id" in self._tdf.columns else self._tdf.iloc[0:0]
        gdelt = self._gdelt[
            (self._gdelt["phrase_id"] == query.phrase_id)
            & (self._gdelt["observed_time"] < scan_ts)
        ]

        table = build_feature_table(
            events, tdf, gdelt, label_availability=self._label_availability, audit=True
        )
        row = table[table["event_id"] == query_event["event_id"]]
        return row.reset_index(drop=True) if not row.empty else None


def default_research_world(*, seed: int = 42, events_per_pair: int = 80):
    """The synthetic research world used as the (placeholder) training corpus."""
    from mention_market.synthetic import SyntheticParams, generate

    return generate(SyntheticParams(seed=seed, events_per_pair=events_per_pair))


def build_training_table(world) -> pd.DataFrame:
    return build_feature_table(world.events, world.transcripts, world.gdelt, audit=False)

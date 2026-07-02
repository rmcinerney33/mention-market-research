"""Feature engineering.

Builds a feature table keyed by ``(speaker_id, phrase_id, event_id)``. Every
feature for an event at time ``t`` is constructed from data strictly before
``t`` and, crucially, only from information that was *actually available* by
``t``.

The subtle point — and a genuine source of lookahead leaks — is label
availability. A prior event's outcome (did the phrase appear?) is only knowable
once that event's transcript has been *published*, which can lag the utterance
by hours. So when we compute "recency-weighted usage rate" or "time since last
use", a prior event counts only if its transcript ``publish_time < t``, not
merely its ``event_time < t``. This is controlled by ``label_availability``:

- ``"publish"`` (default, rigorous): a prior label is usable only once its
  transcript is public.
- ``"event"`` (optimistic): a prior label is usable as soon as the event
  occurs. Provided for sensitivity analysis; documented as looser.

Features produced (all point-in-time):

- ``lap_smoothed_rate``      Laplace-smoothed prior usage rate (cold-start = 0.5)
- ``rr_hl{H}``               recency-weighted usage rate, exp half-life H days
- ``use_count_{W}``          uses among the last W available prior events
- ``use_rate_{W}``           mean label among the last W available prior events
- ``days_since_last_use``    days since last known use (NaN if never)
- ``ever_used_before``       1 if any known prior use else 0
- ``n_prior_events``         count of available prior events (control)
- ``topical_relevance``      GDELT salience as-of t (0.5 if none)
- ``expected_speech_len``    point-in-time mean word count of prior transcripts
- ``venue``                  categorical context (encoded downstream)
- ``dow`` / ``hour`` / ``is_weekend``   schedule-known calendar features
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .lookahead import audit_feature_row
from .schema import Transcript

LAPLACE_ALPHA = 0.5


def _transcripts_to_frame(transcripts: Iterable[Transcript]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": t.transcript_id,
                "publish_time": pd.Timestamp(t.publish_time),
                "word_count": t.word_count,
            }
            for t in transcripts
        ]
    )


def build_feature_table(
    events: pd.DataFrame,
    transcripts: Iterable[Transcript] | pd.DataFrame,
    gdelt: pd.DataFrame,
    *,
    halflives_days: list[float] | None = None,
    rolling_windows: list[int] | None = None,
    label_availability: str = "publish",
    audit: bool = True,
) -> pd.DataFrame:
    """Construct the point-in-time feature table.

    Parameters
    ----------
    events:
        Columns: event_id, speaker_id, phrase_id, event_time (tz-aware),
        venue, label. Extra columns (e.g. diagnostics) are ignored.
    transcripts:
        Iterable of :class:`Transcript` or a DataFrame with columns
        event_id, publish_time, word_count.
    gdelt:
        Columns: phrase_id, observed_time (tz-aware), volume.
    label_availability:
        ``"publish"`` (default) or ``"event"`` — see module docstring.
    audit:
        If True, assert every row's source timestamps are strictly before the
        event time (defense in depth on top of the point-in-time construction).
    """
    if label_availability not in ("publish", "event"):
        raise ValueError("label_availability must be 'publish' or 'event'")
    halflives_days = halflives_days or [7.0, 30.0, 90.0]
    rolling_windows = rolling_windows or [5, 10, 20]

    ev = events.copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)

    tdf = transcripts if isinstance(transcripts, pd.DataFrame) else _transcripts_to_frame(transcripts)
    tdf = tdf.copy()
    tdf["publish_time"] = pd.to_datetime(tdf["publish_time"], utc=True)
    ev = ev.merge(tdf[["event_id", "publish_time", "word_count"]], on="event_id", how="left")

    # The moment a prior event's label/word_count becomes usable.
    if label_availability == "publish":
        ev["available_time"] = ev["publish_time"].fillna(ev["event_time"])
    else:
        ev["available_time"] = ev["event_time"]

    gd = gdelt.copy()
    gd["observed_time"] = pd.to_datetime(gd["observed_time"], utc=True)
    gd = gd.sort_values("observed_time")
    gdelt_by_phrase = {pid: g.reset_index(drop=True) for pid, g in gd.groupby("phrase_id")}

    rows: list[dict] = []
    for (speaker_id, phrase_id), grp in ev.groupby(["speaker_id", "phrase_id"], sort=False):
        grp = grp.sort_values("event_time").reset_index(drop=True)
        gph = gdelt_by_phrase.get(phrase_id)

        for i in range(len(grp)):
            cur = grp.iloc[i]
            t = cur["event_time"]

            # Prior events whose LABEL is available by t (publish- or event-based).
            prior = grp.iloc[:i]
            prior_avail = prior[prior["available_time"] < t]

            feat = _base_calendar_features(t)
            feat.update(
                {
                    "event_id": cur["event_id"],
                    "speaker_id": speaker_id,
                    "phrase_id": phrase_id,
                    "event_time": t,
                    "venue": cur["venue"],
                    "label": cur.get("label"),
                    "n_prior_events": len(prior_avail),
                }
            )
            feat.update(_usage_features(prior_avail, t, halflives_days, rolling_windows))
            feat["topical_relevance"] = _topical_as_of(gph, t)
            feat["expected_speech_len"] = (
                float(prior_avail["word_count"].mean()) if len(prior_avail) else np.nan
            )
            rows.append(feat)

            if audit:
                # Every source timestamp that fed this row must be < t.
                src = {"prior_labels": list(prior_avail["available_time"])}
                if gph is not None:
                    used = gph[gph["observed_time"] < t]["observed_time"]
                    src["gdelt"] = list(used)
                audit_feature_row(t, src)

    out = pd.DataFrame(rows)
    # Stable column order: keys first, then features.
    key_cols = ["event_id", "speaker_id", "phrase_id", "event_time", "venue", "label"]
    feat_cols = [c for c in out.columns if c not in key_cols]
    return out[key_cols + feat_cols]


def _base_calendar_features(t: pd.Timestamp) -> dict:
    """Calendar features known when the event is *scheduled* (not a leak)."""
    dow = int(t.dayofweek)
    return {"dow": dow, "hour": int(t.hour), "is_weekend": int(dow >= 5)}


def _usage_features(
    prior_avail: pd.DataFrame,
    t: pd.Timestamp,
    halflives_days: list[float],
    rolling_windows: list[int],
) -> dict:
    """Recency-weighted and rolling usage features from available prior events."""
    feat: dict = {}
    n = len(prior_avail)
    labels = prior_avail["label"].to_numpy(dtype=float) if n else np.array([])

    # Laplace-smoothed overall prior rate (cold-start -> 0.5).
    uses = float(labels.sum()) if n else 0.0
    feat["lap_smoothed_rate"] = (uses + LAPLACE_ALPHA) / (n + 2 * LAPLACE_ALPHA)

    # Recency-weighted rate at each half-life (weight by event recency).
    if n:
        ages_days = (t - prior_avail["event_time"]).dt.total_seconds().to_numpy() / 86400.0
    else:
        ages_days = np.array([])
    for H in halflives_days:
        key = f"rr_hl{int(H)}"
        if n:
            lam = np.log(2) / H
            w = np.exp(-lam * ages_days)
            feat[key] = float(np.sum(w * labels) / np.sum(w))
        else:
            feat[key] = 0.5  # cold-start prior

    # Rolling counts/rates over the last W available events.
    for W in rolling_windows:
        recent = labels[-W:] if n else np.array([])
        feat[f"use_count_{W}"] = float(recent.sum()) if recent.size else 0.0
        feat[f"use_rate_{W}"] = float(recent.mean()) if recent.size else 0.5

    # Time since last known use.
    if n and labels.sum() > 0:
        used_times = prior_avail.loc[prior_avail["label"] == 1, "event_time"]
        last_use = used_times.max()
        feat["days_since_last_use"] = (t - last_use).total_seconds() / 86400.0
        feat["ever_used_before"] = 1
    else:
        feat["days_since_last_use"] = np.nan
        feat["ever_used_before"] = 0

    return feat


def _topical_as_of(gph: pd.DataFrame | None, t: pd.Timestamp) -> float:
    """Most recent GDELT salience strictly before ``t`` (0.5 if none)."""
    if gph is None:
        return 0.5
    prior = gph[gph["observed_time"] < t]
    if prior.empty:
        return 0.5
    return float(prior.iloc[-1]["volume"])


# Columns the model layer treats as numeric predictors (venue handled separately).
NUMERIC_FEATURES = [
    "lap_smoothed_rate",
    "rr_hl7",
    "rr_hl30",
    "rr_hl90",
    "use_count_5",
    "use_count_10",
    "use_count_20",
    "use_rate_5",
    "use_rate_10",
    "use_rate_20",
    "days_since_last_use",
    "ever_used_before",
    "n_prior_events",
    "topical_relevance",
    "expected_speech_len",
    "dow",
    "hour",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["venue"]

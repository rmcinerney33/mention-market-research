"""Synthetic data generator with known ground truth.

Purpose: exercise the *entire* analysis stack — features, all six models,
walk-forward validation, and the Mincer-Zarnowitz market-efficiency test — end
to end, without depending on live API access or historical-data availability.
Because we control the data-generating process, we can also *validate* the
pipeline itself: models should recover the structure we planted, and the
no-lookahead machinery can be stress-tested against a fully known timeline.

Generative model (per speaker x phrase, events in time order):

    usage_decay_t  = exp-decayed count of prior *uses* (half-life H)
    topical_t      = news-cycle salience for the phrase at t (from a latent
                     GDELT-like series), in [0, 1]
    logit_t        = a0 + a_usage*(usage_decay_t - mean) + a_topical*(topical_t - 0.5)
                     + venue_effect[venue]
    p_true_t       = sigmoid(logit_t)
    label_t        ~ Bernoulli(p_true_t)

The market observes a version of ``p_true`` controlled by ``market_efficiency``:

    market_mean = e * p_true + (1 - e) * base_rate
    implied_L   = clip( market_mean + Normal(0, noise0 * (1 + L_hours/24)) )

- ``market_efficiency = 1`` → market ≈ p_true (noisily). A model built on the
  *same* information cannot beat it; the MZ test should drive the model
  coefficient toward 0. This is the honest, expected "market is efficient"
  headline.
- ``market_efficiency < 1`` → the market shrinks toward the base rate and
  ignores the per-event signal that the model's features capture; the MZ test
  should then find a surviving model coefficient (detectable edge). This lets
  us confirm the test has power, not just that it fails to reject.

Nothing about ``p_true`` is ever exposed to features or models — only labels,
transcripts, market snapshots, and the GDELT series are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from .schema import Event, Phrase, Speaker, Transcript, VenueType

_VENUES = [
    VenueType.RALLY,
    VenueType.PRESS_CONFERENCE,
    VenueType.INTERVIEW,
    VenueType.FORMAL_ADDRESS,
]
_VENUE_EFFECT = {
    VenueType.RALLY: 0.8,          # off-the-cuff, phrase-heavy
    VenueType.PRESS_CONFERENCE: 0.1,
    VenueType.INTERVIEW: 0.3,
    VenueType.FORMAL_ADDRESS: -0.9,  # scripted, phrase-light
}

_FILLER = ["we", "are", "going", "to", "talk", "about", "the", "economy", "and", "the", "border", "and", "jobs", "and", "trade", "and", "our", "great", "country", "and", "the", "people", "who", "work", "so", "hard", "every", "single", "day"]


@dataclass
class SyntheticParams:
    n_speakers: int = 2
    phrases_per_speaker: int = 3
    events_per_pair: int = 80
    start: datetime = field(
        default_factory=lambda: datetime(2023, 1, 1, tzinfo=UTC)
    )
    mean_days_between_events: float = 3.5
    lead_times_hours: tuple[int, ...] = (24, 6, 1)
    market_efficiency: float = 1.0   # 1.0 => efficient market (expected headline)
    market_noise0: float = 0.05
    usage_halflife_days: float = 30.0
    a0: float = -0.4                 # base logit
    a_usage: float = 2.5
    a_topical: float = 1.8
    seed: int = 42


@dataclass
class SyntheticWorld:
    speakers: list[Speaker]
    phrases: dict[str, list[Phrase]]  # speaker_id -> phrases
    events: pd.DataFrame              # one row per (speaker, phrase, event)
    transcripts: list[Transcript]
    market_snapshots: pd.DataFrame    # one row per (event, lead_time)
    gdelt: pd.DataFrame               # topical volume series per phrase
    params: SyntheticParams


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_gdelt_series(
    rng: np.random.Generator, phrase_id: str, start: datetime, days: int
) -> pd.DataFrame:
    """A smooth-ish daily salience series in [0, 1] for one phrase."""
    n = days
    # AR(1)-style walk mapped through a logistic to stay in (0, 1).
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.9 * z[i - 1] + rng.normal(0, 0.5)
    vol = _sigmoid(z - z.mean())
    times = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {"phrase_id": phrase_id, "observed_time": pd.to_datetime(times, utc=True), "volume": vol}
    )


def _topical_at(gdelt_phrase: pd.DataFrame, when: datetime) -> float:
    """Most recent salience value strictly before ``when`` (point-in-time)."""
    prior = gdelt_phrase[gdelt_phrase["observed_time"] < pd.Timestamp(when)]
    if prior.empty:
        return 0.5
    return float(prior.iloc[-1]["volume"])


def _make_text(rng: np.random.Generator, phrase_text: str, label: int) -> str:
    """Build filler transcript text, embedding the phrase iff label == 1."""
    words = list(rng.choice(_FILLER, size=rng.integers(60, 140), replace=True))
    if label == 1:
        pos = int(rng.integers(0, len(words)))
        words[pos:pos] = phrase_text.split()
    return " ".join(words)


def generate(params: SyntheticParams | None = None) -> SyntheticWorld:
    """Generate a full synthetic world. Deterministic given ``params.seed``."""
    p = params or SyntheticParams()
    rng = np.random.default_rng(p.seed)

    speakers = [Speaker(f"speaker_{i}", f"Speaker {i}") for i in range(p.n_speakers)]
    phrases: dict[str, list[Phrase]] = {}
    phrase_texts = [
        "witch hunt", "fake news", "total disaster", "very strongly",
        "believe me", "many people", "tremendous success", "radical left",
    ]
    horizon_days = int(p.events_per_pair * p.mean_days_between_events) + 30

    gdelt_frames: list[pd.DataFrame] = []
    event_rows: list[dict] = []
    transcripts: list[Transcript] = []
    snap_rows: list[dict] = []

    tpi = 0
    for sp in speakers:
        sp_phrases = []
        for k in range(p.phrases_per_speaker):
            text = phrase_texts[(tpi) % len(phrase_texts)]
            tpi += 1
            phrase = Phrase(phrase_id=f"{sp.speaker_id}__ph{k}", canonical=text)
            sp_phrases.append(phrase)

            gdelt_phrase = _make_gdelt_series(rng, phrase.phrase_id, p.start, horizon_days)
            gdelt_frames.append(gdelt_phrase)

            # Latent per-pair base rate for the market's inefficient fallback.
            base_rate = float(_sigmoid(p.a0 + rng.normal(0, 0.3)))

            # Walk events forward in time, tracking recency-weighted usage.
            t = p.start + timedelta(days=float(rng.uniform(0, 5)))
            usage_decay = 0.0
            last_decay_time = t
            decay_lambda = np.log(2) / p.usage_halflife_days  # per day
            for e in range(p.events_per_pair):
                gap_days = float(rng.exponential(p.mean_days_between_events)) + 0.25
                t = t + timedelta(days=gap_days)
                venue = _VENUES[int(rng.integers(0, len(_VENUES)))]

                # Decay accumulated usage to now.
                dt_days = (t - last_decay_time).total_seconds() / 86400.0
                usage_decay *= float(np.exp(-decay_lambda * dt_days))
                last_decay_time = t

                topical = _topical_at(gdelt_phrase, t)
                logit = (
                    p.a0
                    + p.a_usage * (usage_decay - 0.5)
                    + p.a_topical * (topical - 0.5)
                    + _VENUE_EFFECT[venue]
                )
                p_true = float(_sigmoid(logit))
                label = int(rng.random() < p_true)

                event_id = f"{phrase.phrase_id}__ev{e}"
                event_rows.append(
                    {
                        "event_id": event_id,
                        "speaker_id": sp.speaker_id,
                        "phrase_id": phrase.phrase_id,
                        "event_time": pd.Timestamp(t),
                        "venue": venue.value,
                        "label": label,
                        # Diagnostics only — NOT features:
                        "p_true": p_true,
                        "usage_decay_true": usage_decay,
                        "topical_true": topical,
                    }
                )

                transcripts.append(
                    Transcript(
                        transcript_id=event_id,
                        speaker_id=sp.speaker_id,
                        event_time=t,
                        publish_time=t + timedelta(hours=float(rng.uniform(1, 12))),
                        venue=venue,
                        text=_make_text(rng, phrase.canonical, label),
                        source="synthetic",
                    )
                )

                # Market snapshots at each lead time.
                market_mean = p.market_efficiency * p_true + (1 - p.market_efficiency) * base_rate
                for L in p.lead_times_hours:
                    sd = p.market_noise0 * (1 + L / 24.0)
                    implied = float(np.clip(market_mean + rng.normal(0, sd), 0.01, 0.99))
                    snap_rows.append(
                        {
                            "market_id": f"mkt_{event_id}",
                            "speaker_id": sp.speaker_id,
                            "phrase_id": phrase.phrase_id,
                            "event_id": event_id,
                            "lead_time_hours": L,
                            "observed_time": pd.Timestamp(t) - pd.Timedelta(hours=L),
                            "implied_prob": implied,
                            "source": "synthetic",
                        }
                    )

                # Update recency-weighted usage if the phrase was actually used.
                if label == 1:
                    usage_decay += 1.0

        phrases[sp.speaker_id] = sp_phrases

    events = pd.DataFrame(event_rows).sort_values("event_time").reset_index(drop=True)
    snaps = pd.DataFrame(snap_rows).sort_values("observed_time").reset_index(drop=True)
    gdelt = pd.concat(gdelt_frames, ignore_index=True)

    return SyntheticWorld(
        speakers=speakers,
        phrases=phrases,
        events=events,
        transcripts=transcripts,
        market_snapshots=snaps,
        gdelt=gdelt,
        params=p,
    )


def build_event_objects(world: SyntheticWorld) -> list[Event]:
    """Materialize schema ``Event`` objects from the events frame."""
    out = []
    for row in world.events.itertuples(index=False):
        out.append(
            Event(
                event_id=row.event_id,
                speaker_id=row.speaker_id,
                phrase_id=row.phrase_id,
                event_time=row.event_time.to_pydatetime(),
                venue=VenueType(row.venue),
                label=int(row.label),
                transcript_id=row.event_id,
            )
        )
    return out

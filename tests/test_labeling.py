"""Tests for ground-truth phrase labeling.

Robust to transcriber variation (case, punctuation, curly quotes, hyphenation,
spacing) without matching unrelated text (word-boundary anchoring).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mention_market.labeling import label_transcript, normalize, phrase_appears
from mention_market.schema import Phrase, Transcript, VenueType


def _transcript(text: str) -> Transcript:
    t = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
    return Transcript(
        transcript_id="t1",
        speaker_id="s1",
        event_time=t,
        publish_time=t + timedelta(hours=2),
        venue=VenueType.RALLY,
        text=text,
        source="test",
    )


def test_normalize_basic():
    assert normalize("  Fake, NEWS!!  ") == "fake news"
    assert normalize("witch—hunt") == "witch hunt"  # em dash -> space


WITCH_HUNT = Phrase("ph", "witch hunt")


@pytest.mark.parametrize(
    "text",
    [
        "It is a witch hunt, believe me.",
        "It is a WITCH HUNT!",
        "the greatest   witch\nhunt in history",   # odd spacing / newline
        "a so-called “witch hunt” they said",       # curly quotes
        "witch hunt",                               # exact, no surrounding text
    ],
)
def test_matches_variants(text):
    assert phrase_appears(text, WITCH_HUNT) is True


@pytest.mark.parametrize(
    "text",
    [
        "there was no hunt at all",
        "witchhunt as one word should not count",   # no boundary
        "the witch went hunting yesterday",
        "",
    ],
)
def test_does_not_falsely_match(text):
    assert phrase_appears(text, WITCH_HUNT) is False


def test_word_boundary_prevents_prefix_match():
    fake_news = Phrase("fn", "fake news")
    assert phrase_appears("this is fake newsletter content", fake_news) is False
    assert phrase_appears("this is fake news content", fake_news) is True


def test_explicit_variant_matches():
    p = Phrase("wh", "witch hunt", variants=("witch-hunt",))
    # The hyphenated form normalizes to "witch hunt" anyway, but variants also
    # cover genuinely different spellings the canonical wouldn't catch.
    assert phrase_appears("a witch-hunt indeed", p) is True

    p2 = Phrase("cov", "coronavirus", variants=("covid", "covid 19"))
    assert phrase_appears("we beat COVID-19 together", p2) is True
    assert phrase_appears("the coronavirus response", p2) is True
    assert phrase_appears("a viral video", p2) is False


def test_label_transcript_reports_matched_form():
    p = Phrase("p", "fake news", variants=("phony news",))
    r_pos = label_transcript(_transcript("what phony news this is"), p)
    assert r_pos.label == 1
    assert r_pos.matched_form == "phony news"

    r_neg = label_transcript(_transcript("nothing to see here"), p)
    assert r_neg.label == 0
    assert r_neg.matched_form is None

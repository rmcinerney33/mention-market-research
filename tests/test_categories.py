"""Tests for category matching and the validated gate."""

from __future__ import annotations

from kalshi_scanner.categories import CategoryMatcher
from kalshi_scanner.config import CategoryRule


def _matcher():
    return CategoryMatcher([
        CategoryRule(
            name="mention", validated=False,
            title_patterns=[r"\bsay\b", r"\bmention"],
            series_tickers=["MENTIONSPEECH"], kalshi_categories=[],
        ),
        CategoryRule(
            name="econ", validated=True,
            title_patterns=[], series_tickers=[], kalshi_categories=["Economics"],
        ),
    ])


def test_matches_by_title():
    assert _matcher().match({"title": 'Will Trump say "X"?', "event_ticker": "FOO-1"}) == "mention"


def test_matches_by_series_ticker():
    assert _matcher().match({"title": "no keywords", "series_ticker": "MENTIONSPEECH"}) == "mention"


def test_matches_by_series_derived_from_event_ticker():
    assert _matcher().match({"title": "no keywords", "event_ticker": "MENTIONSPEECH-24NOV"}) == "mention"


def test_matches_by_kalshi_category():
    assert _matcher().match(
        {"title": "CPI above 3.5%?", "category": "Economics", "event_ticker": "CPI-1"}
    ) == "econ"


def test_no_match_returns_none():
    assert _matcher().match({"title": "unrelated question", "event_ticker": "ZZZ-1"}) is None


def test_is_validated_reflects_config():
    m = _matcher()
    assert m.is_validated("econ") is True
    assert m.is_validated("mention") is False
    assert m.is_validated("unknown") is False
    assert m.category_names == ["mention", "econ"]

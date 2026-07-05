"""Tests for the signal generator: scoring, skip reasons, validated tagging."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kalshi_scanner.categories import CategoryMatcher
from kalshi_scanner.config import CategoryRule
from kalshi_scanner.feature_resolver import (
    ResearchFeatureResolver,
    build_training_table,
    default_research_world,
)
from kalshi_scanner.market import MarketSnapshot
from kalshi_scanner.model_provider import ModelProvider
from kalshi_scanner.signal_generator import SignalGenerator
from kalshi_scanner.store import SnapshotStore


@pytest.fixture(scope="module")
def wiring():
    world = default_research_world(seed=3, events_per_pair=30)
    provider = ModelProvider("gbdt", n_bootstrap=4, seed=1).fit(build_training_table(world))
    resolver = ResearchFeatureResolver(world.events, world.transcripts, world.gdelt)
    return provider, resolver


def _matcher(validated=False):
    return CategoryMatcher([
        CategoryRule(name="mention", validated=validated,
                     title_patterns=[r"\bsay\b", r"\bmention"]),
    ])


def _snap(title, ticker="T1"):
    return MarketSnapshot.from_api(
        {"ticker": ticker, "title": title, "close_time": "2024-11-06T00:00:00Z",
         "yes_bid": 40, "yes_ask": 45},
        scan_ts=datetime(2024, 11, 1, tzinfo=UTC), category="mention",
    )


def test_scores_parseable_market(wiring):
    provider, resolver = wiring
    gen = SignalGenerator(_matcher(), provider, resolver)
    (s,) = gen.generate([_snap('Will Trump say "tariff" during the rally?')])
    assert s.is_scored
    assert 0.0 <= s.model_prob <= 1.0 and s.ci_lo <= s.ci_hi
    assert s.model_version and s.features_json
    assert s.market_implied_prob == (40 + 45) / 2 / 100


def test_skips_unparseable_market(wiring):
    provider, resolver = wiring
    gen = SignalGenerator(_matcher(), provider, resolver)
    (s,) = gen.generate([_snap("Will CPI be above 3.5%?")])
    assert not s.is_scored
    assert s.reason == "unparseable_market"


def test_validated_flag_tracks_matcher(wiring):
    provider, resolver = wiring
    title = 'Will Trump say "tariff"?'
    unval = SignalGenerator(_matcher(False), provider, resolver).generate([_snap(title)])[0]
    val = SignalGenerator(_matcher(True), provider, resolver).generate([_snap(title)])[0]
    assert unval.validated is False
    assert val.validated is True


def test_records_signals_to_store(wiring, tmp_path):
    provider, resolver = wiring
    store = SnapshotStore(tmp_path / "s.sqlite")
    scan_id = store.start_run(datetime(2024, 11, 1, tzinfo=UTC), ["mention"])
    gen = SignalGenerator(_matcher(), provider, resolver, store=store)
    gen.generate(
        [_snap('Will Trump say "tariff"?'), _snap("Will CPI be high?", "T2")],
        scan_id=scan_id,
    )
    assert store.count_signals() == 2
    assert store.count_signals(scored_only=True) == 1
    rows = store.signals_for_scan(scan_id)
    assert {r["reason"] for r in rows} == {None, "unparseable_market"}

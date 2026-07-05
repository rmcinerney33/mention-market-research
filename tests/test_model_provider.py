"""Tests for the model provider: valid intervals, determinism, versioning."""

from __future__ import annotations

import pytest

from kalshi_scanner.feature_resolver import build_training_table, default_research_world
from kalshi_scanner.model_provider import ModelProvider


@pytest.fixture(scope="module")
def training():
    return build_training_table(default_research_world(seed=3, events_per_pair=30))


def test_predict_returns_valid_interval(training):
    prov = ModelProvider("gbdt", n_bootstrap=6, seed=1).fit(training)
    pred = prov.predict(training.iloc[[0]])
    assert 0.0 <= pred.ci_lo <= pred.ci_hi <= 1.0
    assert 0.0 <= pred.prob <= 1.0
    assert pred.n_bootstrap == 6


def test_deterministic_across_refits(training):
    row = training.iloc[[5]]
    a = ModelProvider("gbdt", n_bootstrap=6, seed=1).fit(training)
    b = ModelProvider("gbdt", n_bootstrap=6, seed=1).fit(training)
    assert a.model_version == b.model_version
    assert a.predict(row).prob == b.predict(row).prob


def test_version_changes_with_training_data(training):
    a = ModelProvider("gbdt", n_bootstrap=4, seed=1).fit(training)
    b = ModelProvider("gbdt", n_bootstrap=4, seed=1).fit(training.iloc[: len(training) // 2])
    assert a.model_hash != b.model_hash


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        ModelProvider("does_not_exist")


def test_predict_before_fit_raises(training):
    with pytest.raises(RuntimeError):
        ModelProvider("gbdt").predict(training.iloc[[0]])

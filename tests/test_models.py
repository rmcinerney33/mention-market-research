"""Tests for the model suite.

Verifies the shared interface (fit -> predict_proba), that probabilities are
valid, that degenerate single-class folds don't crash, and — importantly — that
the trainable models recover real signal (AUC > 0.5) on the informative
synthetic data. The Bayesian sampler is tested separately with tiny draws and
marked slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mention_market.features import build_feature_table
from mention_market.models import (
    BayesianHierarchicalModel,
    CoxSurvivalModel,
    GBDTModel,
    LaplaceBaseRateModel,
    LLMContextualModel,
    LogisticModel,
)
from mention_market.synthetic import SyntheticParams, generate


@pytest.fixture(scope="module")
def split():
    world = generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2,
                                     events_per_pair=80, seed=5))
    feats = build_feature_table(world.events, world.transcripts, world.gdelt, audit=False)
    feats = feats.sort_values("event_time").reset_index(drop=True)
    cut = int(len(feats) * 0.7)
    return feats.iloc[:cut].copy(), feats.iloc[cut:].copy()


def _check_valid_probs(p, n):
    assert p.shape == (n,)
    assert np.all(p > 0) and np.all(p < 1)
    assert not np.isnan(p).any()


@pytest.mark.parametrize(
    "model_cls", [LaplaceBaseRateModel, LogisticModel, CoxSurvivalModel, GBDTModel]
)
def test_fit_predict_valid(split, model_cls):
    train, test = split
    model = model_cls().fit(train)
    p = model.predict_proba(test)
    _check_valid_probs(p, len(test))


def test_llm_offline_is_deterministic_and_valid(split):
    train, test = split
    m1 = LLMContextualModel(use_api=False).fit(train)
    m2 = LLMContextualModel(use_api=False).fit(train)
    p1, p2 = m1.predict_proba(test), m2.predict_proba(test)
    _check_valid_probs(p1, len(test))
    np.testing.assert_allclose(p1, p2)  # deterministic offline heuristic


@pytest.mark.parametrize("model_cls", [LogisticModel, GBDTModel])
def test_trainable_models_recover_signal(split, model_cls):
    train, test = split
    y = test["label"].to_numpy()
    # Need both classes present in test to compute AUC.
    assert len(np.unique(y)) == 2
    p = model_cls().fit(train).predict_proba(test)
    assert roc_auc_score(y, p) > 0.5


def test_degenerate_single_class_training():
    # All-zero labels: models must return valid constant-ish probabilities.
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(40)],
            "speaker_id": ["s"] * 40,
            "phrase_id": ["p"] * 40,
            "event_time": pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC"),
            "venue": ["rally"] * 40,
            "label": [0] * 40,
            "lap_smoothed_rate": [0.1] * 40,
            "rr_hl7": [0.1] * 40, "rr_hl30": [0.1] * 40, "rr_hl90": [0.1] * 40,
            "use_count_5": [0.0] * 40, "use_count_10": [0.0] * 40, "use_count_20": [0.0] * 40,
            "use_rate_5": [0.0] * 40, "use_rate_10": [0.0] * 40, "use_rate_20": [0.0] * 40,
            "days_since_last_use": [np.nan] * 40, "ever_used_before": [0] * 40,
            "n_prior_events": list(range(40)), "topical_relevance": [0.5] * 40,
            "expected_speech_len": [100.0] * 40, "dow": [1] * 40, "hour": [12] * 40,
            "is_weekend": [0] * 40,
        }
    )
    for model_cls in (LaplaceBaseRateModel, LogisticModel, GBDTModel, CoxSurvivalModel):
        p = model_cls().fit(df).predict_proba(df)
        _check_valid_probs(p, len(df))


@pytest.mark.slow
def test_bayesian_runs(split):
    train, test = split
    model = BayesianHierarchicalModel(draws=50, tune=50, chains=1).fit(train)
    p = model.predict_proba(test)
    _check_valid_probs(p, len(test))

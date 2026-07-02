"""Tests for the Mincer-Zarnowitz market-efficiency regression.

We construct scenarios with known ground truth to prove the test detects the
right thing:
- when only the *market* forecast is informative, the model coefficient is
  driven to insignificance (market subsumes the model);
- when only the *model* forecast is informative, the model coefficient survives
  (model adds information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mention_market.efficiency import (
    merge_forecasts,
    mincer_zarnowitz,
    run_efficiency_analysis,
)


def _draw(n, seed):
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < latent).astype(int)
    return rng, latent, y


def test_market_subsumes_model_when_only_market_informative():
    rng, latent, y = _draw(1200, 0)
    p_market = np.clip(latent + rng.normal(0, 0.03, len(y)), 0.01, 0.99)  # sharp
    p_model = np.clip(rng.uniform(0, 1, len(y)), 0.01, 0.99)              # pure noise
    res = mincer_zarnowitz(y, p_model, p_market, lead_time_hours=1)

    assert res.pvalue["p_market"] < 0.01     # market clearly informative
    assert res.pvalue["p_model"] > 0.05      # model adds nothing
    assert not res.model_adds_info
    assert res.verdict == "market subsumes the model's information"


def test_model_adds_info_when_only_model_informative():
    rng, latent, y = _draw(1200, 1)
    p_model = np.clip(latent + rng.normal(0, 0.03, len(y)), 0.01, 0.99)   # sharp
    p_market = np.clip(rng.uniform(0, 1, len(y)), 0.01, 0.99)             # pure noise
    res = mincer_zarnowitz(y, p_model, p_market, lead_time_hours=1)

    assert res.pvalue["p_model"] < 0.01
    assert res.coef["p_model"] > 0
    assert res.model_adds_info
    assert res.verdict == "model adds information beyond the market"


def test_logit_method_runs():
    rng, latent, y = _draw(600, 2)
    p_model = np.clip(latent + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    p_market = np.clip(rng.uniform(0, 1, len(y)), 0.01, 0.99)
    res = mincer_zarnowitz(y, p_model, p_market, lead_time_hours=6, method="logit")
    assert res.method == "logit"
    assert set(res.terms) == {"const", "p_model", "p_market"}
    assert 0.0 <= res.r2 <= 1.0


def test_merge_forecasts_inner_join():
    preds = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "model": ["m", "m", "m"],
            "y": [1, 0, 1],
            "p": [0.6, 0.4, 0.7],
        }
    )
    snaps = pd.DataFrame(
        {
            "event_id": ["a", "b", "d"],
            "lead_time_hours": [1, 1, 1],
            "implied_prob": [0.5, 0.5, 0.9],
        }
    )
    merged = merge_forecasts(preds, snaps, "m", 1)
    assert list(merged["event_id"]) == ["a", "b"]  # 'c' has no snapshot, 'd' no pred
    assert set(merged.columns) == {"event_id", "y", "p_model", "p_market"}


def test_run_efficiency_analysis_structure():
    # Small synthetic end-to-end: structure + one row per (lead_time, term).
    from mention_market.features import build_feature_table
    from mention_market.models import GBDTModel, LaplaceBaseRateModel
    from mention_market.synthetic import SyntheticParams, generate
    from mention_market.validation import collect_oos_predictions, expanding_window_splits

    world = generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2,
                                     events_per_pair=70, seed=4))
    feats = build_feature_table(world.events, world.transcripts, world.gdelt, audit=False)
    feats = feats.sort_values("event_time").reset_index(drop=True)
    splits = expanding_window_splits(feats["event_time"], n_splits=3, min_train_frac=0.5)
    preds = collect_oos_predictions(
        {"gbdt": GBDTModel, "laplace": LaplaceBaseRateModel}, feats, splits
    )
    results, table = run_efficiency_analysis(
        preds, world.market_snapshots, "gbdt", lead_times_hours=[24, 6, 1]
    )
    assert len(results) >= 1
    # 3 terms per lead time present in the table.
    for L in table["lead_time_hours"].unique():
        assert set(table[table["lead_time_hours"] == L]["term"]) == {"const", "p_model", "p_market"}
    assert set(table.columns) >= {"lead_time_hours", "term", "coef", "se", "t", "pvalue", "verdict"}

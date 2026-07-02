"""Tests for the validation methodology.

Includes the time-ordering guarantee for walk-forward splits (train strictly
before test), metric correctness, bootstrap CI sanity, Diebold-Mariano behavior
on identical vs clearly-different forecasts, and calibration data integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mention_market.validation import (
    blocked_splits,
    bootstrap_ci,
    brier,
    collect_oos_predictions,
    diebold_mariano,
    diebold_mariano_matrix,
    expanding_window_splits,
    expected_calibration_error,
    logloss,
    reliability_curve,
    summarize_metrics,
)


@pytest.fixture
def event_times():
    # 100 timestamps, deliberately shuffled to prove the splitter sorts.
    ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    return ts.sample(frac=1.0, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitters — no lookahead
# ---------------------------------------------------------------------------
def test_expanding_window_train_strictly_before_test(event_times):
    splits = expanding_window_splits(event_times, n_splits=5, min_train_frac=0.4)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) and len(test_idx)
        # No overlap between train and test rows.
        assert set(train_idx).isdisjoint(set(test_idx))
        # The guarantee: every train time is strictly before every test time.
        assert event_times.iloc[train_idx].max() < event_times.iloc[test_idx].min()


def test_expanding_window_train_grows(event_times):
    splits = expanding_window_splits(event_times, n_splits=4, min_train_frac=0.4)
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)  # non-decreasing training set


def test_blocked_splits_train_before_test_within_block(event_times):
    splits = blocked_splits(event_times, n_blocks=4, train_frac_within_block=0.6)
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert event_times.iloc[train_idx].max() < event_times.iloc[test_idx].min()


def test_expanding_window_rejects_bad_frac(event_times):
    with pytest.raises(ValueError):
        expanding_window_splits(event_times, min_train_frac=1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metric_values_known():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier(y, p) == pytest.approx(0.0)
    assert logloss(y, p) < 1e-4  # clipped, near-zero


def test_brier_worst_case():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier(y, p) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(0, 1, 500)
    est = bootstrap_ci(y, p, brier, n_boot=400, seed=1)
    assert est.lo <= est.point <= est.hi
    assert est.hi > est.lo


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------
def test_dm_identical_forecasts_not_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0, 1, 200)
    dm, pv = diebold_mariano(y, p, p.copy(), loss="brier")
    assert dm == pytest.approx(0.0)
    assert pv == pytest.approx(1.0)


def test_dm_detects_clearly_better_forecast():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, 0.9, 0.1)          # near-perfect
    bad = np.full(400, 0.5)                     # uninformative
    dm, pv = diebold_mariano(y, bad, good, loss="brier")
    # bad has higher loss than good => positive stat, significant.
    assert dm > 0
    assert pv < 0.01


def test_dm_matrix_shape_and_diag(sample_preds):
    m = diebold_mariano_matrix(sample_preds, loss="brier")
    assert list(m.index) == list(m.columns)
    assert m.values.diagonal().astype(float).tolist() == pytest.approx(
        [float("nan")] * len(m), nan_ok=True
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_reliability_curve_counts_sum():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(0, 1, 300)
    rc = reliability_curve(y, p, n_bins=10)
    assert rc["count"].sum() == 300


def test_ece_zero_for_perfectly_calibrated():
    # Construct predictions equal to the true bin frequency (2/10 = 0.2 positive).
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.full(10, 0.2)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end prediction collection
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_preds():
    from mention_market.features import build_feature_table
    from mention_market.models import GBDTModel, LaplaceBaseRateModel, LogisticModel
    from mention_market.synthetic import SyntheticParams, generate

    world = generate(SyntheticParams(n_speakers=2, phrases_per_speaker=2,
                                     events_per_pair=60, seed=9))
    feats = build_feature_table(world.events, world.transcripts, world.gdelt, audit=False)
    feats = feats.sort_values("event_time").reset_index(drop=True)
    splits = expanding_window_splits(feats["event_time"], n_splits=3, min_train_frac=0.5)
    factories = {
        "laplace": LaplaceBaseRateModel,
        "logistic": LogisticModel,
        "gbdt": GBDTModel,
    }
    return collect_oos_predictions(factories, feats, splits)


def test_collect_oos_predictions_columns(sample_preds):
    assert set(sample_preds.columns) == {"event_id", "event_time", "fold", "model", "y", "p"}
    assert sample_preds["model"].nunique() == 3
    assert ((sample_preds["p"] > 0) & (sample_preds["p"] < 1)).all()


def test_summarize_metrics_orders_by_brier(sample_preds):
    table = summarize_metrics(sample_preds, n_boot=200)
    assert list(table["model"])  # non-empty
    assert (table["brier"].values == np.sort(table["brier"].values)).all()
    for col in ("brier", "log_loss", "auc"):
        assert (table[f"{col}_lo"] <= table[col] + 1e-9).all()
        assert (table[col] <= table[f"{col}_hi"] + 1e-9).all()

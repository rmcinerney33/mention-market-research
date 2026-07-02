"""Smoke test for the end-to-end pipeline.

Runs a small, fast configuration (no Bayesian sampler) into a temp directory
and checks that every artifact — metric table, DM matrix, MZ results, figures,
and the Markdown report — is produced and self-consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mention_market.pipeline import run_pipeline
from mention_market.synthetic import SyntheticParams


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    params = SyntheticParams(n_speakers=2, phrases_per_speaker=2,
                             events_per_pair=60, seed=3)
    return run_pipeline(
        params=params, output_dir=out, n_splits=3, min_train_frac=0.5,
        include_bayesian=False, n_boot=200, seed=3,
    )


def test_metric_table(artifacts):
    m = artifacts.metrics
    # One row per non-Bayesian model.
    assert set(m["model"]) == {
        "laplace_base_rate", "logistic", "cox_survival", "gbdt", "llm_contextual"
    }
    for col in ("brier", "auc", "ece"):
        assert col in m.columns
    # Ranked by Brier ascending.
    assert (m["brier"].values == sorted(m["brier"].values)).all()


def test_mz_table_has_all_terms_per_lead_time(artifacts):
    t = artifacts.mz_table
    assert not t.empty
    for L in t["lead_time_hours"].unique():
        assert set(t[t["lead_time_hours"] == L]["term"]) == {"const", "p_model", "p_market"}


def test_figures_written(artifacts):
    for key in ("reliability", "brier", "auc", "efficiency"):
        p = artifacts.figure_paths.get(key)
        assert p is not None and Path(p).exists() and Path(p).stat().st_size > 0


def test_report_written_and_mentions_headline(artifacts):
    text = Path(artifacts.report_path).read_text()
    assert "market-efficiency" in text.lower()
    assert artifacts.best_model in text
    # CSV siblings exist.
    rep_dir = Path(artifacts.report_path).parent
    for name in ("metrics.csv", "dm_matrix.csv", "mz_results.csv", "oos_predictions.csv"):
        assert (rep_dir / name).exists()

"""End-to-end pipeline: data -> features -> models -> validation -> efficiency -> report.

Runs the whole study on the synthetic world (or, once wired, real ingested
data) and writes tables, figures, and a Markdown report to ``outputs/``. This is
the single command that reproduces the headline result.

    python -m mention_market.pipeline --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import repo_root
from .efficiency import run_efficiency_analysis
from .features import build_feature_table
from .models import default_model_factories
from .plots import plot_efficiency, plot_metric_with_ci, plot_reliability
from .synthetic import SyntheticParams, generate
from .validation import (
    collect_oos_predictions,
    diebold_mariano_matrix,
    expanding_window_splits,
    expected_calibration_error,
    summarize_metrics,
)


@dataclass
class PipelineArtifacts:
    metrics: pd.DataFrame
    dm_matrix: pd.DataFrame
    predictions: pd.DataFrame
    mz_table: pd.DataFrame
    best_model: str
    data_summary: dict
    figure_paths: dict[str, Path]
    report_path: Path


def run_pipeline(
    *,
    params: SyntheticParams | None = None,
    output_dir: Path | None = None,
    n_splits: int = 4,
    min_train_frac: float = 0.4,
    include_bayesian: bool = True,
    mz_method: str = "ols",
    lead_times_hours: tuple[int, ...] = (24, 6, 1),
    n_boot: int = 1000,
    seed: int = 42,
) -> PipelineArtifacts:
    root = repo_root()
    out = output_dir or (root / "outputs")
    fig_dir = out / "figures"
    rep_dir = out / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data (synthetic; swap in real ingest here when available).
    p = params or SyntheticParams(n_speakers=2, phrases_per_speaker=3,
                                  events_per_pair=80, seed=seed)
    world = generate(p)

    # 2. Features — with the no-lookahead audit on.
    feats = build_feature_table(world.events, world.transcripts, world.gdelt, audit=True)
    feats = feats.sort_values("event_time").reset_index(drop=True)

    # 3. Walk-forward out-of-sample predictions for every model.
    splits = expanding_window_splits(feats["event_time"], n_splits=n_splits,
                                     min_train_frac=min_train_frac)
    factories = default_model_factories(
        include_bayesian=include_bayesian,
        bayesian_kwargs={"draws": 500, "tune": 1000, "chains": 2, "target_accept": 0.95},
    )
    preds = collect_oos_predictions(factories, feats, splits)

    # 4. Metric table (+ bootstrap CIs) and calibration error.
    metrics = summarize_metrics(preds, n_boot=n_boot, seed=seed)
    ece = {
        m: expected_calibration_error(g["y"].to_numpy(), g["p"].to_numpy())
        for m, g in preds.groupby("model")
    }
    metrics["ece"] = metrics["model"].map(ece)

    # 5. Diebold-Mariano pairwise significance.
    dm_matrix = diebold_mariano_matrix(preds, loss="brier")

    # 6. Headline: market-efficiency regression for the best (lowest-Brier) model.
    best_model = metrics.iloc[0]["model"]
    _, mz_table = run_efficiency_analysis(
        preds, world.market_snapshots, best_model,
        lead_times_hours=list(lead_times_hours), method=mz_method,
    )

    # 7. Figures.
    figs = {
        "reliability": plot_reliability(preds, fig_dir / "reliability.png"),
        "brier": plot_metric_with_ci(metrics, "brier", fig_dir / "brier_ci.png"),
        "auc": plot_metric_with_ci(metrics, "auc", fig_dir / "auc_ci.png"),
    }
    if not mz_table.empty:
        figs["efficiency"] = plot_efficiency(mz_table, fig_dir / "efficiency.png")

    data_summary = {
        "n_events": int(len(feats)),
        "base_rate": float(feats["label"].mean()),
        "n_speakers": p.n_speakers,
        "n_phrases": p.n_speakers * p.phrases_per_speaker,
        "date_start": str(feats["event_time"].min().date()),
        "date_end": str(feats["event_time"].max().date()),
        "market_efficiency_setting": p.market_efficiency,
        "n_folds": len(splits),
        "mz_method": mz_method,
    }

    # 8. Persist tables + report.
    metrics.to_csv(rep_dir / "metrics.csv", index=False)
    dm_matrix.to_csv(rep_dir / "dm_matrix.csv")
    mz_table.to_csv(rep_dir / "mz_results.csv", index=False)
    preds.to_csv(rep_dir / "oos_predictions.csv", index=False)

    report_path = _write_report(
        rep_dir / "REPORT.md", metrics, dm_matrix, mz_table, best_model, data_summary, figs
    )

    return PipelineArtifacts(
        metrics=metrics, dm_matrix=dm_matrix, predictions=preds, mz_table=mz_table,
        best_model=best_model, data_summary=data_summary, figure_paths=figs,
        report_path=report_path,
    )


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_report(path, metrics, dm_matrix, mz_table, best_model, summary, figs) -> Path:
    metric_view = metrics[
        ["model", "n", "brier", "brier_lo", "brier_hi", "log_loss", "auc", "auc_lo", "auc_hi", "ece"]
    ].copy()

    # Verdict text from the sharpest available lead time.
    verdict_lines = []
    if not mz_table.empty:
        for L in sorted(mz_table["lead_time_hours"].unique(), reverse=True):
            sub = mz_table[mz_table["lead_time_hours"] == L]
            model_row = sub[sub["term"] == "p_model"].iloc[0]
            market_row = sub[sub["term"] == "p_market"].iloc[0]
            verdict = model_row["verdict"]
            verdict_lines.append(
                f"- **T-{L}h** (n={int(model_row['n'])}): "
                f"model coef={model_row['coef']:.3f} (p={model_row['pvalue']:.3f}), "
                f"market coef={market_row['coef']:.3f} (p={market_row['pvalue']:.3f}) "
                f"— _{verdict}_."
            )

    dm_reset = dm_matrix.copy()
    dm_reset.insert(0, "model", dm_reset.index)

    content = f"""# Mention-Market Prediction & Market-Efficiency Report

_Generated by `mention_market.pipeline`. Numbers below are from a run on the
**synthetic** world (known ground truth); swap in real ingested data to reproduce
on live markets._

## 1. Data summary

| field | value |
| --- | --- |
| events (prediction units) | {summary['n_events']} |
| base rate P(phrase used) | {summary['base_rate']:.3f} |
| speakers / phrases | {summary['n_speakers']} / {summary['n_phrases']} |
| date range | {summary['date_start']} → {summary['date_end']} |
| walk-forward folds | {summary['n_folds']} |
| synthetic market-efficiency setting | {summary['market_efficiency_setting']} |
| MZ regression method | {summary['mz_method']} |

Every feature is constructed strictly from data available before each event
(`features.build_feature_table(..., audit=True)`), and a prior event's label is
only used once its transcript would have been *published* — see
`src/mention_market/lookahead.py` and `tests/test_lookahead.py`.

## 2. Model comparison (walk-forward OOS, 95% bootstrap CIs)

Lower Brier / log loss is better; higher AUC is better. `ece` is expected
calibration error.

{_md_table(metric_view)}

Best model by Brier: **{best_model}**.

## 3. Pairwise Diebold-Mariano p-values (Brier loss)

Cell (row, col) = p-value for H0 that the two models have equal expected loss.

{_md_table(dm_reset)}

## 4. Calibration

See `figures/reliability.png`. `ece` in the table above is the count-weighted
mean gap between predicted probability and empirical frequency.

## 5. Headline: market-efficiency regression

Forecast-encompassing regression `y ~ p_model + p_market` for the best model
(**{best_model}**) at each lead time. A significant, positive `p_model`
coefficient after controlling for the market means the model carries information
beyond the price; a `p_model` driven to ~0 means the market already subsumes it.

{chr(10).join(verdict_lines) if verdict_lines else '_Insufficient overlapping market data to run the regression._'}

Full coefficients in `reports/mz_results.csv`; coefficient-vs-lead-time plot in
`figures/efficiency.png`.

## 6. Limitations

- **Synthetic data.** These numbers exercise and validate the methodology on a
  known-ground-truth world; they are not a claim about real mention markets. On
  the default `market_efficiency=1.0` setting the market observes the true
  per-event probability, so the expected — and correct — headline is that the
  market subsumes the model. Lowering that setting injects a detectable edge,
  which the same test recovers (see `tests/test_efficiency.py`).
- **Sample size per phrase.** Real mention markets are thin; per-phrase N is
  small, which is exactly why the Bayesian hierarchical model pools across
  phrases.
- **Market price is trade-derived.** Historical Kalshi/Polymarket prices are
  reconstructed mids at each lead time, not true book mids (see README).
- **Venue heterogeneity & transcript timing lag.** Venue mix and publish-time
  lag vary across speakers and are only partially captured.
- **Survivorship bias.** Markets are only made on phrases someone expected to be
  interesting; the traded universe is not a random sample of phrases.
"""
    Path(path).write_text(content)
    return Path(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the mention-market research pipeline.")
    ap.add_argument("--no-bayesian", action="store_true", help="skip the (slow) Bayesian model")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--events-per-pair", type=int, default=80)
    ap.add_argument("--market-efficiency", type=float, default=1.0,
                    help="1.0 = efficient market; lower injects a detectable model edge")
    ap.add_argument("--mz-method", choices=["ols", "logit"], default="ols")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    params = SyntheticParams(
        n_speakers=2, phrases_per_speaker=3, events_per_pair=args.events_per_pair,
        market_efficiency=args.market_efficiency, seed=args.seed,
    )
    art = run_pipeline(
        params=params, n_splits=args.n_splits,
        include_bayesian=not args.no_bayesian, mz_method=args.mz_method, seed=args.seed,
    )
    print(f"Best model: {art.best_model}")
    print(f"Report:     {art.report_path}")
    print(art.metrics[["model", "brier", "auc", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()

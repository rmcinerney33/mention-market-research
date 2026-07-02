# mention-market-research

Predicting whether a public figure will use a specific phrase in an upcoming
speech/press conference, and testing whether the model's forecasts contain
information beyond what is already reflected in prediction-market prices
("mention markets" on Kalshi / Polymarket).

The **headline research question** is a market-efficiency test — not raw
predictive accuracy. A result of "the market already subsumes the model's
information" is a legitimate finding.

## Status

All seven phases implemented and runnable end-to-end. The full study runs on a
synthetic world with known ground truth (so the methodology itself can be
validated); the real ingestion adapters are wired for when API access and
historical data are available. See **Results** below and
[`outputs/reports/REPORT.md`](outputs/reports/REPORT.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                  # 90+ tests; lookahead suite is the core
python -m mention_market.pipeline       # runs the whole study -> outputs/
```

Useful flags: `--no-bayesian` (skip the slow sampler), `--market-efficiency 0.3`
(inject a detectable edge), `--mz-method logit`, `--events-per-pair N`.

## Results (synthetic demonstration)

Walk-forward out-of-sample, 4 folds, 288 test predictions per model, 95%
bootstrap CIs. Lower Brier is better; higher AUC is better; `ece` is expected
calibration error.

| model | Brier | AUC | ECE |
| --- | --- | --- | --- |
| bayesian_hierarchical | **0.235** | 0.590 | 0.047 |
| llm_contextual | 0.237 | 0.590 | 0.048 |
| gbdt | 0.240 | 0.581 | **0.040** |
| logistic | 0.244 | **0.599** | 0.095 |
| laplace_base_rate (floor) | 0.244 | 0.531 | 0.052 |
| cox_survival | 0.328 | 0.425 | 0.262 |

The trained models edge out the Laplace base rate on Brier/AUC but — by
Diebold-Mariano — **not significantly** on this thin sample; the survival model
underperforms the base rate here (an honest negative result). This is the
expected texture of a hard, low-signal prediction problem.

### Headline: market-efficiency test

Forecast-encompassing regression `y ~ p_model + p_market` for the best model,
run at three lead times, on the default **efficient** synthetic market:

| lead time | model coef (p) | market coef (p) | verdict |
| --- | --- | --- | --- |
| T-24h | 0.57 (0.066) | 0.58 (<0.001) | market subsumes model |
| T-6h | 0.49 (0.109) | 0.73 (<0.001) | market subsumes model |
| T-1h | 0.43 (0.162) | 0.86 (<0.001) | market subsumes model |

The market coefficient is strongly significant at every lead time; the model
coefficient never clears significance after controlling for the price, and its
marginal contribution **decays as the event approaches** while the market
sharpens. That is the textbook signature of an efficient market — and a
legitimate finding, not a failure.

**The test has power in the other direction too.** Re-running with an
inefficient market (`--market-efficiency 0.3`), the same regression recovers a
significant model coefficient at every lead time (p = 0.007 / 0.011 / 0.015 →
"model adds information beyond the market"). Same machinery, opposite verdict,
both correct.

## Phased build

## Phased build

1. **Scaffolding & environment** — layout, deps, config, pytest.
2. **Data pipeline** — transcripts, market prices, GDELT, labeling, no-lookahead guarantee.
3. **Feature engineering** — recency-weighted usage, hazard time features, topical relevance, venue, etc.
4. **Model suite** — Laplace baseline, logistic, Cox PH, GBDT, Bayesian hierarchical, LLM contextual.
5. **Validation** — walk-forward CV, blocked CV, bootstrap CIs, Diebold-Mariano, calibration.
6. **Market efficiency test** — Mincer-Zarnowitz regression on model probability + market price at multiple lead times.
7. **Write-up** — data pipeline docs, model comparison, calibration, headline regression, limitations.

## Layout

```
mention-market-research/
├── config.yaml            # speakers, phrases, date ranges, endpoints (secrets via env vars)
├── requirements.txt       # pinned deps
├── pyproject.toml         # package + pytest + ruff config
├── src/mention_market/    # library code
├── tests/                 # pytest; special focus on timestamp/lookahead correctness
├── notebooks/             # exploratory analysis
├── data/
│   ├── raw/               # untouched pulls from each source
│   ├── processed/         # cleaned, timestamped feature tables
│   └── external/          # third-party reference data (e.g. GDELT slices)
└── outputs/
    ├── figures/
    ├── models/
    └── reports/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Known upfront constraints

- **Historical Kalshi order books.** Public API exposes real-time L2 and trade
  history but not arbitrary historical book snapshots. Pre-event "market price"
  will be a trade-derived mid at the target lead time.
- **Polymarket** provides better historical price time-series (Gamma/CLOB + subgraph).
- **Speaker coverage.** Trump is the only figure with dense mention-market history
  across both venues. Config supports a list; working corpus will be Trump-heavy.
- **Transcript sources.** Rev.com is paid; C-SPAN and Roll Call coverage is uneven.
  Expect this to be the messiest part of ingestion.

## Guarantees

Every predictive feature for an event at time `t` must be constructible from
data whose timestamp is strictly less than `t`. This is enforced by the
pipeline, not by convention — see `src/mention_market/lookahead.py` (Phase 2)
and the dedicated test suite in `tests/test_lookahead.py`.

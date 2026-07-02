# mention-market-research

Predicting whether a public figure will use a specific phrase in an upcoming
speech/press conference, and testing whether the model's forecasts contain
information beyond what is already reflected in prediction-market prices
("mention markets" on Kalshi / Polymarket).

The **headline research question** is a market-efficiency test — not raw
predictive accuracy. A result of "the market already subsumes the model's
information" is a legitimate finding.

## Status

Phase 1 (scaffolding) — see `Phased build` below.

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

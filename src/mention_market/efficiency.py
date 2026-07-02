"""Market-efficiency test — the headline result.

We ask whether a model's forecast carries information *beyond* what the market
price already reflects. The tool is a Mincer-Zarnowitz-style forecast-encompassing
regression of the realized binary outcome on both forecasts, simultaneously:

    y = beta0 + beta1 * p_model + beta2 * p_market + e         (HC1-robust SEs)

Interpretation at a fixed lead time:

- ``beta1`` (model) statistically significant after controlling for the market
  => the model contains information the market has not priced in (an edge).
- ``beta1`` driven to ~0 while ``beta2`` (market) carries the weight
  => the market already subsumes the model's information (efficient market).

We run this at each configured lead time (e.g. 24h / 6h / 1h before the event)
to see whether any edge decays as the event approaches. We use a linear
probability model (classic MZ); a logistic variant is available for robustness.
Single-regressor MZ fits for each forecast alone are reported for context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class MZResult:
    lead_time_hours: int
    n: int
    terms: list[str]
    coef: dict[str, float]
    se: dict[str, float]
    tstat: dict[str, float]
    pvalue: dict[str, float]
    r2: float
    method: str = "ols"
    extra: dict = field(default_factory=dict)

    @property
    def model_adds_info(self) -> bool:
        """True if the model coefficient is positive and significant (p<0.05)
        after controlling for the market price."""
        return self.coef.get("p_model", 0.0) > 0 and self.pvalue.get("p_model", 1.0) < 0.05

    @property
    def verdict(self) -> str:
        if self.model_adds_info:
            return "model adds information beyond the market"
        return "market subsumes the model's information"


def merge_forecasts(
    preds: pd.DataFrame,
    market_snapshots: pd.DataFrame,
    model_name: str,
    lead_time_hours: int,
) -> pd.DataFrame:
    """Align a model's OOS forecasts with the market price at one lead time.

    Returns columns ``[event_id, y, p_model, p_market]`` for events present in
    both the predictions and the market snapshots.
    """
    mp = preds[preds["model"] == model_name][["event_id", "y", "p"]].rename(columns={"p": "p_model"})
    snap = market_snapshots[market_snapshots["lead_time_hours"] == lead_time_hours]
    snap = snap[["event_id", "implied_prob"]].rename(columns={"implied_prob": "p_market"})
    merged = mp.merge(snap, on="event_id", how="inner")
    return merged.dropna(subset=["y", "p_model", "p_market"]).reset_index(drop=True)


def mincer_zarnowitz(
    y: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    lead_time_hours: int,
    method: str = "ols",
) -> MZResult:
    """Joint forecast-encompassing regression with robust inference.

    ``method='ols'`` fits a linear probability model with HC1 robust SEs (the
    classic MZ). ``method='logit'`` fits a logistic regression (SEs from the
    MLE information matrix) for robustness to the LPM's out-of-range fitted
    values. Terms: const, p_model, p_market.
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([p_model, p_market])
    X = sm.add_constant(X, has_constant="add")
    terms = ["const", "p_model", "p_market"]

    if method == "ols":
        res = sm.OLS(y, X).fit(cov_type="HC1")
        r2 = float(res.rsquared)
    elif method == "logit":
        res = sm.Logit(y, X).fit(disp=False)
        r2 = float(res.prsquared)  # McFadden pseudo-R^2
    else:
        raise ValueError(f"unknown method {method!r}")

    return MZResult(
        lead_time_hours=lead_time_hours,
        n=int(len(y)),
        terms=terms,
        coef=dict(zip(terms, res.params, strict=True)),
        se=dict(zip(terms, res.bse, strict=True)),
        tstat=dict(zip(terms, res.tvalues, strict=True)),
        pvalue=dict(zip(terms, res.pvalues, strict=True)),
        r2=r2,
        method=method,
        extra={
            # Single-regressor MZ slopes for context (unbiasedness ~ slope 1).
            "market_only_slope": _single_slope(y, p_market),
            "model_only_slope": _single_slope(y, p_model),
        },
    )


def _single_slope(y: np.ndarray, x: np.ndarray) -> float:
    X = sm.add_constant(x, has_constant="add")
    return float(sm.OLS(y, X).fit().params[1])


def run_efficiency_analysis(
    preds: pd.DataFrame,
    market_snapshots: pd.DataFrame,
    model_name: str,
    lead_times_hours: list[int],
    method: str = "ols",
) -> tuple[list[MZResult], pd.DataFrame]:
    """Run the MZ regression at each lead time for one model.

    Returns the list of :class:`MZResult` plus a tidy table with one row per
    (lead_time, term): coef / se / t / p, and the per-lead-time verdict.
    """
    results: list[MZResult] = []
    rows: list[dict] = []
    for L in lead_times_hours:
        merged = merge_forecasts(preds, market_snapshots, model_name, L)
        if len(merged) < 20:
            continue
        res = mincer_zarnowitz(
            merged["y"].to_numpy(),
            merged["p_model"].to_numpy(),
            merged["p_market"].to_numpy(),
            lead_time_hours=L,
            method=method,
        )
        results.append(res)
        for term in res.terms:
            rows.append(
                {
                    "lead_time_hours": L,
                    "term": term,
                    "coef": res.coef[term],
                    "se": res.se[term],
                    "t": res.tstat[term],
                    "pvalue": res.pvalue[term],
                    "n": res.n,
                    "verdict": res.verdict,
                }
            )
    return results, pd.DataFrame(rows)

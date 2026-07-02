"""Validation methodology.

Everything here respects the arrow of time. Splits are **never** random k-fold:
they are expanding-window walk-forward (train strictly precedes test) or blocked
(contiguous, non-overlapping test windows). On top of out-of-sample predictions
we compute:

- Brier score, log loss, AUC (the reported metrics)
- bootstrap confidence intervals on each
- the Diebold-Mariano test for pairwise forecast-accuracy significance
- reliability-curve data for calibration diagrams

The design keeps the statistics separate from plotting: this module returns
tidy DataFrames; ``plots.py`` (Phase 7) renders them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# --------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------
Split = tuple[np.ndarray, np.ndarray]


def expanding_window_splits(
    event_times: pd.Series, n_splits: int = 5, min_train_frac: float = 0.4
) -> list[Split]:
    """Walk-forward expanding-window splits ordered by ``event_times``.

    The first ``min_train_frac`` of the timeline seeds the initial training set;
    the remainder is divided into ``n_splits`` consecutive test blocks. For each
    block, the training set is everything strictly before it. Train event times
    are therefore always < test event times — no lookahead.
    """
    order = np.argsort(event_times.to_numpy())
    n = len(order)
    start = int(np.floor(min_train_frac * n))
    if start < 1 or start >= n:
        raise ValueError("min_train_frac leaves no room for train/test blocks")

    boundaries = np.linspace(start, n, n_splits + 1, dtype=int)
    splits: list[Split] = []
    for i in range(n_splits):
        test_lo, test_hi = boundaries[i], boundaries[i + 1]
        if test_hi <= test_lo:
            continue
        train_idx = order[:test_lo]
        test_idx = order[test_lo:test_hi]
        splits.append((train_idx, test_idx))
    return splits


def blocked_splits(
    event_times: pd.Series, n_blocks: int = 5, train_frac_within_block: float = 0.6
) -> list[Split]:
    """Blocked CV: cut the timeline into ``n_blocks`` contiguous blocks and,
    within each block, train on the earlier ``train_frac_within_block`` and test
    on the rest. Yields a *distribution* of performance across disjoint time
    windows rather than one number (still walk-forward within each block).
    """
    order = np.argsort(event_times.to_numpy())
    n = len(order)
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    splits: list[Split] = []
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi - lo < 5:
            continue
        cut = lo + int(np.floor((hi - lo) * train_frac_within_block))
        if cut <= lo or cut >= hi:
            continue
        splits.append((order[lo:cut], order[cut:hi]))
    return splits


# --------------------------------------------------------------------------
# Out-of-sample prediction collection
# --------------------------------------------------------------------------
def collect_oos_predictions(
    model_factories: dict[str, Callable[[], object]],
    feats: pd.DataFrame,
    splits: list[Split],
) -> pd.DataFrame:
    """Fit each model on each split's train and predict its test.

    ``model_factories`` maps model name -> a zero-arg callable returning a fresh
    model (fresh so folds never share fitted state). Returns a long DataFrame:
    ``[event_id, event_time, fold, model, y, p]``.
    """
    rows: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train = feats.iloc[train_idx]
        test = feats.iloc[test_idx]
        y = test["label"].to_numpy(dtype=int)
        for name, factory in model_factories.items():
            model = factory()
            model.fit(train)
            p = np.asarray(model.predict_proba(test), dtype=float)
            rows.append(
                pd.DataFrame(
                    {
                        "event_id": test["event_id"].to_numpy(),
                        "event_time": test["event_time"].to_numpy(),
                        "fold": fold,
                        "model": name,
                        "y": y,
                        "p": p,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    # AUC is undefined when only one class is present in the sample.
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "brier": brier,
    "log_loss": logloss,
    "auc": auc,
}
# Lower is better for these; used when ranking.
LOWER_IS_BETTER = {"brier": True, "log_loss": True, "auc": False}


# --------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------
@dataclass
class Estimate:
    point: float
    lo: float
    hi: float


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Estimate:
    """Percentile bootstrap CI for a metric over the pooled OOS observations."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = metric(y, p)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = metric(y[idx], p[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point=point, lo=float(lo), hi=float(hi))


def summarize_metrics(
    preds: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Per-model metric table with bootstrap CIs over pooled OOS predictions."""
    out_rows = []
    for model, grp in preds.groupby("model"):
        y = grp["y"].to_numpy(dtype=int)
        p = grp["p"].to_numpy(dtype=float)
        row: dict[str, object] = {"model": model, "n": len(y)}
        for mname, mfn in METRICS.items():
            est = bootstrap_ci(y, p, mfn, n_boot=n_boot, seed=seed)
            row[mname] = est.point
            row[f"{mname}_lo"] = est.lo
            row[f"{mname}_hi"] = est.hi
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("brier").reset_index(drop=True)


# --------------------------------------------------------------------------
# Diebold-Mariano test
# --------------------------------------------------------------------------
def _pointwise_loss(y: np.ndarray, p: np.ndarray, loss: str) -> np.ndarray:
    if loss == "brier":
        return (y - p) ** 2
    if loss == "log_loss":
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    raise ValueError(f"unknown loss {loss!r}")


def diebold_mariano(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, loss: str = "brier", h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test comparing two probabilistic forecasts.

    Returns ``(dm_stat, p_value)`` for H0: equal expected loss. Positive stat
    means model 1 has *higher* loss (model 2 better). Uses a Newey-West HAC
    variance with ``h-1`` lags; for one-step forecasts (``h=1``) that's the
    plain mean/variance t-statistic. Two-sided p-value from the normal.
    """
    from scipy import stats

    d = _pointwise_loss(y, p1, loss) - _pointwise_loss(y, p2, loss)
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return 0.0, 1.0
    d_bar = d.mean()
    # Newey-West long-run variance with h-1 lags.
    gamma0 = np.mean((d - d_bar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        weight = 1 - lag / h
        var += 2 * weight * cov
    if var <= 0:
        return 0.0, 1.0
    dm = d_bar / np.sqrt(var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p_value)


def diebold_mariano_matrix(preds: pd.DataFrame, loss: str = "brier") -> pd.DataFrame:
    """Pairwise DM p-values across models on their common OOS observations.

    Models are aligned on ``event_id`` so the loss differential is paired.
    """
    models = sorted(preds["model"].unique())
    wide = preds.pivot_table(index="event_id", columns="model", values="p")
    y = preds.drop_duplicates("event_id").set_index("event_id")["y"].reindex(wide.index).to_numpy()
    pmat = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                pmat.loc[a, b] = np.nan
                continue
            _, pv = diebold_mariano(y, wide[a].to_numpy(), wide[b].to_numpy(), loss=loss)
            pmat.loc[a, b] = pv
    return pmat


# --------------------------------------------------------------------------
# Calibration / reliability
# --------------------------------------------------------------------------
def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned reliability data: predicted vs empirical frequency per bin.

    Returns columns ``[bin_lo, bin_hi, mean_pred, frac_pos, count]`` for the
    non-empty bins. Feeds calibration diagrams and the calibration error.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "mean_pred": float(p[mask].mean()),
                "frac_pos": float(y[mask].mean()),
                "count": cnt,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |mean_pred - frac_pos| across bins."""
    rc = reliability_curve(y, p, n_bins=n_bins)
    if rc.empty:
        return float("nan")
    w = rc["count"] / rc["count"].sum()
    return float((w * (rc["mean_pred"] - rc["frac_pos"]).abs()).sum())


def iter_model_predictions(preds: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(model, y, p)`` for each model in a predictions frame."""
    for model, grp in preds.groupby("model"):
        yield model, grp["y"].to_numpy(dtype=int), grp["p"].to_numpy(dtype=float)

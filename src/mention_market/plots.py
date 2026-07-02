"""Figure rendering for the report (Phase 7).

Kept separate from the statistics in ``validation.py`` / ``efficiency.py``:
these functions take the tidy frames those modules produce and write PNGs. Uses
the non-interactive Agg backend so it runs headless in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .validation import iter_model_predictions, reliability_curve  # noqa: E402


def plot_reliability(preds: pd.DataFrame, out_path: Path, n_bins: int = 10) -> Path:
    """Reliability (calibration) diagram: predicted vs empirical frequency."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for model, y, p in iter_model_predictions(preds):
        rc = reliability_curve(y, p, n_bins=n_bins)
        if rc.empty:
            continue
        ax.plot(rc["mean_pred"], rc["frac_pos"], marker="o", ms=4, label=model)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration / reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_metric_with_ci(metrics: pd.DataFrame, metric: str, out_path: Path) -> Path:
    """Horizontal bar chart of a metric per model with bootstrap CI whiskers."""
    df = metrics.sort_values(metric, ascending=(metric != "auc")).reset_index(drop=True)
    point = df[metric].to_numpy()
    lo = df[f"{metric}_lo"].to_numpy()
    hi = df[f"{metric}_hi"].to_numpy()
    err = np.vstack([point - lo, hi - point])
    ypos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1.5))
    ax.barh(ypos, point, xerr=err, color="#4c72b0", alpha=0.85,
            error_kw={"ecolor": "#333", "capsize": 4})
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} by model (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_efficiency(mz_table: pd.DataFrame, out_path: Path) -> Path:
    """Model vs market coefficient across lead times, with ±1.96·SE bands."""
    fig, ax = plt.subplots(figsize=(7, 5))
    lead_times = sorted(mz_table["lead_time_hours"].unique(), reverse=True)
    x = np.arange(len(lead_times))
    for term, color in (("p_model", "#dd8452"), ("p_market", "#4c72b0")):
        sub = mz_table[mz_table["term"] == term].set_index("lead_time_hours").reindex(lead_times)
        coef = sub["coef"].to_numpy()
        err = 1.96 * sub["se"].to_numpy()
        ax.errorbar(x, coef, yerr=err, marker="o", capsize=4, color=color, label=term)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}h" for L in lead_times])
    ax.set_xlabel("Lead time before event")
    ax.set_ylabel("Mincer-Zarnowitz coefficient")
    ax.set_title("Forecast-encompassing coefficients vs lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

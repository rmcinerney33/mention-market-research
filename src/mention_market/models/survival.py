"""Cox proportional-hazards model on time-since-last-use.

Framing: treat each speech as an observation of the recurrence process for a
phrase. Given it has been ``days_since_last_use`` days since the phrase was last
used, what is the probability it recurs *at this speech*? We fit a Cox PH model
with ``duration = days_since_last_use`` and ``event = label`` on rows where the
phrase has a defined prior use, then predict per row as the model's CDF at the
observed inter-event duration, ``1 - S(t | x)``. Cold-start rows (phrase never
used before, duration undefined) fall back to the training base rate.

This is an intentionally simple survival mapping — documented as such — that
still yields calibratable per-event probabilities from a hazard model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .base import MentionModel

# Small, low-collinearity covariate set for the hazard model.
_COVARIATES = ["rr_hl30", "topical_relevance", "use_rate_10", "n_prior_events"]


class CoxSurvivalModel(MentionModel):
    name = "cox_survival"

    def __init__(self, penalizer: float = 0.1) -> None:
        self.penalizer = penalizer
        self.cph: CoxPHFitter | None = None
        self._base_rate = 0.5
        self._cov_means: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> CoxSurvivalModel:
        y = self._labels(train)
        self._base_rate = float(y.mean()) if len(y) else 0.5

        usable = train[(train["ever_used_before"] == 1) & train["days_since_last_use"].notna()].copy()
        # Need both classes and a few rows for a stable fit.
        if len(usable) < 20 or usable["label"].nunique() < 2:
            self.cph = None
            return self

        cols = [c for c in _COVARIATES if c in usable.columns]
        self._cov_means = usable[cols].mean()
        df = usable[cols].fillna(self._cov_means)
        df = df.assign(
            duration=usable["days_since_last_use"].clip(lower=0.1).to_numpy(),
            event=usable["label"].astype(int).to_numpy(),
        )
        try:
            cph = CoxPHFitter(penalizer=self.penalizer)
            cph.fit(df, duration_col="duration", event_col="event")
            self.cph = cph
            self._cov_cols = cols
        except Exception:
            # Convergence failures fall back to base rate rather than crashing a fold.
            self.cph = None
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(test), self._base_rate, dtype=float)
        if self.cph is None:
            return self._clip(preds)

        mask = (test["ever_used_before"] == 1) & test["days_since_last_use"].notna()
        if not mask.any():
            return self._clip(preds)

        sub = test.loc[mask]
        X = sub[self._cov_cols].fillna(self._cov_means)
        durations = sub["days_since_last_use"].clip(lower=0.1).to_numpy()
        # Survival at the observed duration, evaluated per-row on its own covariates.
        surv = self.cph.predict_survival_function(X, times=None)
        # surv columns are the subjects; index is time grid. Read S(t_i) per subject.
        probs = []
        time_grid = surv.index.to_numpy()
        for j, t in enumerate(durations):
            col = surv.iloc[:, j].to_numpy()
            idx = np.searchsorted(time_grid, t, side="right") - 1
            idx = min(max(idx, 0), len(col) - 1)
            probs.append(1.0 - col[idx])
        preds[mask.to_numpy()] = np.asarray(probs)
        return self._clip(preds)

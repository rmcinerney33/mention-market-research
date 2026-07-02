"""Bayesian hierarchical logistic regression with partial pooling.

Each phrase gets its own intercept, drawn from a shared hyperprior:

    mu_alpha ~ Normal(0, 1.5)
    sigma_alpha ~ HalfNormal(1)
    alpha[phrase] ~ Normal(mu_alpha, sigma_alpha)
    beta ~ Normal(0, 1)                       (standardized feature effects)
    logit = alpha[phrase] + X @ beta
    y ~ Bernoulli(sigmoid(logit))

Partial pooling lets small-N phrases borrow strength from the population, and a
phrase unseen in training is predicted from the population mean intercept
``mu_alpha`` — exactly the small-N borrowing the project calls for. Predictions
average sigmoid(logit) over posterior draws (proper posterior mean probability).

If sampling fails for any reason, the model degrades to the training base rate
rather than breaking a validation fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import MentionModel

_FEATURES = ["rr_hl30", "topical_relevance", "use_rate_10"]


class BayesianHierarchicalModel(MentionModel):
    name = "bayesian_hierarchical"

    def __init__(self, draws: int = 500, tune: int = 500, chains: int = 2,
                 target_accept: float = 0.9, seed: int = 42) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.seed = seed
        self._ok = False
        self._base_rate = 0.5

    def fit(self, train: pd.DataFrame) -> BayesianHierarchicalModel:
        import pymc as pm  # local import — heavy dependency

        y = self._labels(train)
        self._base_rate = float(y.mean()) if len(y) else 0.5
        if len(np.unique(y)) < 2 or len(train) < 30:
            self._ok = False
            return self

        X = train[_FEATURES].fillna(train[_FEATURES].median())
        self._mu = X.mean()
        self._sd = X.replace(0, np.nan).std().fillna(1.0).replace(0, 1.0)
        Xs = ((X - self._mu) / self._sd).to_numpy()

        phrases = pd.Categorical(train["phrase_id"])
        self._phrase_categories = list(phrases.categories)
        phrase_idx = phrases.codes

        try:
            with pm.Model():
                mu_alpha = pm.Normal("mu_alpha", 0.0, 1.5)
                sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
                alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, shape=len(self._phrase_categories))
                beta = pm.Normal("beta", 0.0, 1.0, shape=Xs.shape[1])
                logit = alpha[phrase_idx] + pm.math.dot(Xs, beta)
                pm.Bernoulli("obs", logit_p=logit, observed=y.astype(int))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    idata = pm.sample(
                        draws=self.draws, tune=self.tune, chains=self.chains,
                        cores=1, target_accept=self.target_accept,
                        random_seed=self.seed, progressbar=False,
                    )
            post = idata.posterior
            self._mu_alpha = post["mu_alpha"].to_numpy().reshape(-1)          # (S,)
            self._alpha = post["alpha"].to_numpy().reshape(-1, len(self._phrase_categories))  # (S, P)
            self._beta = post["beta"].to_numpy().reshape(-1, Xs.shape[1])     # (S, F)
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if not self._ok:
            return self._clip(np.full(len(test), self._base_rate))

        X = test[_FEATURES].fillna(self._mu)
        Xs = ((X - self._mu) / self._sd).to_numpy()  # (N, F)

        cat_index = {c: i for i, c in enumerate(self._phrase_categories)}
        # Per-row intercept draws: known phrase -> its alpha column; unknown -> mu_alpha.
        S = self._alpha.shape[0]
        n = len(test)
        alpha_draws = np.empty((S, n))
        for j, ph in enumerate(test["phrase_id"].to_numpy()):
            idx = cat_index.get(ph)
            alpha_draws[:, j] = self._alpha[:, idx] if idx is not None else self._mu_alpha

        # alpha_draws: (S, N); beta: (S, F); Xs: (N, F) -> feature term (S, N).
        logit = alpha_draws + self._beta @ Xs.T  # (S, N)
        probs = 1.0 / (1.0 + np.exp(-logit))
        return self._clip(probs.mean(axis=0))

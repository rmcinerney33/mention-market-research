"""Logistic regression on the engineered feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import MentionModel
from .preprocessing import make_linear_transformer


class LogisticModel(MentionModel):
    name = "logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.pipeline = Pipeline(
            [
                ("prep", make_linear_transformer()),
                ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight=None)),
            ]
        )
        self._degenerate: float | None = None

    def fit(self, train: pd.DataFrame) -> LogisticModel:
        y = self._labels(train)
        # A fold can be all-0 or all-1; LogisticRegression can't fit one class.
        if len(np.unique(y)) < 2:
            self._degenerate = float(y.mean()) if len(y) else 0.5
            return self
        self._degenerate = None
        self.pipeline.fit(train, y)
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._degenerate is not None:
            return self._clip(np.full(len(test), self._degenerate))
        p = self.pipeline.predict_proba(test)[:, 1]
        return self._clip(p)

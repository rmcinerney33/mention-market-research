"""Common model interface.

Every model in the suite implements the same contract so the validation layer
(Phase 5) can treat them interchangeably:

    model.fit(train_df).predict_proba(test_df) -> np.ndarray of P(label=1)

``train_df`` / ``test_df`` are feature tables produced by ``features.py`` and
must include the ``label`` column on train. Each model pulls the columns it
needs; none of them may look at anything but the row's own features (the
no-lookahead guarantee is already baked into how those features were built).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

EPS = 1e-6


class MentionModel(ABC):
    """Abstract base for all mention-prediction models."""

    #: Short identifier used in result tables and plots.
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> MentionModel:
        """Fit on a feature table that includes a ``label`` column."""

    @abstractmethod
    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        """Return P(label=1) for each row of ``test``; shape (len(test),)."""

    @staticmethod
    def _clip(p: np.ndarray | float) -> np.ndarray:
        """Keep probabilities inside (0, 1) so log-loss stays finite."""
        return np.clip(p, EPS, 1.0 - EPS)

    @staticmethod
    def _labels(df: pd.DataFrame) -> np.ndarray:
        if "label" not in df.columns:
            raise KeyError("training frame must contain a 'label' column")
        return df["label"].to_numpy(dtype=float)

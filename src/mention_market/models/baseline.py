"""Naive Laplace-smoothed base-rate model — the floor baseline.

Predicts, for each row, the Laplace-smoothed historical usage rate of that
(speaker, phrase) learned from the training fold, falling back to the global
smoothed rate for pairs unseen in training. It uses no recency, no context —
just identity base rates — so every other model must beat it to justify itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MentionModel

ALPHA = 0.5  # Laplace pseudo-count (matches features.LAPLACE_ALPHA)


class LaplaceBaseRateModel(MentionModel):
    name = "laplace_base_rate"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self._global_rate = 0.5
        self._pair_rate: dict[tuple[str, str], float] = {}

    def fit(self, train: pd.DataFrame) -> LaplaceBaseRateModel:
        y = self._labels(train)
        n = len(y)
        self._global_rate = (y.sum() + self.alpha) / (n + 2 * self.alpha) if n else 0.5

        self._pair_rate = {}
        for (sp, ph), grp in train.groupby(["speaker_id", "phrase_id"], sort=False):
            yy = grp["label"].to_numpy(dtype=float)
            self._pair_rate[(sp, ph)] = (yy.sum() + self.alpha) / (len(yy) + 2 * self.alpha)
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        preds = [
            self._pair_rate.get((row.speaker_id, row.phrase_id), self._global_rate)
            for row in test.itertuples(index=False)
        ]
        return self._clip(np.asarray(preds, dtype=float))

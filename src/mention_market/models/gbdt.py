"""Gradient-boosted decision trees on the full feature set.

Implementation note: this uses scikit-learn's ``HistGradientBoostingClassifier``
— a histogram-based GBDT, the same algorithmic family as LightGBM/XGBoost — so
the suite runs with no native OpenMP (``libomp``) dependency. On a machine with
``libomp`` available, swapping in ``lightgbm.LGBMClassifier`` or
``xgboost.XGBClassifier`` is a one-line change (both are pinned in
``requirements.txt``); the model interface here is unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import MentionModel
from .preprocessing import make_tree_transformer


class GBDTModel(MentionModel):
    name = "gbdt"

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 31, random_state: int = 42) -> None:
        self.transformer = make_tree_transformer()
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self._degenerate: float | None = None

    def fit(self, train: pd.DataFrame) -> GBDTModel:
        y = self._labels(train)
        if len(np.unique(y)) < 2:
            self._degenerate = float(y.mean()) if len(y) else 0.5
            return self
        self._degenerate = None
        X = self.transformer.fit_transform(train)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._degenerate is not None:
            return self._clip(np.full(len(test), self._degenerate))
        X = self.transformer.transform(test)
        p = self.clf.predict_proba(X)[:, 1]
        return self._clip(p)

"""Model suite — six comparably-evaluated mention-prediction models.

All share the :class:`~mention_market.models.base.MentionModel` interface:
``fit(train_df).predict_proba(test_df) -> np.ndarray``.
"""

from __future__ import annotations

from .base import MentionModel
from .baseline import LaplaceBaseRateModel
from .bayesian import BayesianHierarchicalModel
from .gbdt import GBDTModel
from .llm import LLMContextualModel
from .logistic import LogisticModel
from .survival import CoxSurvivalModel


def default_model_suite(include_bayesian: bool = True) -> list[MentionModel]:
    """Fresh instances of every model, in a sensible display order.

    ``include_bayesian`` lets callers skip the (slow) sampling model for quick
    iterations.
    """
    models: list[MentionModel] = [
        LaplaceBaseRateModel(),
        LogisticModel(),
        CoxSurvivalModel(),
        GBDTModel(),
    ]
    if include_bayesian:
        models.append(BayesianHierarchicalModel())
    models.append(LLMContextualModel())
    return models


__all__ = [
    "MentionModel",
    "LaplaceBaseRateModel",
    "LogisticModel",
    "CoxSurvivalModel",
    "GBDTModel",
    "BayesianHierarchicalModel",
    "LLMContextualModel",
    "default_model_suite",
]

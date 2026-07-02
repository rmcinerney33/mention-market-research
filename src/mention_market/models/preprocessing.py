"""Shared feature preprocessing for the sklearn-style models.

Numeric features are median-imputed (``days_since_last_use`` is NaN for
never-used phrases) and standardized; ``venue`` is one-hot encoded with unknown
categories ignored so a venue unseen in training doesn't break test scoring.
Trees don't need scaling, so ``make_tree_transformer`` skips it.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_linear_transformer() -> ColumnTransformer:
    """Impute + scale numerics, one-hot venues (for logistic regression)."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_tree_transformer() -> ColumnTransformer:
    """Impute numerics (no scaling), one-hot venues (for gradient boosting)."""
    numeric = SimpleImputer(strategy="median")
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )

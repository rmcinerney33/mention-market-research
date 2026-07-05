"""Load a research model and serve predictions *with uncertainty*.

The signal generator must record not just a point probability but the model's
own uncertainty (the spec makes this non-optional — it gates flagging in Phase
3). We get uncertainty in a model-agnostic way: fit ``n_bootstrap`` replicas of
the chosen model on bootstrap resamples of the training data, then a prediction's
interval is the percentile spread of the replicas. The point estimate comes from
the model fit on the full data.

The model itself is imported from the research package — never re-implemented
here — via ``default_model_factories``. Each provider carries a deterministic
``model_version`` (name + fingerprint hash of the training data and feature set)
so every stored signal is reproducible and auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mention_market import __version__ as _MM_VERSION
from mention_market.models import default_model_factories

_KEY_COLUMNS = ["event_id", "speaker_id", "phrase_id", "event_time", "label"]


@dataclass(frozen=True)
class Prediction:
    prob: float          # point estimate P(phrase used)
    ci_lo: float         # lower bootstrap quantile
    ci_hi: float         # upper bootstrap quantile
    n_bootstrap: int

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo


class ModelProvider:
    def __init__(
        self,
        model_name: str = "gbdt",
        *,
        n_bootstrap: int = 40,
        ci: tuple[float, float] = (0.05, 0.95),
        seed: int = 42,
    ) -> None:
        factories = default_model_factories(include_bayesian=False)
        if model_name not in factories:
            raise ValueError(
                f"unknown model {model_name!r}; choose from {sorted(factories)}"
            )
        self.model_name = model_name
        self._factory = factories[model_name]
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.seed = seed
        self._fitted = False

    def fit(self, training_table: pd.DataFrame) -> ModelProvider:
        if "label" not in training_table.columns:
            raise KeyError("training_table must contain a 'label' column")
        self.feature_columns = [c for c in training_table.columns if c not in _KEY_COLUMNS]
        self.n_train = len(training_table)

        self._model = self._factory().fit(training_table)

        rng = np.random.default_rng(self.seed)
        n = len(training_table)
        self._ensemble = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, n)
            resample = training_table.iloc[idx].reset_index(drop=True)
            self._ensemble.append(self._factory().fit(resample))

        self._hash = self._fingerprint(training_table)
        self._fitted = True
        return self

    def predict(self, feature_row: pd.DataFrame) -> Prediction:
        if not self._fitted:
            raise RuntimeError("ModelProvider.fit must be called before predict")
        point = float(np.asarray(self._model.predict_proba(feature_row))[0])
        if self._ensemble:
            boot = np.array([
                float(np.asarray(m.predict_proba(feature_row))[0]) for m in self._ensemble
            ])
            lo, hi = np.percentile(boot, [self.ci[0] * 100, self.ci[1] * 100])
        else:  # pragma: no cover - n_bootstrap=0 edge
            lo = hi = point
        return Prediction(prob=point, ci_lo=float(lo), ci_hi=float(hi), n_bootstrap=len(self._ensemble))

    @property
    def model_version(self) -> str:
        return f"{self.model_name}@{self._hash[:12]}"

    @property
    def model_hash(self) -> str:
        return self._hash

    def _fingerprint(self, training_table: pd.DataFrame) -> str:
        feats = training_table[self.feature_columns].fillna(0.0)
        data_hash = int(pd.util.hash_pandas_object(feats, index=False).sum() % (2**63))
        parts = [
            self.model_name,
            ",".join(sorted(self.feature_columns)),
            str(self.n_train),
            str(int(training_table["label"].fillna(0).sum())),
            str(data_hash),
            f"mm={_MM_VERSION}",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

"""Turn market snapshots into scored signals.

For each snapshot in a configured category: parse it into a prediction unit,
resolve its point-in-time features, and score it with the model (+ uncertainty).

Note on the validated gate: this stage computes and stores signals for **every
scanned category**, tagging each with whether its category is validated. That is
deliberate — you need signals on candidate categories to paper-trade them (Phase
6), which is exactly how a category earns validation. The hard "only act on
validated categories" rule is enforced at **flagging** time (Phase 3), not here.
With nothing validated today, signals still get recorded, but none can be
flagged for real.
"""

from __future__ import annotations

import logging

from .categories import CategoryMatcher
from .feature_resolver import ResearchFeatureResolver
from .model_provider import ModelProvider
from .signal import Signal

logger = logging.getLogger("kalshi_scanner")


class SignalGenerator:
    def __init__(
        self,
        matcher: CategoryMatcher,
        model_provider: ModelProvider,
        feature_resolver: ResearchFeatureResolver,
        *,
        store=None,
    ) -> None:
        self.matcher = matcher
        self.model = model_provider
        self.resolver = feature_resolver
        self.store = store

    def generate(self, snapshots, scan_id: int | None = None) -> list[Signal]:
        signals: list[Signal] = []
        for snap in snapshots:
            category = snap.category
            validated = self.matcher.is_validated(category)

            query = self._parse(snap)
            if query is None:
                signals.append(Signal.skipped(snap, category, validated, "unparseable_market", scan_id))
                continue

            row = self.resolver.resolve(query, snap.scan_ts)
            if row is None:
                signals.append(Signal.skipped(snap, category, validated, "no_features", scan_id))
                continue

            prediction = self.model.predict(row)
            features = self._features_dict(row)
            signals.append(
                Signal.scored(
                    snap, category, validated, prediction,
                    model_version=self.model.model_version,
                    model_hash=self.model.model_hash,
                    features=features, scan_id=scan_id,
                )
            )

        if self.store is not None and scan_id is not None:
            self.store.record_signals(signals)
        n_scored = sum(s.is_scored for s in signals)
        logger.info("signals: %d scored, %d skipped (validated categories: %s)",
                    n_scored, len(signals) - n_scored,
                    [c for c in self.matcher.category_names if self.matcher.is_validated(c)] or "[]")
        return signals

    @staticmethod
    def _parse(snap):
        from .mention_query import parse_mention_market
        return parse_mention_market(snap)

    def _features_dict(self, row) -> dict:
        out = {}
        for col in self.model.feature_columns:
            if col not in row.columns:
                continue
            val = row.iloc[0][col]
            try:
                out[col] = None if val is None or (isinstance(val, float) and val != val) else (
                    val.item() if hasattr(val, "item") else val
                )
            except (TypeError, ValueError):
                out[col] = str(val)
        return out


def build_default_signal_generator(
    config, store=None, *, model_name: str = "gbdt", n_bootstrap: int = 40, seed: int = 42
) -> SignalGenerator:
    """Wire a generator from the synthetic research world (the placeholder corpus).

    Swap the world for real ingested data when it exists — nothing else changes.
    """
    from .feature_resolver import (
        ResearchFeatureResolver,
        build_training_table,
        default_research_world,
    )

    world = default_research_world(seed=seed)
    training = build_training_table(world)
    provider = ModelProvider(model_name, n_bootstrap=n_bootstrap, seed=seed).fit(training)
    resolver = ResearchFeatureResolver(world.events, world.transcripts, world.gdelt)
    matcher = CategoryMatcher(config.categories)
    return SignalGenerator(matcher, provider, resolver, store=store)

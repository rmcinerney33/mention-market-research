"""The Signal record.

One row per market per signal run. Either a *scored* signal (model probability
+ bootstrap CI + the exact model version and features used) or a *skipped* one
with a machine-readable ``reason``. Everything needed to reproduce or audit the
number later is captured inline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime


def market_implied_prob(snapshot) -> float | None:
    """Reference implied probability from the book mid (cents -> [0,1]).

    Phase 3 will switch to the *executable* side (the ask if we'd buy YES); the
    mid is only a convenience reference here.
    """
    yb, ya = snapshot.yes_bid, snapshot.yes_ask
    if yb is not None and ya is not None and (yb > 0 or ya > 0):
        return (yb + ya) / 2.0 / 100.0
    if snapshot.last_price is not None:
        return snapshot.last_price / 100.0
    return None


@dataclass
class Signal:
    scan_id: int | None
    ticker: str
    event_ticker: str | None
    scan_ts: datetime
    category: str
    validated: bool
    event_time: datetime | None
    market_yes_bid: int | None
    market_yes_ask: int | None
    market_no_bid: int | None
    market_no_ask: int | None
    market_implied_prob: float | None
    model_prob: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    model_version: str | None = None
    model_hash: str | None = None
    features_json: str | None = None
    reason: str | None = None  # None => a real scored signal; else why it was skipped

    @property
    def is_scored(self) -> bool:
        return self.model_prob is not None

    @classmethod
    def _base(cls, snapshot, category: str, validated: bool, scan_id: int | None) -> dict:
        return {
            "scan_id": scan_id,
            "ticker": snapshot.ticker,
            "event_ticker": snapshot.event_ticker,
            "scan_ts": snapshot.scan_ts,
            "category": category,
            "validated": validated,
            "event_time": snapshot.close_time,
            "market_yes_bid": snapshot.yes_bid,
            "market_yes_ask": snapshot.yes_ask,
            "market_no_bid": snapshot.no_bid,
            "market_no_ask": snapshot.no_ask,
            "market_implied_prob": market_implied_prob(snapshot),
        }

    @classmethod
    def skipped(cls, snapshot, category, validated, reason, scan_id=None) -> Signal:
        return cls(**cls._base(snapshot, category, validated, scan_id), reason=reason)

    @classmethod
    def scored(cls, snapshot, category, validated, prediction, *, model_version, model_hash,
               features: dict, scan_id=None) -> Signal:
        return cls(
            **cls._base(snapshot, category, validated, scan_id),
            model_prob=prediction.prob,
            ci_lo=prediction.ci_lo,
            ci_hi=prediction.ci_hi,
            model_version=model_version,
            model_hash=model_hash,
            features_json=json.dumps(features, default=str),
        )

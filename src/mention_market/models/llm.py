"""LLM-based contextual probability estimate — a distinct modeling paradigm.

Instead of fitting parameters, this model *reasons* about each event from a
natural-language description of the recent context (usage stats, recency, venue,
topical salience) and returns a calibrated probability.

Two execution paths:

- **Live** (when ``ANTHROPIC_API_KEY`` is set and ``use_api`` allows it): prompts
  Claude with structured output constrained to a JSON ``{"probability": ...}``.
  The default model is ``claude-opus-4-8``; override via config for a cheaper
  model (e.g. ``claude-haiku-4-5``) since this runs once per event.
- **Offline heuristic** (default when no key): a transparent, deterministic
  monotone blend of the same features. This keeps the whole pipeline runnable
  and reproducible without credentials or network, and stands in for the LLM's
  qualitative reasoning. It is clearly labeled as a stand-in in the write-up.

``fit`` only records the training base rate (used to anchor the heuristic and as
a fallback); the model is otherwise zero-shot.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .base import MentionModel

_PROMPT_FEATURES = [
    "lap_smoothed_rate", "rr_hl7", "rr_hl30", "use_rate_10",
    "days_since_last_use", "topical_relevance", "venue",
]


class LLMContextualModel(MentionModel):
    name = "llm_contextual"

    def __init__(self, model: str = "claude-opus-4-8", use_api: bool | None = None,
                 max_calls: int | None = None) -> None:
        self.model = model
        # Auto: use the API only if a key is present; explicit override wins.
        self.use_api = (os.environ.get("ANTHROPIC_API_KEY") is not None) if use_api is None else use_api
        self.max_calls = max_calls
        self._base_rate = 0.5
        self._cache: dict[str, float] = {}
        self._client = None

    def fit(self, train: pd.DataFrame) -> LLMContextualModel:
        y = self._labels(train)
        self._base_rate = float(y.mean()) if len(y) else 0.5
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        estimator = self._claude_probability if self.use_api else self._heuristic_probability
        preds = np.empty(len(test), dtype=float)
        calls = 0
        for i, row in enumerate(test.itertuples(index=False)):
            key = getattr(row, "event_id", str(i))
            if key in self._cache:
                preds[i] = self._cache[key]
                continue
            if self.use_api and self.max_calls is not None and calls >= self.max_calls:
                preds[i] = self._heuristic_probability(row)  # budget exhausted -> offline
            else:
                preds[i] = estimator(row)
                calls += int(self.use_api)
            self._cache[key] = preds[i]
        return self._clip(preds)

    # ------------------------------------------------------------------
    # Offline heuristic — deterministic stand-in for LLM reasoning
    # ------------------------------------------------------------------
    def _heuristic_probability(self, row) -> float:
        """Monotone blend of recency-weighted usage, rolling rate, and topic.

        Weights are fixed (not fit) so the paradigm stays genuinely distinct
        from the trained models. Anchored on the (point-in-time) smoothed rate.
        """
        lap = _get(row, "lap_smoothed_rate", self._base_rate)
        rr = _get(row, "rr_hl30", lap)
        rate10 = _get(row, "use_rate_10", lap)
        topical = _get(row, "topical_relevance", 0.5)
        base = 0.40 * lap + 0.35 * rr + 0.25 * rate10
        # Topical salience nudges up/down around 0.5.
        adj = base + 0.15 * (topical - 0.5)
        return float(np.clip(adj, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Live path — Claude with structured output
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _claude_probability(self, row) -> float:
        try:
            client = self._get_client()
            prompt = _build_prompt(row)
            resp = client.messages.create(
                model=self.model,
                max_tokens=256,
                thinking={"type": "disabled"},
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"probability": {"type": "number"}},
                            "required": ["probability"],
                            "additionalProperties": False,
                        },
                    }
                },
                system=(
                    "You estimate the probability that a public figure will use a "
                    "specific phrase in an upcoming speech, given recent usage "
                    "statistics and context. Return only a probability in [0,1]. "
                    "Use no information beyond what is provided."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            prob = float(json.loads(text)["probability"])
            return float(np.clip(prob, 0.01, 0.99))
        except Exception:
            # Any API/parse failure degrades gracefully to the offline heuristic.
            return self._heuristic_probability(row)


def _get(row, attr: str, default: float) -> float:
    val = getattr(row, attr, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _build_prompt(row) -> str:
    parts = ["Estimate P(the phrase is used in the next speech).", "", "Context:"]
    for f in _PROMPT_FEATURES:
        val = getattr(row, f, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = "unknown"
        parts.append(f"- {f}: {val}")
    parts.append("")
    parts.append('Respond with JSON: {"probability": <number between 0 and 1>}.')
    return "\n".join(parts)

"""Category matching.

Decides which configured category (if any) a raw Kalshi market belongs to, by
title regex, series ticker, or Kalshi's own category field. Also answers whether
a category is *validated* — the gate that later phases use before flagging.
"""

from __future__ import annotations

import re

from .config import CategoryRule


class CategoryMatcher:
    def __init__(self, rules: list[CategoryRule]) -> None:
        self._rules = rules
        self._patterns: dict[str, list[re.Pattern[str]]] = {
            r.name: [re.compile(p, re.IGNORECASE) for p in r.title_patterns] for r in rules
        }
        self._validated = {r.name: r.validated for r in rules}

    def match(self, market: dict) -> str | None:
        """Return the name of the first category this market matches, else None."""
        title = market.get("title", "") or ""
        event_ticker = market.get("event_ticker")
        series = market.get("series_ticker")
        if not series and event_ticker and "-" in event_ticker:
            series = event_ticker.split("-", 1)[0]
        kalshi_category = market.get("category")

        for rule in self._rules:
            if series and series in rule.series_tickers:
                return rule.name
            if kalshi_category and kalshi_category in rule.kalshi_categories:
                return rule.name
            if any(pat.search(title) for pat in self._patterns[rule.name]):
                return rule.name
        return None

    def is_validated(self, category_name: str) -> bool:
        return self._validated.get(category_name, False)

    @property
    def category_names(self) -> list[str]:
        return [r.name for r in self._rules]

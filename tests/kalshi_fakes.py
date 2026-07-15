"""Test doubles and fixture loaders for the Kalshi scanner (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "kalshi"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        json_data: Any = None,
        headers: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.text = text or (json.dumps(json_data) if json_data is not None else "")

    def json(self) -> Any:
        return self._json


class FakeSession:
    """A requests.Session stand-in.

    Provide either a fixed ``responses`` queue (popped in order) or a
    ``handler(method, url, params)`` returning a :class:`FakeResponse`. Every
    call is recorded on ``self.calls``.
    """

    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        handler: Callable[[str, str, dict | None], FakeResponse] | None = None,
    ) -> None:
        self._responses = list(responses) if responses else None
        self._handler = handler
        self.calls: list[dict] = []

    def request(self, method, url, headers=None, params=None, timeout=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers or {}, "params": params or {}}
        )
        if self._handler is not None:
            return self._handler(method, url, params)
        if not self._responses:
            raise AssertionError("FakeSession ran out of programmed responses")
        return self._responses.pop(0)


class StubClient:
    """Minimal client stand-in for Scanner tests.

    ``markets`` are yielded by :meth:`iter_markets`; if ``error`` is set, the
    iteration raises it (to exercise downtime handling).
    """

    def __init__(self, markets: list[dict] | None = None, error: Exception | None = None) -> None:
        self._markets = markets or []
        self._error = error
        self.calls: list[dict] = []

    def iter_markets(self, *, status="open", series_ticker=None, page_limit=1000, max_pages=None):
        self.calls.append(
            {"status": status, "series_ticker": series_ticker, "max_pages": max_pages}
        )
        if self._error is not None:
            raise self._error
        yield from self._markets

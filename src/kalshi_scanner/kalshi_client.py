"""A small, mockable Kalshi REST client.

Design goals:

- **Read-only.** Only GET endpoints are implemented. There is no order code.
- **Mockable.** The HTTP transport (``session``) is injectable, so tests run
  entirely off recorded JSON fixtures with no network.
- **Polite.** A token-bucket rate limiter plus exponential backoff on ``429`` /
  ``5xx`` (honoring ``Retry-After``) keeps us within Kalshi's limits.
- **Auth is optional.** Public market reads need no credentials. When an API key
  + RSA private key are configured, requests are signed (RSA-PSS/SHA-256).
  Signing pulls in ``cryptography`` lazily, so the package imports without it.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("kalshi_scanner")


class KalshiError(Exception):
    """Base error for the Kalshi client."""


class KalshiAPIError(KalshiError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Kalshi API {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class KalshiAuthRequired(KalshiError):
    """Raised when an authenticated endpoint is called without credentials."""


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Token-bucket limiter. ``acquire`` blocks until a token is available."""

    def __init__(
        self,
        rate_per_sec: float,
        burst: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate = max(rate_per_sec, 1e-9)
        self.capacity = max(burst, 1)
        self._tokens = float(self.capacity)
        self._monotonic = monotonic
        self._sleep = sleep
        self._last = monotonic()

    def acquire(self, n: int = 1) -> None:
        while True:
            now = self._monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return
            self._sleep((n - self._tokens) / self.rate)


# ---------------------------------------------------------------------------
# Request signing (lazy cryptography)
# ---------------------------------------------------------------------------
class _RsaSigner:
    def __init__(self, private_key_pem: str) -> None:
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError as e:  # pragma: no cover - only when creds are set
            raise KalshiError(
                "Kalshi credentials are configured but the 'cryptography' package "
                "is not installed. Run: pip install cryptography"
            ) from e

        self._padding = padding
        self._hashes = hashes
        self._key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)

    def sign(self, message: str) -> str:
        signature = self._key.sign(
            message.encode(),
            self._padding.PSS(
                mgf=self._padding.MGF1(self._hashes.SHA256()),
                salt_length=self._padding.PSS.DIGEST_LENGTH,
            ),
            self._hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class KalshiClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key_id: str | None = None,
        private_key_pem: str | None = None,
        session: Any = None,
        rate_limiter: RateLimiter | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        backoff_jitter_frac: float = 0.1,
        sleep: Callable[[float], None] = time.sleep,
        time_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        rng: random.Random | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.backoff_jitter_frac = backoff_jitter_frac
        self._sleep = sleep
        self._time_ms = time_ms
        self._rng = rng or random.Random()

        self._api_key_id = api_key_id
        self._signer = _RsaSigner(private_key_pem) if (api_key_id and private_key_pem) else None

        if session is None:
            import requests  # local import so the package imports without requests

            session = requests.Session()
        self._session = session
        self._rate_limiter = rate_limiter or RateLimiter(5.0, 5, sleep=sleep)

    @property
    def authenticated(self) -> bool:
        return self._signer is not None

    # -- signing ---------------------------------------------------------
    def _auth_headers(self, method: str, url: str) -> dict[str, str]:
        if self._signer is None:
            return {}
        ts = str(self._time_ms())
        path = urlparse(url).path
        signature = self._signer.sign(ts + method.upper() + path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id or "",
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    # -- core request with retry/backoff ---------------------------------
    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        url = self.base_url + path
        headers = {"Accept": "application/json", **self._auth_headers(method, url)}

        last_message = ""
        for attempt in range(self.max_retries + 1):
            self._rate_limiter.acquire()
            resp = self._session.request(
                method, url, headers=headers, params=params, timeout=self.timeout_s
            )
            status = resp.status_code

            if 200 <= status < 300:
                return resp.json()

            if status == 429 or 500 <= status < 600:
                last_message = _safe_text(resp)
                if attempt >= self.max_retries:
                    break
                self._sleep(self._backoff_delay(attempt, resp))
                logger.warning("Kalshi %s %s -> %d, retrying (attempt %d)",
                               method, path, status, attempt + 1)
                continue

            # Non-retryable 4xx.
            raise KalshiAPIError(status, _safe_text(resp))

        raise KalshiAPIError(429 if last_message else 500, last_message or "exhausted retries")

    def _backoff_delay(self, attempt: int, resp: Any) -> float:
        retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
        if retry_after:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass
        delay = min(self.backoff_base_s * (2 ** attempt), self.backoff_max_s)
        if self.backoff_jitter_frac:
            delay += self._rng.uniform(0.0, self.backoff_jitter_frac * delay)
        return delay

    # -- endpoints -------------------------------------------------------
    def iter_markets(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        page_limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterator[dict]:
        """Yield market dicts across all pages, following Kalshi's cursor."""
        cursor: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {"limit": page_limit, "status": status}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor

            payload = self._request("GET", "/markets", params=params)
            yield from payload.get("markets", [])

            pages += 1
            cursor = payload.get("cursor") or None
            if not cursor or (max_pages is not None and pages >= max_pages):
                return

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict:
        """Order book for one market. Requires authentication."""
        if not self.authenticated:
            raise KalshiAuthRequired(
                f"orderbook for {ticker} requires Kalshi API credentials "
                "(set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH/PEM)"
            )
        params = {"depth": depth} if depth is not None else None
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)


def _safe_text(resp: Any) -> str:
    try:
        return resp.text
    except Exception:  # pragma: no cover - defensive
        return "<no body>"

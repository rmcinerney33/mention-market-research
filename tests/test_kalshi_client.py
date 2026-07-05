"""Tests for the Kalshi client: pagination, backoff, rate limiting, auth."""

from __future__ import annotations

import base64
from urllib.parse import urlparse

import pytest

from kalshi_scanner.kalshi_client import (
    KalshiAPIError,
    KalshiAuthRequired,
    KalshiClient,
    RateLimiter,
)
from tests.kalshi_fakes import FakeResponse, FakeSession, load_fixture

BASE = "https://api.test/trade-api/v2"


def _client(session, **kw):
    kw.setdefault("backoff_jitter_frac", 0.0)
    kw.setdefault("backoff_base_s", 0.5)
    return KalshiClient(BASE, session=session, sleep=kw.pop("sleep", lambda _: None), **kw)


# -- pagination -------------------------------------------------------------
def test_pagination_follows_cursor():
    def handler(method, url, params):
        if not params.get("cursor"):
            return FakeResponse(200, load_fixture("markets_page1.json"))
        return FakeResponse(200, load_fixture("markets_page2.json"))

    session = FakeSession(handler=handler)
    client = _client(session)
    markets = list(client.iter_markets(status="open"))

    assert [m["ticker"] for m in markets] == [
        "MENTIONSPEECH-24NOV05-TARIFF", "CPI-24NOV-ABOVE35", "MENTIONSPEECH-24NOV07-INFLATION",
    ]
    # Second request carried the cursor from page 1.
    assert session.calls[1]["params"].get("cursor") == "CURSOR_PAGE2"


def test_pagination_stops_on_empty_cursor():
    session = FakeSession([FakeResponse(200, {"markets": [{"ticker": "A"}], "cursor": ""})])
    client = _client(session)
    assert len(list(client.iter_markets())) == 1


# -- backoff ----------------------------------------------------------------
def test_retries_on_429_then_succeeds():
    sleeps: list[float] = []
    session = FakeSession([
        FakeResponse(429, text="rate limited"),
        FakeResponse(200, {"markets": [], "cursor": ""}),
    ])
    client = _client(session, sleep=sleeps.append, backoff_base_s=0.5)
    list(client.iter_markets())
    assert sleeps == [0.5]  # 2^0 * base, no jitter


def test_honors_retry_after_header():
    sleeps: list[float] = []
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}, text="slow down"),
        FakeResponse(200, {"markets": [], "cursor": ""}),
    ])
    client = _client(session, sleep=sleeps.append)
    list(client.iter_markets())
    assert sleeps == [7.0]


def test_exhausts_retries_and_raises():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(429, text="nope")] * 6)
    client = _client(session, sleep=sleeps.append, max_retries=3)
    with pytest.raises(KalshiAPIError) as ei:
        list(client.iter_markets())
    assert ei.value.status_code == 429
    assert len(sleeps) == 3  # retried exactly max_retries times


def test_5xx_is_retried():
    sleeps: list[float] = []
    session = FakeSession([
        FakeResponse(503, text="unavailable"),
        FakeResponse(200, {"markets": [], "cursor": ""}),
    ])
    client = _client(session, sleep=sleeps.append)
    list(client.iter_markets())
    assert len(sleeps) == 1


def test_non_retryable_4xx_raises_immediately():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(404, text="not found")])
    client = _client(session, sleep=sleeps.append)
    with pytest.raises(KalshiAPIError) as ei:
        list(client.iter_markets())
    assert ei.value.status_code == 404
    assert sleeps == []


# -- auth -------------------------------------------------------------------
def test_orderbook_requires_auth():
    client = _client(FakeSession([]))
    assert not client.authenticated
    with pytest.raises(KalshiAuthRequired):
        client.get_orderbook("SOME-TICKER")


def test_unauthenticated_sends_no_signing_headers():
    session = FakeSession([FakeResponse(200, {"markets": [], "cursor": ""})])
    client = _client(session)
    list(client.iter_markets())
    assert "KALSHI-ACCESS-KEY" not in session.calls[0]["headers"]


def test_signing_produces_valid_signature():
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()

    session = FakeSession([FakeResponse(200, {"markets": [], "cursor": ""})])
    client = _client(session, api_key_id="kid-123", private_key_pem=pem,
                     time_ms=lambda: 1700000000000)
    assert client.authenticated
    list(client.iter_markets())

    call = session.calls[0]
    headers = call["headers"]
    assert headers["KALSHI-ACCESS-KEY"] == "kid-123"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"

    path = urlparse(call["url"]).path
    message = "1700000000000" + "GET" + path
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    # Raises InvalidSignature if the client signed the wrong bytes.
    key.public_key().verify(
        signature, message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    assert crypto  # imported


# -- rate limiter -----------------------------------------------------------
def test_rate_limiter_blocks_when_tokens_exhausted():
    now = [0.0]
    sleeps: list[float] = []

    def fake_sleep(d):
        sleeps.append(d)
        now[0] += d  # advance virtual clock so tokens refill

    rl = RateLimiter(rate_per_sec=2.0, burst=2, monotonic=lambda: now[0], sleep=fake_sleep)
    for _ in range(4):
        rl.acquire()
    # burst=2 free, then 2 more each need 0.5s at 2/s.
    assert len(sleeps) >= 2
    assert all(s > 0 for s in sleeps)

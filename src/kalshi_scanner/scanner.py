"""Scan orchestration.

One :meth:`Scanner.scan_once` pulls open markets, keeps those matching a
configured category, snapshots them, and persists the result — recording the
run's success or failure either way. :meth:`Scanner.run_forever` repeats that on
the configured interval.

Downtime is handled by *recording it*, not hiding it: any exception during a
scan is caught, logged, and written as a ``failed`` run with zero snapshots. We
never interpolate over a gap.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator

from .categories import CategoryMatcher
from .config import ScannerConfig
from .kalshi_client import KalshiClient
from .market import MarketSnapshot, ScanResult
from .store import SnapshotStore, utcnow

logger = logging.getLogger("kalshi_scanner")


class Scanner:
    def __init__(
        self,
        config: ScannerConfig,
        client: KalshiClient,
        store: SnapshotStore,
        *,
        clock: Callable[[], object] = utcnow,
        sleep: Callable[[float], None] = time.sleep,
        matcher: CategoryMatcher | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.store = store
        self._clock = clock
        self._sleep = sleep
        self.matcher = matcher or CategoryMatcher(config.categories)

    def _iter_relevant_markets(self) -> Iterator[dict]:
        """Yield candidate markets, preferring cheap server-side queries and
        never exceeding the ``max_markets_per_scan`` hard cap.

        Kalshi's ``/markets`` endpoint can filter by series server-side but not
        by title, so:
        - categories with ``series_tickers`` are fetched with one targeted query
          each (cheap — a series is small);
        - categories that match only on title need a full-market crawl (Kalshi
          has >1M open markets), which is gated behind ``allow_full_scan`` and,
          when enabled, still bounded by the cap.
        """
        cap = self.config.max_markets_per_scan
        page_limit = 1000
        max_pages = math.ceil(cap / page_limit) if cap and cap > 0 else None

        cats = self.config.categories
        series = sorted({s for c in cats for s in c.series_tickers})
        title_only = [c for c in cats if not c.series_tickers]

        seen: set[str] = set()
        count = 0

        def _fresh(market: dict) -> bool:
            nonlocal count
            tk = market.get("ticker")
            if tk in seen:
                return False
            if tk:
                seen.add(tk)
            count += 1
            return True

        # 1) Targeted, server-side: one query per configured series.
        for st in series:
            for m in self.client.iter_markets(
                status=self.config.market_status, series_ticker=st, max_pages=max_pages
            ):
                if _fresh(m):
                    yield m
                if cap and count >= cap:
                    logger.warning("scan hit max_markets_per_scan=%d cap (series stage)", cap)
                    return

        # 2) Title-only categories: no server-side title filter exists.
        if title_only:
            names = [c.name for c in title_only]
            if not self.config.allow_full_scan:
                logger.warning(
                    "skipping title-only categor%s %s: matching them requires crawling "
                    "ALL open markets (allow_full_scan=false). Add series_tickers to scope "
                    "the query, or set allow_full_scan=true (bounded by "
                    "max_markets_per_scan=%d).",
                    "y" if len(names) == 1 else "ies", names, cap,
                )
                return
            logger.warning(
                "allow_full_scan=true: crawling open markets for %s (hard cap %d markets). "
                "This is the expensive path — prefer series_tickers when possible.", names, cap,
            )
            for m in self.client.iter_markets(
                status=self.config.market_status, max_pages=max_pages
            ):
                if _fresh(m):
                    yield m
                if cap and count >= cap:
                    logger.warning("scan hit max_markets_per_scan=%d cap (full-scan stage)", cap)
                    return

    def scan_once(self) -> ScanResult:
        scan_ts = self._clock()

        if not self.config.enabled:
            logger.info("scanning disabled (config.enabled=false); recording skipped run")
            scan_id = self.store.start_run(scan_ts, self.matcher.category_names)
            self.store.finish_run(scan_id, self._clock(), "skipped", 0)
            return ScanResult(scan_ts, ok=False, n_markets=0, error="disabled")

        if self.config.kill_switch:
            logger.warning(
                "kill_switch is ON: flagging is paused. Scanning still runs "
                "(Phase 1 produces no flags), so snapshots continue to accrue."
            )

        scan_id = self.store.start_run(scan_ts, self.matcher.category_names)
        snapshots: list[MarketSnapshot] = []
        n_seen = 0
        try:
            for market in self._iter_relevant_markets():
                n_seen += 1
                category = self.matcher.match(market)
                if category is None:
                    continue
                snapshots.append(
                    MarketSnapshot.from_api(market, scan_ts=scan_ts, category=category)
                )
            self.store.record_snapshots(scan_id, snapshots)
            self.store.finish_run(scan_id, self._clock(), "ok", len(snapshots))
            logger.info(
                "scan ok: matched %d/%d markets in %s",
                len(snapshots), n_seen, self.matcher.category_names,
            )
            return ScanResult(
                scan_ts, ok=True, n_markets=len(snapshots), snapshots=snapshots
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.store.finish_run(scan_id, self._clock(), "failed", 0, error=msg)
            logger.error("scan failed (gap logged, no interpolation): %s", msg)
            return ScanResult(scan_ts, ok=False, n_markets=0, error=msg)

    def run_forever(self, max_iterations: int | None = None) -> None:
        i = 0
        while True:
            self.scan_once()
            i += 1
            if max_iterations is not None and i >= max_iterations:
                return
            self._sleep(self.config.poll_interval_s)


def build_scanner(config: ScannerConfig) -> Scanner:
    """Wire a live Scanner from config (real HTTP client + on-disk store)."""
    from .kalshi_client import RateLimiter

    client = KalshiClient(
        base_url=config.api.base_url,
        api_key_id=config.api.api_key_id,
        private_key_pem=config.api.private_key_pem,
        timeout_s=config.api.timeout_s,
        max_retries=config.rate_limit.max_retries,
        backoff_base_s=config.rate_limit.backoff_base_s,
        backoff_max_s=config.rate_limit.backoff_max_s,
        rate_limiter=RateLimiter(
            config.rate_limit.max_requests_per_sec, config.rate_limit.burst
        ),
    )
    store = SnapshotStore(config.db_path)
    return Scanner(config, client, store)

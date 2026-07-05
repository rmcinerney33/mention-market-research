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
        cats = self.config.categories
        # Only narrow the query to specific series if *every* category is
        # series-scoped; otherwise a title/category-matched market would be
        # missed. The mention placeholder is title-based, so we scan all.
        if cats and all(c.series_tickers for c in cats):
            series = sorted({s for c in cats for s in c.series_tickers})
            for st in series:
                yield from self.client.iter_markets(
                    status=self.config.market_status, series_ticker=st
                )
        else:
            yield from self.client.iter_markets(status=self.config.market_status)

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

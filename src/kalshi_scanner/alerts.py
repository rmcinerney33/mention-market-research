"""Optional push notifications for new flags.

Disabled by default. When enabled, posts a short message per qualifying flag to
an ntfy.sh topic (a simple authenticated-by-obscurity HTTP push). The transport
is injectable so tests never touch the network, and only flags whose
EV*size clears ``min_ev_notional`` are sent — you don't want a buzz for a
one-cent edge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .config import AlertConfig

logger = logging.getLogger("kalshi_scanner")

# transport(url, data, headers) -> HTTP status code
Transport = Callable[[str, bytes, dict], int]


def _requests_transport(url: str, data: bytes, headers: dict) -> int:
    import requests

    return requests.post(url, data=data, headers=headers, timeout=10).status_code


class Alerter:
    def __init__(self, config: AlertConfig, *, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or _requests_transport

    def notify_flags(self, flags: list) -> list:
        """Send an alert per qualifying flag; returns the flags actually sent."""
        c = self.config
        if not c.enabled:
            return []
        if c.channel != "ntfy" or not c.ntfy_topic:
            logger.warning("alerts enabled but no usable channel/topic; skipping")
            return []

        sent = []
        for f in flags:
            value = (f.ev_per_contract or 0.0) * f.contracts
            if value < c.min_ev_notional:
                continue
            title = f"FLAG {f.ticker} {f.side}"
            body = (f"model {f.model_prob:.2f} vs px {f.market_price:.2f} | "
                    f"EV/ctr {f.ev_per_contract:+.3f} | size {f.contracts} | "
                    f"~${value:.2f} expected")
            try:
                status = self._transport(
                    f"{c.ntfy_server}/{c.ntfy_topic}", body.encode(),
                    {"Title": title, "Tags": "chart_with_upwards_trend"},
                )
                if 200 <= status < 300:
                    sent.append(f)
                else:
                    logger.warning("alert POST for %s returned %d", f.ticker, status)
            except Exception as e:  # never let alerting break the pipeline
                logger.error("alert POST failed for %s: %s", f.ticker, e)
        return sent

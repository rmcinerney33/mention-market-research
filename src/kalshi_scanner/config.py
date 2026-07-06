"""Typed configuration for the Kalshi scanner.

Mirrors the research package's YAML + ``config.local.yaml`` override pattern
(and reuses its :func:`repo_root`), but produces typed dataclasses because the
downstream sizing/gating math wants strong types, not loose dicts.

Secrets never live in the YAML — they are read from the environment here.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mention_market.config import repo_root

logger = logging.getLogger("kalshi_scanner")

_DEFAULT_CONFIG_PATH = repo_root() / "kalshi_scanner.yaml"
_LOCAL_OVERRIDE_PATH = repo_root() / "kalshi_scanner.local.yaml"


@dataclass(frozen=True)
class CategoryRule:
    """One scannable category and whether it has a validated edge.

    A market matches this category if *any* configured rule matches: its title
    matches one of ``title_patterns`` (regex, case-insensitive), its
    ``series_ticker`` is in ``series_tickers``, or its Kalshi ``category`` is in
    ``kalshi_categories``.
    """

    name: str
    validated: bool = False
    title_patterns: list[str] = field(default_factory=list)
    series_tickers: list[str] = field(default_factory=list)
    kalshi_categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RateLimitConfig:
    max_requests_per_sec: float = 5.0
    burst: int = 5
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0


@dataclass(frozen=True)
class TradingConfig:
    """Sizing/gating parameters for the edge calculator (Phase 3).

    None of this places orders — it only decides what a human would be shown as
    flaggable. Real-money use is gated behind paper trading regardless.
    """

    bankroll: float = 10_000.0
    kelly_fraction: float = 0.25          # quarter Kelly by default
    max_position_frac: float = 0.02       # hard cap: 2% of bankroll per position
    fee_coefficient: float = 0.07         # Kalshi general fee coef (re-confirm vs schedule)
    ci_margin: float = 0.0                # extra margin beyond the model CI for the gate
    min_ev_per_contract: float = 0.0      # minimum fee/slippage-adjusted EV to flag (dollars)
    slippage_buffer_cents: float = 1.0    # conservative buffer when depth is unknown
    require_book_for_flag: bool = True    # refuse to flag without real order-book depth
    # Portfolio exposure caps (Phase 4), as fractions of bankroll:
    max_event_frac: float = 0.05          # any single event (e.g. one speech)
    max_category_frac: float = 0.10       # any single category
    max_total_frac: float = 0.25          # overall


@dataclass(frozen=True)
class AlertConfig:
    """Optional push notifications for new flags (Phase 5). Disabled by default."""

    enabled: bool = False
    channel: str = "ntfy"                 # "ntfy" (HTTP) supported
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str = ""
    min_ev_notional: float = 1.0          # only alert flags with EV*size >= this ($)


@dataclass(frozen=True)
class KalshiApiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    timeout_s: float = 10.0
    # Secrets, resolved from the environment (None => unauthenticated reads).
    api_key_id: str | None = None
    private_key_pem: str | None = None

    @property
    def authenticated(self) -> bool:
        return bool(self.api_key_id and self.private_key_pem)


@dataclass(frozen=True)
class ScannerConfig:
    enabled: bool
    kill_switch: bool
    poll_interval_s: int
    market_status: str
    db_path: Path
    api: KalshiApiConfig
    rate_limit: RateLimitConfig
    trading: TradingConfig
    alerts: AlertConfig
    categories: list[CategoryRule]

    @property
    def validated_categories(self) -> list[str]:
        """Names of categories the research has validated — the ONLY ones that
        may ever be flagged. Empty is a legitimate, expected state."""
        return [c.name for c in self.categories if c.validated]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_private_key() -> str | None:
    """Read the RSA private key from env: inline PEM wins, else a file path."""
    pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
    if pem:
        return pem
    path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            logger.warning("KALSHI_PRIVATE_KEY_PATH set but file not found: %s", p)
            return None
        return p.read_text()
    return None


def load_scanner_config(
    path: Path | str | None = None,
    local_override: Path | str | None = _LOCAL_OVERRIDE_PATH,
    env: dict[str, str] | None = None,
) -> ScannerConfig:
    """Load and type the scanner config.

    ``env`` defaults to ``os.environ`` and supplies secrets + optional base-URL
    override. Categories with ``validated: false`` emit a loud warning so an
    UNVALIDATED placeholder can never be mistaken for a real edge.
    """
    env = os.environ if env is None else env

    base_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(base_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if local_override is not None:
        op = Path(local_override)
        if op.exists():
            with open(op) as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    api_cfg = cfg.get("api", {})
    api = KalshiApiConfig(
        base_url=env.get("KALSHI_API_BASE") or api_cfg.get("base_url", KalshiApiConfig.base_url),
        timeout_s=float(api_cfg.get("timeout_s", 10.0)),
        api_key_id=env.get("KALSHI_API_KEY_ID"),
        private_key_pem=_resolve_private_key() if env is os.environ else env.get("KALSHI_PRIVATE_KEY_PEM"),
    )

    rl_cfg = cfg.get("rate_limit", {})
    rate_limit = RateLimitConfig(
        max_requests_per_sec=float(rl_cfg.get("max_requests_per_sec", 5.0)),
        burst=int(rl_cfg.get("burst", 5)),
        max_retries=int(rl_cfg.get("max_retries", 5)),
        backoff_base_s=float(rl_cfg.get("backoff_base_s", 1.0)),
        backoff_max_s=float(rl_cfg.get("backoff_max_s", 60.0)),
    )

    tr_cfg = cfg.get("trading", {})
    trading = TradingConfig(
        bankroll=float(tr_cfg.get("bankroll", 10_000.0)),
        kelly_fraction=float(tr_cfg.get("kelly_fraction", 0.25)),
        max_position_frac=float(tr_cfg.get("max_position_frac", 0.02)),
        fee_coefficient=float(tr_cfg.get("fee_coefficient", 0.07)),
        ci_margin=float(tr_cfg.get("ci_margin", 0.0)),
        min_ev_per_contract=float(tr_cfg.get("min_ev_per_contract", 0.0)),
        slippage_buffer_cents=float(tr_cfg.get("slippage_buffer_cents", 1.0)),
        require_book_for_flag=bool(tr_cfg.get("require_book_for_flag", True)),
        max_event_frac=float(tr_cfg.get("max_event_frac", 0.05)),
        max_category_frac=float(tr_cfg.get("max_category_frac", 0.10)),
        max_total_frac=float(tr_cfg.get("max_total_frac", 0.25)),
    )

    al_cfg = cfg.get("alerts", {})
    alerts = AlertConfig(
        enabled=bool(al_cfg.get("enabled", False)),
        channel=str(al_cfg.get("channel", "ntfy")),
        ntfy_server=str(al_cfg.get("ntfy_server", "https://ntfy.sh")),
        ntfy_topic=str(al_cfg.get("ntfy_topic", "")),
        min_ev_notional=float(al_cfg.get("min_ev_notional", 1.0)),
    )

    categories = [
        CategoryRule(
            name=c["name"],
            validated=bool(c.get("validated", False)),
            title_patterns=list(c.get("title_patterns", [])),
            series_tickers=list(c.get("series_tickers", [])),
            kalshi_categories=list(c.get("kalshi_categories", [])),
        )
        for c in cfg.get("categories", [])
    ]

    db_path = Path(cfg.get("db_path", "data/kalshi/snapshots.sqlite"))
    if not db_path.is_absolute():
        db_path = repo_root() / db_path

    config = ScannerConfig(
        enabled=bool(cfg.get("enabled", True)),
        kill_switch=bool(cfg.get("kill_switch", False)),
        poll_interval_s=int(cfg.get("poll_interval_s", 300)),
        market_status=str(cfg.get("market_status", "open")),
        db_path=db_path,
        api=api,
        rate_limit=rate_limit,
        trading=trading,
        alerts=alerts,
        categories=categories,
    )

    unvalidated = [c.name for c in categories if not c.validated]
    if unvalidated:
        logger.warning(
            "UNVALIDATED categories present and will NOT be flagged for real money: %s. "
            "Only validated categories %s may be flagged.",
            unvalidated, config.validated_categories or "[]",
        )
    return config

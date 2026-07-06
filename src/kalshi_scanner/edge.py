"""Edge calculator (Phase 3).

Turns a scored :class:`Signal` into an :class:`EdgeResult`: which side (if any)
to buy, the raw edge against the *executable* price, the fee- and
slippage-adjusted expected value, a fractional-Kelly size, and — most
importantly — whether it is **flaggable**.

A market is flaggable only if ALL of these hold:

1. its category is *validated* (the hard rule; nothing is validated today),
2. the executable price is **outside** the model's confidence interval (a gap
   inside the uncertainty band is noise, not edge),
3. the fee/slippage-adjusted EV per contract clears ``min_ev_per_contract``,
4. we actually have order-book depth (unless ``require_book_for_flag`` is off),
5. the suggested size is at least one contract.

The statistical gate uses the executable side: to buy YES we pay the YES ask, so
we flag YES only when that ask is below the model's lower CI bound; symmetrically
for NO. This is deliberately conservative — we compare against the price we would
actually pay, not the mid.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TradingConfig
from .fees import fee_rate_per_contract
from .sizing import size_position
from .slippage import BookLevel, fallback_fill, walk_book


@dataclass
class EdgeResult:
    scan_id: int | None
    ticker: str
    event_ticker: str | None
    category: str
    validated: bool
    side: str | None            # "YES" | "NO" | None
    model_prob: float | None
    ci_lo: float | None
    ci_hi: float | None
    market_price: float | None  # executable ask for the chosen side (dollars)
    raw_edge: float | None      # p_side - market_price (before fees/slippage)
    fee_per_contract: float
    effective_cost: float | None  # price + fee + slippage (dollars)
    ev_per_contract: float | None  # p_side - effective_cost
    kelly_fraction: float       # risked fraction of bankroll after fraction+cap
    contracts: int
    notional: float
    slippage: float
    book_available: bool
    gate_pass: bool
    flaggable: bool
    reason: str


class EdgeEvaluator:
    def __init__(
        self,
        config: TradingConfig,
        *,
        books: dict[str, tuple[list[BookLevel], list[BookLevel]]] | None = None,
        kill_switch: bool = False,
    ) -> None:
        """``books`` maps ticker -> (yes_ask_levels, no_ask_levels) when real
        order-book depth is available (requires Kalshi auth). Absent it, a
        conservative top-of-book-plus-buffer fill is used.

        ``kill_switch`` (Phase 7) forces every market to be non-flaggable — an
        instant pause on flagging with no other change to the computation."""
        self.config = config
        self.books = books or {}
        self.kill_switch = kill_switch

    def evaluate_all(self, signals, scan_id: int | None = None, store=None) -> list[EdgeResult]:
        results = [self.evaluate(s) for s in signals]
        if store is not None and scan_id is not None:
            store.record_edges(results)
        return results

    def evaluate(self, signal) -> EdgeResult:
        c = self.config
        base = dict(
            scan_id=signal.scan_id, ticker=signal.ticker, event_ticker=signal.event_ticker,
            category=signal.category,
            validated=signal.validated, model_prob=signal.model_prob,
            ci_lo=signal.ci_lo, ci_hi=signal.ci_hi, fee_per_contract=0.0,
            kelly_fraction=0.0, contracts=0, notional=0.0, slippage=0.0,
            book_available=False, gate_pass=False,
        )

        if not signal.is_scored:
            return _empty(base, reason=signal.reason or "no_signal")

        p, lo, hi = signal.model_prob, signal.ci_lo, signal.ci_hi
        yes_ask, no_ask = signal.market_yes_ask, signal.market_no_ask
        yes_price = yes_ask / 100.0 if yes_ask else None
        no_price = no_ask / 100.0 if no_ask else None
        if yes_price is None and no_price is None:
            return _empty(base, reason="no_prices")

        # Statistical gate on the executable price vs the model CI.
        margin = c.ci_margin
        yes_gate = yes_price is not None and yes_price < (lo - margin)
        no_gate = no_price is not None and no_price < (1.0 - hi - margin)
        yes_raw = (p - yes_price) if yes_price is not None else None
        no_raw = ((1.0 - p) - no_price) if no_price is not None else None

        if yes_gate and no_gate:
            side = "YES" if (yes_raw or 0) >= (no_raw or 0) else "NO"
        elif yes_gate:
            side = "YES"
        elif no_gate:
            side = "NO"
        else:
            return _empty(base, reason="inside_ci_band")

        # Chosen-side economics.
        if side == "YES":
            price, ask_cents, p_side, raw_edge = yes_price, yes_ask, p, yes_raw
        else:
            price, ask_cents, p_side, raw_edge = no_price, no_ask, 1.0 - p, no_raw

        fee_rate = fee_rate_per_contract(price, c.fee_coefficient)
        effective_cost, slippage, book_available, depth = self._fill_cost(
            signal.ticker, side, ask_cents, price, fee_rate, p_side
        )
        ev = p_side - effective_cost
        size = size_position(
            p_side, effective_cost, c.bankroll,
            kelly_fraction=c.kelly_fraction, max_position_frac=c.max_position_frac,
        )
        contracts = size.contracts
        if book_available and depth is not None:
            contracts = min(contracts, depth)
        notional = contracts * effective_cost

        flaggable, reason = self._decide(signal.validated, ev, book_available, contracts)

        return EdgeResult(
            **{k: base[k] for k in ("scan_id", "ticker", "event_ticker", "category",
                                    "validated", "model_prob", "ci_lo", "ci_hi")},
            side=side, market_price=price, raw_edge=raw_edge,
            fee_per_contract=fee_rate, effective_cost=effective_cost, ev_per_contract=ev,
            kelly_fraction=size.position_fraction, contracts=contracts, notional=notional,
            slippage=slippage, book_available=book_available, gate_pass=True,
            flaggable=flaggable, reason=reason,
        )

    def _fill_cost(self, ticker, side, ask_cents, price, fee_rate, p_side):
        """Return (effective_cost, slippage, book_available, depth)."""
        levels = None
        if ticker in self.books:
            yes_levels, no_levels = self.books[ticker]
            levels = yes_levels if side == "YES" else no_levels
        if levels:
            prelim = size_position(
                p_side, price + fee_rate, self.config.bankroll,
                kelly_fraction=self.config.kelly_fraction,
                max_position_frac=self.config.max_position_frac,
            )
            fill = walk_book(levels, max(prelim.contracts, 1))
            return fill.avg_price + fee_rate, fill.slippage, True, fill.filled

        fb = fallback_fill(ask_cents, size=1, buffer_cents=self.config.slippage_buffer_cents)
        return price + fee_rate + fb.slippage, fb.slippage, False, None

    def _decide(self, validated, ev, book_available, contracts):
        c = self.config
        if self.kill_switch:
            return False, "kill_switch"
        if ev <= c.min_ev_per_contract:
            return False, "ev_below_min"
        if c.require_book_for_flag and not book_available:
            return False, "no_orderbook_depth"
        if not validated:
            return False, "category_not_validated"
        if contracts < 1:
            return False, "size_zero"
        return True, "flaggable"


def _empty(base: dict, reason: str) -> EdgeResult:
    return EdgeResult(
        **base, side=None, market_price=None, raw_edge=None, effective_cost=None,
        ev_per_contract=None, flaggable=False, reason=reason,
    )

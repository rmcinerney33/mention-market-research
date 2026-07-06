"""Paper-trading engine — the required gate before any real money.

Every flag opens a simulated position at the price actually available at flag
time (with the slippage model's effective cost). When the market resolves, the
position is settled against the real outcome and we track realized performance.

The report answers the only question that matters: **is live performance
consistent with the backtest?** It computes simulated P&L, hit rate, the realized
Brier score of flagged predictions, realized edge vs. the model's expected edge
(with a proper t-test, not eyeballing), max drawdown, and whether the market
drifted toward or away from the model between flag and resolution. The go/no-go
verdict requires a meaningful sample — **≥ 8 weeks and ≥ 50 settled flags** — and
realized edge that is positive and not significantly below expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class PaperPosition:
    ticker: str
    event_ticker: str | None
    side: str                    # "YES" | "NO"
    contracts: int
    entry_cost: float            # all-in cost paid per contract (price + fee + slippage)
    market_entry_price: float    # executable price at flag (dollars)
    model_prob: float            # P(YES)
    expected_ev: float           # model-expected EV per contract (= p_side - entry_cost)
    flag_time: datetime
    scan_id: int | None = None
    outcome: int | None = None            # 1 if OUR side won, else 0; None while open
    settled_at: datetime | None = None
    exit_market_price: float | None = None  # market implied prob just before resolution

    @property
    def p_side(self) -> float:
        return self.model_prob if self.side == "YES" else 1.0 - self.model_prob

    @property
    def won(self) -> bool:
        return self.outcome == 1

    @property
    def pnl(self) -> float | None:
        if self.outcome is None:
            return None
        payout = 1.0 if self.won else 0.0
        return self.contracts * (payout - self.entry_cost)

    @property
    def realized_edge(self) -> float | None:
        if self.outcome is None:
            return None
        return (1.0 if self.won else 0.0) - self.entry_cost


def open_from_flag(flag) -> PaperPosition:
    """Open a paper position from a :class:`Flag` at its flag-time price."""
    p_side = flag.model_prob if flag.side == "YES" else 1.0 - flag.model_prob
    entry_cost = p_side - flag.ev_per_contract  # ev = p_side - effective_cost
    return PaperPosition(
        ticker=flag.ticker, event_ticker=flag.event_ticker, side=flag.side,
        contracts=flag.contracts, entry_cost=entry_cost,
        market_entry_price=flag.market_price, model_prob=flag.model_prob,
        expected_ev=flag.ev_per_contract, flag_time=flag.flagged_at, scan_id=flag.scan_id,
    )


def settle(position: PaperPosition, event_label: int, *, settled_at: datetime | None = None,
           exit_market_price: float | None = None) -> PaperPosition:
    """Settle against the event's real label (1 if the phrase was used)."""
    side_won = (event_label == 1) if position.side == "YES" else (event_label == 0)
    position.outcome = 1 if side_won else 0
    position.settled_at = settled_at
    position.exit_market_price = exit_market_price
    return position


@dataclass
class PaperReport:
    n_positions: int
    n_settled: int
    pnl: float
    hit_rate: float
    realized_brier: float
    realized_edge_mean: float
    expected_edge_mean: float
    edge_diff_tstat: float
    edge_diff_pvalue: float
    max_drawdown: float
    decay_toward_model: float     # fraction where the market drifted toward the model
    span_weeks: float
    go: bool
    notes: list[str]


def build_report(positions: list[PaperPosition], *, min_flags: int = 50,
                 min_weeks: float = 8.0) -> PaperReport:
    settled = [p for p in positions if p.outcome is not None]
    n_settled = len(settled)
    if n_settled == 0:
        return PaperReport(len(positions), 0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 1.0,
                           0.0, 0.0, 0.0, go=False, notes=["no settled positions yet"])

    won = np.array([1.0 if p.won else 0.0 for p in settled])
    p_side = np.array([p.p_side for p in settled])
    realized = np.array([p.realized_edge for p in settled], dtype=float)
    expected = np.array([p.expected_ev for p in settled], dtype=float)

    pnl = float(sum(p.pnl for p in settled))
    hit_rate = float(won.mean())
    realized_brier = float(np.mean((p_side - won) ** 2))

    # Realized vs expected edge — one-sample t-test on the per-flag differential.
    diff = realized - expected
    if n_settled >= 2 and np.std(diff) > 0:
        from scipy import stats
        tstat, pvalue = stats.ttest_1samp(diff, 0.0)
        tstat, pvalue = float(tstat), float(pvalue)
    else:
        tstat, pvalue = 0.0, 1.0

    # Max drawdown of cumulative P&L in settlement order.
    ordered = sorted(settled, key=lambda p: (p.settled_at or p.flag_time))
    cum = np.cumsum([p.pnl for p in ordered])
    running_peak = np.maximum.accumulate(cum)
    max_drawdown = float(np.max(running_peak - cum)) if len(cum) else 0.0

    # Did the market move toward the model? Use the pre-resolution price if we
    # have it, else the settlement value. Positions are entered when the model
    # sees the side as underpriced, so "toward model" = price/outcome rose.
    moves = []
    for p in settled:
        exit_val = p.exit_market_price if p.exit_market_price is not None else (1.0 if p.won else 0.0)
        moves.append(1.0 if exit_val > p.market_entry_price else 0.0)
    decay_toward_model = float(np.mean(moves))

    times = [p.flag_time for p in settled]
    span_weeks = (max(times) - min(times)).total_seconds() / (86400 * 7) if len(times) > 1 else 0.0

    realized_mean = float(realized.mean())
    expected_mean = float(expected.mean())
    significantly_below = realized_mean < expected_mean and pvalue < 0.05
    notes: list[str] = []
    if n_settled < min_flags:
        notes.append(f"insufficient sample: {n_settled}/{min_flags} settled flags")
    if span_weeks < min_weeks:
        notes.append(f"insufficient duration: {span_weeks:.1f}/{min_weeks} weeks")
    if realized_mean <= 0:
        notes.append("realized edge is not positive")
    if significantly_below:
        notes.append("realized edge significantly below model expectation (p<0.05)")
    go = not notes
    if go:
        notes.append("consistent with backtest over a meaningful sample")

    return PaperReport(
        n_positions=len(positions), n_settled=n_settled, pnl=pnl, hit_rate=hit_rate,
        realized_brier=realized_brier, realized_edge_mean=realized_mean,
        expected_edge_mean=expected_mean, edge_diff_tstat=tstat, edge_diff_pvalue=pvalue,
        max_drawdown=max_drawdown, decay_toward_model=decay_toward_model,
        span_weeks=span_weeks, go=go, notes=notes,
    )

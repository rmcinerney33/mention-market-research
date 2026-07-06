"""Model-health monitoring: calibration drift.

Markets adapt; an edge that held in backtest can decay. We watch the rolling
calibration of *resolved* flags (Brier and log-loss on the model's side
probability vs the realized outcome) and compare it to the deployed model's
validation-period baseline. Material degradation raises a flag to pause and
re-evaluate — the whole point being to notice decay before it costs money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MonitoringConfig

_EPS = 1e-6


@dataclass
class DriftStatus:
    n_settled: int
    window: int
    rolling_brier: float
    baseline_brier: float
    rolling_logloss: float
    baseline_logloss: float
    degraded: bool
    note: str


def assess_drift(positions, config: MonitoringConfig) -> DriftStatus:
    settled = sorted(
        (p for p in positions if p.outcome is not None),
        key=lambda p: (p.settled_at or p.flag_time),
    )
    recent = settled[-config.drift_window:]
    n = len(recent)
    if n < config.min_drift_sample:
        return DriftStatus(
            n_settled=len(settled), window=n, rolling_brier=float("nan"),
            baseline_brier=config.baseline_brier, rolling_logloss=float("nan"),
            baseline_logloss=config.baseline_logloss, degraded=False,
            note=f"insufficient sample: {n}/{config.min_drift_sample} resolved flags",
        )

    p_side = np.array([p.p_side for p in recent])
    won = np.array([1.0 if p.won else 0.0 for p in recent])
    pc = np.clip(p_side, _EPS, 1 - _EPS)
    brier = float(np.mean((p_side - won) ** 2))
    logloss = float(np.mean(-(won * np.log(pc) + (1 - won) * np.log(1 - pc))))

    tol = config.drift_rel_tolerance
    brier_bad = brier > config.baseline_brier * (1 + tol)
    ll_bad = logloss > config.baseline_logloss * (1 + tol)
    degraded = brier_bad or ll_bad
    if degraded:
        which = ", ".join(
            w for w, bad in (("Brier", brier_bad), ("log-loss", ll_bad)) if bad
        )
        note = (f"calibration degraded on {which}: "
                f"Brier {brier:.3f} vs {config.baseline_brier:.3f}, "
                f"log-loss {logloss:.3f} vs {config.baseline_logloss:.3f} "
                f"(> {tol:.0%} worse)")
    else:
        note = (f"calibration within tolerance: Brier {brier:.3f} vs "
                f"{config.baseline_brier:.3f}, log-loss {logloss:.3f} vs "
                f"{config.baseline_logloss:.3f}")

    return DriftStatus(
        n_settled=len(settled), window=n, rolling_brier=brier,
        baseline_brier=config.baseline_brier, rolling_logloss=logloss,
        baseline_logloss=config.baseline_logloss, degraded=degraded, note=note,
    )

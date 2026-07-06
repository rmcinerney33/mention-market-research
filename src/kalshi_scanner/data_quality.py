"""Data-quality checks.

Bad inputs silently produce garbage features and garbage flags. These checks
surface the common failure modes — a stale scan, coverage gaps, a Kalshi schema
change (markets missing expected fields), or stale transcripts/GDELT feeding the
model — so the system complains loudly instead of pretending.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import MonitoringConfig


@dataclass(frozen=True)
class DataQualityIssue:
    check: str
    severity: str  # "warn" | "error"
    detail: str


def check_scan_freshness(latest_ok_started_at, now: datetime, max_age_s: float):
    if latest_ok_started_at is None:
        return DataQualityIssue("scan_freshness", "error", "no successful scan on record")
    age = (now - latest_ok_started_at).total_seconds()
    if age > max_age_s:
        return DataQualityIssue(
            "scan_freshness", "warn",
            f"latest ok scan is {age / 60:.1f} min old (> {max_age_s / 60:.0f} min)",
        )
    return None


def check_schema(snapshots_raw: list[dict], required_fields):
    if not snapshots_raw:
        return None
    missing = {
        f: sum(1 for r in snapshots_raw if r.get(f) is None)
        for f in required_fields
    }
    bad = {f: c for f, c in missing.items() if c > 0}
    if bad:
        return DataQualityIssue(
            "schema", "error",
            f"{len(snapshots_raw)} markets; fields missing on some: {bad} "
            "(possible Kalshi schema change)",
        )
    return None


def check_coverage_gaps(gaps):
    if gaps:
        return DataQualityIssue("coverage", "warn", f"{len(gaps)} coverage gap(s) detected")
    return None


def check_corpus_staleness(latest_time, now: datetime, max_age_days: float, label: str):
    if latest_time is None:
        return DataQualityIssue(f"{label}_staleness", "warn", f"no {label} data available")
    age_days = (now - latest_time).total_seconds() / 86400.0
    if age_days > max_age_days:
        return DataQualityIssue(
            f"{label}_staleness", "warn",
            f"latest {label} is {age_days:.1f}d old (> {max_age_days:.0f}d)",
        )
    return None


def run_data_quality(
    store, config: MonitoringConfig, *, now: datetime, expected_interval_s: float, world=None
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []

    latest = store.latest_ok_run()
    started = datetime.fromisoformat(latest["started_at"]) if latest else None
    if (issue := check_scan_freshness(started, now, config.max_scan_age_s)):
        issues.append(issue)

    if (issue := check_coverage_gaps(store.find_gaps(expected_interval_s))):
        issues.append(issue)

    scan_id = store.latest_ok_scan_id()
    if scan_id is not None:
        raws = [s.raw for s in store.load_snapshots(scan_id)]
        if (issue := check_schema(raws, config.required_market_fields)):
            issues.append(issue)

    if world is not None:
        import pandas as pd

        ev_latest = pd.to_datetime(world.events["event_time"], utc=True).max().to_pydatetime()
        gd_latest = pd.to_datetime(world.gdelt["observed_time"], utc=True).max().to_pydatetime()
        for latest_time, label in ((ev_latest, "transcripts"), (gd_latest, "gdelt")):
            if (issue := check_corpus_staleness(latest_time, now, config.max_corpus_age_days, label)):
                issues.append(issue)

    return issues

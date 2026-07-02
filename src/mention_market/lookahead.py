"""No-lookahead guarantee.

The single most important invariant in this project: every feature used to
predict the outcome of an event at time ``t`` must be computable from data
whose timestamp is *strictly* less than ``t``. If that invariant breaks, every
downstream metric is contaminated and the market-efficiency conclusion is
worthless.

We enforce this **structurally**, not by convention:

- :func:`assert_strictly_before` is a vectorized guard that raises
  :class:`LookaheadError` if any source timestamp is at or after the cutoff.
- :class:`PointInTimeView` wraps a timestamped table so the *only* way to read
  it is ``as_of(cutoff)``, which returns strictly-earlier rows and records an
  audit trail of the latest timestamp it ever exposed. Feature builders are
  written against this view, so a leak requires actively circumventing the API
  rather than merely forgetting a filter.

The distinction between "strictly before" and "at or before" matters: if a
market snapshot or transcript shares the exact event timestamp, including it is
a leak (the outcome is contemporaneous). We default to strict ``<``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


class LookaheadError(AssertionError):
    """Raised when data at or after the prediction boundary is used as input."""


def _to_series(times: object) -> pd.Series:
    """Coerce an assortment of timestamp containers to a tz-aware UTC Series."""
    s = pd.Series(pd.to_datetime(list(times) if not isinstance(times, pd.Series) else times, utc=True))
    return s


def assert_strictly_before(
    source_times: object,
    cutoff: datetime | pd.Timestamp,
    *,
    context: str = "",
    inclusive: bool = False,
) -> None:
    """Raise :class:`LookaheadError` if any source timestamp violates the cutoff.

    Parameters
    ----------
    source_times:
        Any iterable / Series of timestamps that fed a feature.
    cutoff:
        The prediction boundary ``t`` (the event time).
    context:
        Human-readable label included in the error (feature name, source, etc.).
    inclusive:
        If ``False`` (default), timestamps must be strictly ``< cutoff``.
        If ``True``, ``<= cutoff`` is permitted. Strict is correct for outcome
        prediction; inclusive exists only for the rare legitimately-simultaneous
        reference datum, and callers must justify it.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        raise ValueError(f"cutoff must be tz-aware; got naive {cutoff!r}")
    cutoff_ts = cutoff_ts.tz_convert("UTC")

    times = _to_series(source_times)
    if times.empty:
        return

    violating = times[times > cutoff_ts] if inclusive else times[times >= cutoff_ts]

    if len(violating) > 0:
        worst = violating.max()
        op = ">" if inclusive else ">="
        raise LookaheadError(
            f"Lookahead leak{f' in {context}' if context else ''}: "
            f"{len(violating)} source timestamp(s) {op} cutoff {cutoff_ts.isoformat()} "
            f"(worst offender: {worst.isoformat()}). "
            "A feature for this event consumed data from at or after the event."
        )


@dataclass
class PointInTimeView:
    """A timestamped table that can only be queried as-of a cutoff time.

    Construct once from a DataFrame with a designated time column, then read it
    exclusively through :meth:`as_of`. Each read is checked and audited, so
    feature code physically cannot see future rows.
    """

    frame: pd.DataFrame
    time_col: str
    name: str = "view"
    # Audit trail: the latest source timestamp ever returned, per cutoff.
    _max_seen: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.time_col not in self.frame.columns:
            raise KeyError(f"time_col {self.time_col!r} not in frame columns {list(self.frame.columns)}")
        times = pd.to_datetime(self.frame[self.time_col], utc=True)
        if times.isna().any():
            raise ValueError(f"{self.name}: {self.time_col} contains unparseable/NaT timestamps")
        # Store a normalized, sorted copy so as_of is cheap and deterministic.
        self.frame = self.frame.assign(**{self.time_col: times}).sort_values(self.time_col).reset_index(drop=True)

    def as_of(self, cutoff: datetime | pd.Timestamp, *, inclusive: bool = False) -> pd.DataFrame:
        """Return rows strictly before ``cutoff`` (or ``<=`` if ``inclusive``).

        The returned frame is a copy; mutating it cannot corrupt the view.
        """
        cutoff_ts = pd.Timestamp(cutoff)
        if cutoff_ts.tzinfo is None:
            raise ValueError(f"cutoff must be tz-aware; got naive {cutoff!r}")
        cutoff_ts = cutoff_ts.tz_convert("UTC")

        times = self.frame[self.time_col]
        mask = times <= cutoff_ts if inclusive else times < cutoff_ts
        out = self.frame.loc[mask].copy()

        if not out.empty:
            latest = out[self.time_col].max()
            # Defense in depth: the mask should already guarantee this.
            assert_strictly_before(
                out[self.time_col], cutoff_ts, context=f"{self.name}.as_of", inclusive=inclusive
            )
            if self._max_seen is None or latest > self._max_seen:
                self._max_seen = latest
        return out

    @property
    def max_timestamp_seen(self) -> pd.Timestamp | None:
        """Latest source timestamp this view has ever exposed (audit)."""
        return self._max_seen


def audit_feature_row(
    event_time: datetime | pd.Timestamp,
    source_timestamps: dict[str, object],
    *,
    inclusive: bool = False,
) -> None:
    """Assert a whole feature row respects the boundary.

    ``source_timestamps`` maps a source name -> the timestamps that source
    contributed to this row. Every one is checked against ``event_time``. Use
    this as a final gate when assembling a feature table, so a single call
    validates the entire row's provenance.
    """
    for source_name, times in source_timestamps.items():
        assert_strictly_before(times, event_time, context=source_name, inclusive=inclusive)

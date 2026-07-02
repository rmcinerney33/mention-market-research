"""Tests for the no-lookahead guarantee.

These are the most important tests in the repo: if a leak slips through here,
every downstream metric and the headline market-efficiency conclusion are
invalid. We test both the primitive guard and the structural PointInTimeView,
including the strict-vs-inclusive boundary and the exact-timestamp edge case.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from mention_market.lookahead import (
    LookaheadError,
    PointInTimeView,
    assert_strictly_before,
    audit_feature_row,
)

pytestmark = pytest.mark.lookahead

T0 = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)


def _ts(days: float = 0, hours: float = 0) -> datetime:
    return T0 + timedelta(days=days, hours=hours)


# ---------------------------------------------------------------------------
# assert_strictly_before
# ---------------------------------------------------------------------------
def test_passes_when_all_before():
    src = [_ts(-3), _ts(-1), _ts(hours=-1)]
    assert_strictly_before(src, T0, context="unit")  # no raise


def test_raises_when_a_source_is_after():
    src = [_ts(-3), _ts(hours=1)]
    with pytest.raises(LookaheadError) as ei:
        assert_strictly_before(src, T0, context="feat_x")
    assert "feat_x" in str(ei.value)


def test_exact_boundary_is_a_leak_under_strict():
    # A datum AT the event time is contemporaneous with the outcome => leak.
    src = [_ts(-1), T0]
    with pytest.raises(LookaheadError):
        assert_strictly_before(src, T0)


def test_exact_boundary_allowed_when_inclusive():
    src = [_ts(-1), T0]
    assert_strictly_before(src, T0, inclusive=True)  # no raise


def test_after_boundary_still_raises_when_inclusive():
    src = [_ts(hours=1)]
    with pytest.raises(LookaheadError):
        assert_strictly_before(src, T0, inclusive=True)


def test_empty_source_is_ok():
    assert_strictly_before([], T0)


def test_naive_cutoff_rejected():
    with pytest.raises(ValueError):
        assert_strictly_before([_ts(-1)], datetime(2024, 1, 10, 12, 0))  # naive


def test_accepts_pandas_series_and_mixed_tz():
    # A source series in a different tz must still be compared correctly.
    src = pd.Series(pd.to_datetime(["2024-01-10T06:00:00-05:00"]))  # 11:00 UTC < T0
    assert_strictly_before(src, T0)
    src_after = pd.Series(pd.to_datetime(["2024-01-10T08:00:00-05:00"]))  # 13:00 UTC > T0
    with pytest.raises(LookaheadError):
        assert_strictly_before(src_after, T0)


# ---------------------------------------------------------------------------
# PointInTimeView
# ---------------------------------------------------------------------------
@pytest.fixture
def view() -> PointInTimeView:
    df = pd.DataFrame(
        {
            "observed_time": [_ts(-5), _ts(-2), _ts(hours=-1), T0, _ts(hours=1)],
            "value": [10, 20, 30, 40, 50],
        }
    )
    return PointInTimeView(df, time_col="observed_time", name="prices")


def test_as_of_returns_only_strictly_earlier(view):
    got = view.as_of(T0)
    assert list(got["value"]) == [10, 20, 30]  # excludes the T0 row and the future row


def test_as_of_inclusive_includes_boundary(view):
    got = view.as_of(T0, inclusive=True)
    assert list(got["value"]) == [10, 20, 30, 40]


def test_as_of_never_returns_future_rows(view):
    # Even querying past the last timestamp, nothing after cutoff leaks.
    got = view.as_of(_ts(hours=0.5))
    assert (got["observed_time"] < pd.Timestamp(_ts(hours=0.5))).all()


def test_view_records_audit_high_water_mark(view):
    view.as_of(_ts(-3))
    assert view.max_timestamp_seen == pd.Timestamp(_ts(-5))
    view.as_of(T0)
    # High-water mark advances to the latest row ever exposed (the -1h row).
    assert view.max_timestamp_seen == pd.Timestamp(_ts(hours=-1))


def test_view_returned_frame_is_a_copy(view):
    got = view.as_of(T0)
    got.loc[got.index[0], "value"] = -999
    # Mutating the result must not corrupt the underlying view.
    assert view.frame.loc[0, "value"] == 10


def test_view_rejects_missing_time_col():
    with pytest.raises(KeyError):
        PointInTimeView(pd.DataFrame({"x": [1]}), time_col="observed_time")


def test_view_rejects_unparseable_times():
    df = pd.DataFrame({"observed_time": ["not-a-date"], "value": [1]})
    with pytest.raises(ValueError):
        PointInTimeView(df, time_col="observed_time")


# ---------------------------------------------------------------------------
# audit_feature_row
# ---------------------------------------------------------------------------
def test_audit_feature_row_passes_clean():
    audit_feature_row(
        T0,
        {"usage": [_ts(-2), _ts(-1)], "market": [_ts(hours=-6)], "gdelt": [_ts(-3)]},
    )


def test_audit_feature_row_flags_the_leaky_source():
    with pytest.raises(LookaheadError) as ei:
        audit_feature_row(
            T0,
            {"usage": [_ts(-2)], "market": [_ts(hours=2)]},  # market leaks
        )
    assert "market" in str(ei.value)

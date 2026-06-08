"""Tests for extended cost and failure reporting."""

from __future__ import annotations

from metrics.cost_analysis import cost_summary, interquartile_range
from metrics.reliability import failure_rate, run_failure_flags


def test_cost_summary_includes_median_and_iqr() -> None:
    summary = cost_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["median"] == 25.0
    assert summary["iqr"] == 15.0


def test_interquartile_range_single_value() -> None:
    assert interquartile_range([42.0]) == 0.0


def test_failure_rate_tail_and_backlog_events() -> None:
    costs = [100.0, 110.0, 120.0, 130.0, 500.0]
    backlogs = {
        "Retailer": [
            [0, 0, 0],
            [0, 0, 0],
            [0, 20, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    }
    flags = run_failure_flags(costs, backlogs_by_echelon=backlogs)
    assert flags[2] is True  # backlog explosion
    assert flags[4] is True  # tail cost
    assert failure_rate(costs, backlogs_by_echelon=backlogs) > 0.0

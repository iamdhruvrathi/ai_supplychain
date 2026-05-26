"""Reliability and tail-risk metrics for repeated-run experiments."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

import numpy as np


def coefficient_of_variation(values: List[float]) -> Optional[float]:
    """CV = std / mean (paper Table 1)."""
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return float(statistics.pstdev(values) / mean)


def run_to_run_instability(values: List[float]) -> float:
    """Population variance across repeated runs of total cost."""
    if len(values) < 2:
        return 0.0
    return float(statistics.pvariance(values))


def detect_order_spikes(
    orders: List[int],
    threshold_multiplier: float = 2.5,
) -> List[int]:
    """Week indices where order exceeds threshold_multiplier * median."""
    if not orders:
        return []
    median = float(np.median(orders))
    if median <= 0:
        median = 1.0
    threshold = threshold_multiplier * median
    return [t for t, q in enumerate(orders) if q > threshold]


def detect_inventory_collapse(
    inventory: List[int],
    collapse_threshold: int = 1,
    min_consecutive: int = 2,
) -> List[int]:
    """Start weeks of consecutive near-zero inventory."""
    events = []
    streak = 0
    for t, inv in enumerate(inventory):
        if inv <= collapse_threshold:
            streak += 1
            if streak == min_consecutive:
                events.append(t - min_consecutive + 1)
        else:
            streak = 0
    return events


def detect_backlog_explosion(
    backlog: List[int],
    threshold: int = 15,
) -> List[int]:
    return [t for t, b in enumerate(backlog) if b >= threshold]


def tail_event_rate(
    values: List[float],
    percentile: float = 90.0,
) -> float:
    """Fraction of runs above the given cost percentile."""
    if not values:
        return 0.0
    cutoff = float(np.percentile(values, percentile))
    return float(sum(1 for v in values if v >= cutoff) / len(values))


def reliability_summary(
    total_costs: List[float],
    orders_by_echelon: Optional[Dict[str, List[List[int]]]] = None,
    inventories_by_echelon: Optional[Dict[str, List[List[int]]]] = None,
    backlogs_by_echelon: Optional[Dict[str, List[List[int]]]] = None,
) -> Dict[str, object]:
    """Aggregate reliability report for R repeated runs."""
    summary: Dict[str, object] = {
        "n_runs": len(total_costs),
        "mean_cost": float(statistics.mean(total_costs)) if total_costs else 0.0,
        "std_cost": float(statistics.pstdev(total_costs)) if len(total_costs) > 1 else 0.0,
        "coefficient_of_variation": coefficient_of_variation(total_costs),
        "run_to_run_instability": run_to_run_instability(total_costs),
        "tail_event_rate_p90": tail_event_rate(total_costs, 90.0),
        "max_cost": max(total_costs) if total_costs else 0.0,
        "min_cost": min(total_costs) if total_costs else 0.0,
    }

    if orders_by_echelon:
        spikes = {}
        for echelon, runs_orders in orders_by_echelon.items():
            all_spikes = []
            for orders in runs_orders:
                all_spikes.extend(detect_order_spikes(orders))
            spikes[echelon] = len(all_spikes)
        summary["order_spike_count"] = spikes

    if inventories_by_echelon:
        collapses = {}
        for echelon, runs_inv in inventories_by_echelon.items():
            count = sum(len(detect_inventory_collapse(inv)) for inv in runs_inv)
            collapses[echelon] = count
        summary["inventory_collapse_events"] = collapses

    if backlogs_by_echelon:
        explosions = {}
        for echelon, runs_bl in backlogs_by_echelon.items():
            count = sum(len(detect_backlog_explosion(bl)) for bl in runs_bl)
            explosions[echelon] = count
        summary["backlog_explosion_events"] = explosions

    return summary

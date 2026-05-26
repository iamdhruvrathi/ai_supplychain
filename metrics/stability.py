"""Supply chain stability metrics for evaluation and RL objectives."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Union


def _series_variance(series: List[Union[int, float]]) -> float:
    """Population variance; returns 0.0 for empty or single-point series."""
    if len(series) < 2:
        return 0.0
    return float(statistics.pvariance(series))


def order_variance(history: Dict) -> Dict[str, float]:
    """Per-agent variance of order quantities."""
    orders = history.get("orders", {})
    return {
        agent: _series_variance(series)
        for agent, series in orders.items()
    }


def inventory_variance(history: Dict) -> Dict[str, float]:
    """Per-agent variance of inventory levels."""
    inventory = history.get("inventory", {})
    return {
        agent: _series_variance(series)
        for agent, series in inventory.items()
    }


def backlog_variance(history: Dict) -> Dict[str, float]:
    """Per-agent variance of backlog levels."""
    backlog = history.get("backlog", {})
    return {
        agent: _series_variance(series)
        for agent, series in backlog.items()
    }


def cumulative_instability(
    history: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate instability score across all echelons.

    Default weights sum order, inventory, and backlog variances equally
    per agent, then average across agents.
    """
    weights = weights or {
        "order": 1.0,
        "inventory": 1.0,
        "backlog": 1.0,
    }

    order_vars = order_variance(history)
    inv_vars = inventory_variance(history)
    backlog_vars = backlog_variance(history)

    agents = set(order_vars) | set(inv_vars) | set(backlog_vars)
    if not agents:
        return 0.0

    per_agent = []
    for agent in agents:
        score = (
            weights.get("order", 1.0) * order_vars.get(agent, 0.0)
            + weights.get("inventory", 1.0) * inv_vars.get(agent, 0.0)
            + weights.get("backlog", 1.0) * backlog_vars.get(agent, 0.0)
        )
        per_agent.append(score)

    return float(statistics.mean(per_agent))


def stability_summary(history: Dict) -> Dict[str, object]:
    """Full stability report from a BeerGame history dict."""
    return {
        "order_variance": order_variance(history),
        "inventory_variance": inventory_variance(history),
        "backlog_variance": backlog_variance(history),
        "cumulative_instability": cumulative_instability(history),
    }

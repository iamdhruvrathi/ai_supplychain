"""Bullwhip and variance-based metrics for the Beer Game.

Classical bullwhip (paper Section 4.1):
  B^{(r)}_k = Var_t(q^{(r)}_{k,t}) / Var_t(q^{(r)}_{k-1,t})
"""
from typing import Dict, List, Optional
import statistics


def bullwhip_ratio(orders: List[int], demand: List[int]) -> Optional[float]:
    """Compute the classical bullwhip ratio for one agent.

    B_k = Var(orders) / Var(demand)
    Returns None if demand variance is zero or insufficient data.
    """
    if len(orders) < 2 or len(demand) < 2:
        return None

    dv = statistics.pvariance(demand)
    if dv == 0:
        return None

    return statistics.pvariance(orders) / dv


def classical_bullwhip_adjacent(
    orders_upstream: List[int],
    orders_downstream: List[int],
) -> Optional[float]:
    """B_k = Var_t(q_{k,t}) / Var_t(q_{k-1,t}) for a single run."""
    if len(orders_upstream) < 2 or len(orders_downstream) < 2:
        return None
    den = statistics.pvariance(orders_downstream)
    if den == 0:
        return None
    return statistics.pvariance(orders_upstream) / den


def rolling_bullwhip(
    orders: List[int],
    demand: List[int],
    window: int = 5,
) -> List[Optional[float]]:
    """Rolling-window classical bullwhip ratio."""
    result: List[Optional[float]] = []
    for t in range(len(orders)):
        start = max(0, t - window + 1)
        o_win = orders[start : t + 1]
        d_win = demand[start : t + 1]
        result.append(bullwhip_ratio(o_win, d_win))
    return result


def bullwhip_per_agent(
    history: Dict[str, List[int]],
    demand_key: str = "demand",
) -> Dict[str, Optional[float]]:
    """Compute bullwhip ratios for each agent from a history dict."""
    demand = history.get(demand_key, [])
    result = {}
    for agent, orders in history.get("orders", {}).items():
        result[agent] = bullwhip_ratio(orders, demand)

    valid = [v for v in result.values() if v is not None]
    result["overall"] = statistics.mean(valid) if valid else None
    return result

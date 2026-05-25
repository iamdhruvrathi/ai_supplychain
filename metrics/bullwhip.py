"""Bullwhip and variance-based metrics for the Beer Game.

Functions here operate on ordered time-series (lists) of demand and orders.
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


def bullwhip_per_agent(history: Dict[str, List[int]], demand_key: str = "demand") -> Dict[str, Optional[float]]:
    """Compute bullwhip ratios for each agent from a history dict.

    The history dict is expected to contain:
      - history['demand'] -> List[int]
      - history['orders'][agent_name] -> List[int]

    Returns a dict mapping agent name -> bullwhip ratio (or None).
    """
    demand = history.get(demand_key, [])
    result = {}
    for agent, orders in history.get("orders", {}).items():
        result[agent] = bullwhip_ratio(orders, demand)

    # overall (mean of available ratios)
    valid = [v for v in result.values() if v is not None]
    result["overall"] = statistics.mean(valid) if valid else None
    return result

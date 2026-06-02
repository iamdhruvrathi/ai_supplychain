"""Simple inventory decision-support tool.

The first implementation intentionally uses a transparent base-stock rule
rather than a full EOQ calculation. It is deterministic, cheap to compute, and
easy to audit in repeated-run experiments.
"""

from __future__ import annotations

from typing import Any, Dict


def eoq_recommendation(state: Dict[str, Any]) -> int:
    """Return a base-stock order recommendation for the provided agent state.

    Formula:
        forecast = last_customer_demand
        target = forecast * lead_time
        recommended_order = max(0, target - inventory + backlog)
    """
    forecast = int(state.get("last_customer_demand", 0) or 0)
    lead_time = int(state.get("lead_time", 2) or 2)
    inventory = int(state.get("inventory", 0) or 0)
    backlog = int(state.get("backlog", 0) or 0)

    target = forecast * lead_time
    return max(0, int(target - inventory + backlog))

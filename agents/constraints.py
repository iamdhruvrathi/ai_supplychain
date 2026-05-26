"""Policy guardrails aligned with paper Section 3.2 (budget, caps, smoothing)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from simulator.config import ConstraintConfig


def apply_constraints(
    proposed_order: int,
    state: Dict[str, Any],
    config: ConstraintConfig,
    last_order: Optional[int] = None,
) -> int:
    """Apply optional guardrails to a proposed order quantity."""
    if not config.enabled:
        order = int(proposed_order)
    else:
        order = int(proposed_order)

    if config.order_cap is not None:
        order = min(order, int(config.order_cap))

    if config.budget_limit is not None:
        unit_cost = float(state.get("unit_cost", 1.0))
        max_affordable = int(config.budget_limit // max(unit_cost, 1e-9))
        order = min(order, max(0, max_affordable))

    inventory = int(state.get("inventory", 0))
    backlog = int(state.get("backlog", 0))
    if config.safety_stock_min is not None:
        target = int(config.safety_stock_min) + backlog
        pipeline = int(state.get("pipeline_inventory", state.get("incoming_shipments", 0)))
        max_needed = max(0, target - (inventory + pipeline))
        order = min(order, max_needed)

    if config.panic_order_threshold is not None and config.panic_max_order is not None:
        if backlog >= int(config.panic_order_threshold):
            order = min(order, int(config.panic_max_order))

    if config.order_smoothing_alpha is not None and last_order is not None:
        alpha = float(config.order_smoothing_alpha)
        alpha = max(0.0, min(1.0, alpha))
        smoothed = alpha * order + (1.0 - alpha) * int(last_order)
        order = int(round(smoothed))

    return max(0, order)

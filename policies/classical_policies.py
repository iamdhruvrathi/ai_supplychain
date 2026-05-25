"""Classical Beer Game replenishment policies."""

from typing import Dict, List


def base_stock_order(
    node_state: Dict[str, object],
    target_inventory: int = 20
) -> int:
    """Order enough to restore pipeline + inventory to a target level."""
    inventory = node_state["inventory"]
    backlog = node_state["backlog"]
    pipeline = sum(node_state["incoming_shipments"])

    target_level = target_inventory + backlog
    order = target_level - (inventory + pipeline)
    return max(0, int(round(order)))


def order_up_to_policy(
    node_state: Dict[str, object],
    target_inventory: int = 20
) -> int:
    """Order-up-to policy using current inventory, backlog, and pipeline."""
    return base_stock_order(node_state, target_inventory)


def moving_average_order(
    node_state: Dict[str, object],
    demand_history: List[int],
    window: int = 3,
    target_inventory: int = 20
) -> int:
    """Use moving average demand to size the next replenishment order."""
    if not demand_history:
        return target_inventory

    window = min(len(demand_history), window)
    average_demand = sum(demand_history[-window:]) / window
    pipeline = sum(node_state["incoming_shipments"])
    backlog = node_state["backlog"]

    order_up_to = int(round(
        average_demand * (len(node_state["incoming_shipments"]) + 1)
        + target_inventory
    ))

    order = order_up_to - (node_state["inventory"] + pipeline - backlog)
    return max(0, order)

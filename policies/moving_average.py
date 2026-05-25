from typing import Dict, List


def moving_average_order(node_state: Dict[str, object], demand_history: List[int], window: int = 3, target_inventory: int = 20) -> int:
    """Use moving average demand to size the next replenishment order."""
    if not demand_history:
        return int(target_inventory)

    window = min(len(demand_history), window)
    average_demand = sum(demand_history[-window:]) / window
    
    # incoming_shipments can be either int (from state_dict) or list (raw)
    incoming = node_state.get("incoming_shipments", 0)
    pipeline = sum(int(x) for x in incoming) if isinstance(incoming, list) else int(incoming)
    lead_time = len(incoming) if isinstance(incoming, list) else 2
    
    backlog = int(node_state.get("backlog", 0))

    order_up_to = int(round(
        average_demand * (lead_time + 1)
        + target_inventory
    ))

    order = order_up_to - (int(node_state.get("inventory", 0)) + pipeline - backlog)
    return max(0, int(order))

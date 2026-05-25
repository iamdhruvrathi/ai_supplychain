from typing import Dict


def base_stock_order(node_state: Dict[str, object], target_inventory: int = 20) -> int:
    """Order enough to restore pipeline + inventory to a target level."""
    inventory = int(node_state.get("inventory", 0))
    backlog = int(node_state.get("backlog", 0))
    
    # incoming_shipments can be either int (from state_dict) or list (raw)
    incoming = node_state.get("incoming_shipments", 0)
    pipeline = sum(int(x) for x in incoming) if isinstance(incoming, list) else int(incoming)

    target_level = int(target_inventory) + backlog
    order = target_level - (inventory + pipeline)
    return max(0, int(round(order)))
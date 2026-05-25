from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


@dataclass
class SupplyChainNode:
    """Represents a single echelon in the Beer Game supply chain.

    The node maintains an inventory level, backlog (unsatisfied demand), a FIFO
    pipeline of incoming shipments (implemented as a deque), and the last order
    it placed. Lead time defines the pipeline length. Costs are accumulated
    per-step and tracked for research experiments.
    """

    name: str
    initial_inventory: int = 20
    lead_time: int = 2
    holding_cost: float = 1.0
    backlog_cost: float = 2.0

    inventory: int = field(init=False)
    backlog: int = field(init=False)
    incoming_shipments: Deque[int] = field(init=False)
    last_order: int = field(init=False)
    order_history: List[int] = field(init=False)

    total_holding_cost: float = field(init=False, default=0.0)
    total_backlog_cost: float = field(init=False, default=0.0)
    total_cost: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset state for new simulation runs."""
        self.inventory = int(self.initial_inventory)
        self.backlog = 0
        # FIFO pipeline: entries represent shipments that will arrive in future steps
        self.incoming_shipments = deque([0] * int(self.lead_time))
        self.last_order = 0
        self.order_history = []

        self.total_holding_cost = 0.0
        self.total_backlog_cost = 0.0
        self.total_cost = 0.0

    def receive_shipment(self) -> int:
        """Pop the next arriving shipment from the pipeline and add to inventory.

        Returns the quantity that arrived this step.
        """
        arrived = self.incoming_shipments.popleft()
        self.inventory += int(arrived)
        return int(arrived)

    def add_incoming_shipment(self, quantity: int) -> None:
        """Push a shipment to the end of the pipeline (arrival after lead_time)."""
        self.incoming_shipments.append(int(quantity))

    def fulfill_demand(self, demand: int) -> int:
        """Satisfy demand (plus backlog) using available inventory.

        Returns the quantity shipped downstream. Any unmet demand becomes backlog.
        """
        total_demand = int(demand) + int(self.backlog)
        shipped = min(self.inventory, total_demand)
        self.inventory -= shipped
        self.backlog = total_demand - shipped
        return int(shipped)

    def place_order(self, quantity: int) -> None:
        self.last_order = int(quantity)
        self.order_history.append(int(quantity))

    def compute_costs(self) -> float:
        """Compute holding and backlog costs for the current step and accumulate."""
        holding = float(self.inventory) * float(self.holding_cost)
        backlog = float(self.backlog) * float(self.backlog_cost)
        step_cost = holding + backlog
        self.total_holding_cost += holding
        self.total_backlog_cost += backlog
        self.total_cost += step_cost
        return float(step_cost)

    def get_state(self) -> dict:
        return {
            "inventory": int(self.inventory),
            "backlog": int(self.backlog),
            "incoming_shipments": list(self.incoming_shipments),
            "last_order": int(self.last_order),
        }

    def __repr__(self) -> str:
        return (
            f"{self.name:<12} | Inventory: {self.inventory:<3} | "
            f"Backlog: {self.backlog:<3} | Last Order: {self.last_order:<3} | "
            f"Total Cost: {self.total_cost:<5}"
        )

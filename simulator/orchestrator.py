"""Central orchestrator: curated information sharing (paper Section 3.3)."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from simulator.config import OrchestratorMode


class Orchestrator:
    """Augments agent observations based on information-sharing mode."""

    def __init__(
        self,
        mode: OrchestratorMode = OrchestratorMode.DECENTRALIZED,
        demand_history_window: int = 5,
    ) -> None:
        self.mode = mode
        self.demand_history_window = demand_history_window

    def augment_state(
        self,
        agent_name: str,
        local_state: Dict[str, Any],
        global_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return local state plus orchestrator fields for prompts / policies."""
        augmented = dict(local_state)
        augmented["orchestrator_mode"] = self.mode.value

        if self.mode == OrchestratorMode.DECENTRALIZED:
            return augmented

        demand_history: List[int] = global_context.get("demand_history", [])
        current_demand = global_context.get("current_demand", 0)

        if self.mode == OrchestratorMode.DEMAND_SHARING:
            augmented["shared_current_demand"] = current_demand
            return augmented

        if self.mode == OrchestratorMode.HISTORY_SHARING:
            window = demand_history[-self.demand_history_window :]
            augmented["shared_current_demand"] = current_demand
            augmented["shared_demand_history"] = window
            if len(window) >= 2:
                augmented["shared_demand_volatility"] = float(
                    statistics.pstdev(window)
                )
            else:
                augmented["shared_demand_volatility"] = 0.0
            return augmented

        if self.mode == OrchestratorMode.CENTRALIZED:
            augmented["shared_current_demand"] = current_demand
            augmented["shared_demand_history"] = demand_history[
                -self.demand_history_window :
            ]
            augmented["system_total_backlog"] = global_context.get(
                "total_backlog", 0
            )
            augmented["system_total_inventory"] = global_context.get(
                "total_inventory", 0
            )
            per_echelon = global_context.get("echelon_snapshot", {})
            augmented["echelon_snapshot"] = per_echelon
            return augmented

        return augmented

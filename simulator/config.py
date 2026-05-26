"""Simulation and experiment configuration (paper-aligned)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class OrchestratorMode(str, Enum):
    """Information-sharing regime (Section 3.3 of paper)."""

    DECENTRALIZED = "decentralized"
    DEMAND_SHARING = "demand_sharing"
    HISTORY_SHARING = "history_sharing"
    CENTRALIZED = "centralized"


@dataclass
class ConstraintConfig:
    """Inference-time guardrails (Section 3.2)."""

    enabled: bool = False
    order_cap: Optional[int] = 100
    budget_limit: Optional[float] = None
    safety_stock_min: Optional[int] = None
    order_smoothing_alpha: Optional[float] = None
    panic_order_threshold: Optional[int] = None
    panic_max_order: Optional[int] = None


@dataclass
class RewardConfig:
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.5


@dataclass
class SimulationConfig:
    """Core Beer Game parameters."""

    max_weeks: int = 30
    lead_time: int = 2
    initial_inventory: int = 20
    holding_cost: float = 1.0
    backlog_cost: float = 2.0
    demand_low: int = 2
    demand_high: int = 8
    demand_seed: Optional[int] = None
    fixed_demand_path: Optional[List[int]] = None
    verbose: bool = False
    orchestrator_mode: OrchestratorMode = OrchestratorMode.DECENTRALIZED
    demand_history_window: int = 5
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    echelon_order: tuple = (
        "Retailer",
        "Wholesaler",
        "Distributor",
        "Factory",
    )

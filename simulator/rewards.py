"""Shaped reward computation (system-level, paper-aligned for GRPO prep)."""

from __future__ import annotations

from typing import Dict, Optional

from simulator.config import RewardConfig


def compute_shaped_reward(
    total_cost: float,
    total_backlog: int,
    bullwhip_overall: Optional[float],
    config: RewardConfig,
) -> tuple[float, Dict[str, float]]:
    """R = -(alpha * cost + beta * bullwhip + gamma * backlog)."""
    bullwhip_penalty = bullwhip_overall if bullwhip_overall is not None else 0.0
    components = {
        "cost": config.alpha * total_cost,
        "bullwhip": config.beta * bullwhip_penalty,
        "backlog": config.gamma * total_backlog,
    }
    reward = -(components["cost"] + components["bullwhip"] + components["backlog"])
    return float(reward), components

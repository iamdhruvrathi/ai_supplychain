"""Standardized trajectory schema for RL post-training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    state: Dict[str, Any]
    action: int
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    policy_type: str = "unknown"
    model_name: Optional[str] = None
    week: int = 0
    agent_role: str = ""
    tool_order: Optional[int] = None
    llm_order: Optional[int] = None
    difference: Optional[int] = None
    consensus_gap: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def standardize_trajectory(
    raw_trajectories: List[Dict[str, Any]],
    policy_type: str,
    model_name: Optional[str] = None,
    max_weeks: int = 0,
) -> List[Dict[str, Any]]:
    """Convert BeerGame trajectory records to standardized RL format."""
    standardized = []
    for record in raw_trajectories:
        week = int(record.get("week", 0))
        standardized.append(
            TrajectoryStep(
                state=record.get("state", {}),
                action=int(record.get("action", 0)),
                reward=float(record.get("reward", 0.0)),
                next_state=record.get("next_state", {}),
                done=week >= max_weeks - 1 if max_weeks else False,
                info={
                    "cost": record.get("cost"),
                    "bullwhip": record.get("bullwhip"),
                    "negotiation_proposals": record.get("negotiation_proposals"),
                },
                policy_type=policy_type,
                model_name=model_name,
                week=week,
                agent_role=str(record.get("agent", "")),
                tool_order=record.get("tool_order"),
                llm_order=record.get("llm_order"),
                difference=record.get("difference"),
                consensus_gap=record.get("consensus_gap"),
            ).to_dict()
        )
    return standardized

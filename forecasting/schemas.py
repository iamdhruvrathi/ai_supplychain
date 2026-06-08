"""Structured forecast summaries for LLM orchestration (Phase 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ForecastSummary:
    """Interpreted forecast signal for one supply-chain echelon.

    Fields mirror the roadmap JSON schema so summaries can later be injected
    into agent prompts without ad-hoc formatting.
    """

    echelon: str
    forecast_model: str
    demand_next_period: float
    demand_uncertainty: str
    trend: str
    seasonality_detected: bool
    recent_mean: float
    recent_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return asdict(self)

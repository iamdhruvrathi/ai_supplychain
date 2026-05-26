"""Cost statistics and confidence intervals for repeated experiments."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Normal-approximation CI for the mean."""
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    se = statistics.pstdev(values) / (len(values) ** 0.5)
    z = 1.96 if confidence >= 0.95 else 1.645
    return (mean - z * se, mean + z * se)


def cost_summary(values: List[float]) -> Dict[str, Optional[float]]:
    mean, std = mean_std(values)
    ci = confidence_interval(values)
    return {
        "mean": mean,
        "std": std,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "median": float(np.median(values)) if values else None,
        "ci_lower": ci[0] if ci else None,
        "ci_upper": ci[1] if ci else None,
        "n": len(values),
    }

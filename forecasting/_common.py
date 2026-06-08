"""Shared helpers for lightweight time-series forecasters."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

MIN_HISTORY_FOR_TREND = 3
MIN_HISTORY_FOR_SEASONALITY = 8


def clean_history(history: Sequence[float]) -> List[float]:
    """Coerce history to a non-empty float list; fall back to [0.0]."""
    values = [float(x) for x in history if x is not None and not math.isnan(float(x))]
    return values if values else [0.0]


def recent_stats(history: Sequence[float], window: int = 5) -> Tuple[float, float]:
    """Mean and population std of the trailing window."""
    values = clean_history(history)
    tail = values[-window:] if len(values) >= window else values
    mean = float(np.mean(tail))
    std = float(np.std(tail)) if len(tail) > 1 else 0.0
    return mean, std


def classify_uncertainty(recent_std: float, recent_mean: float) -> str:
    """Map coefficient-of-variation style spread to a coarse label."""
    if recent_mean <= 0:
        return "high" if recent_std > 0 else "low"
    cv = recent_std / abs(recent_mean)
    if cv < 0.15:
        return "low"
    if cv < 0.40:
        return "medium"
    return "high"


def detect_trend(history: Sequence[float]) -> str:
    """Estimate direction from a simple slope over the trailing window."""
    values = clean_history(history)
    if len(values) < MIN_HISTORY_FOR_TREND:
        return "stable"

    tail = values[-min(6, len(values)) :]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, deg=1)[0])
    scale = max(float(np.mean(tail)), 1.0)
    normalized = slope / scale

    if normalized > 0.05:
        return "rising"
    if normalized > 0.02:
        return "slightly_upward"
    if normalized < -0.05:
        return "falling"
    if normalized < -0.02:
        return "slightly_downward"
    return "stable"


def detect_seasonality(history: Sequence[float], period: int = 4) -> bool:
    """Heuristic seasonality flag based on lagged autocorrelation."""
    values = clean_history(history)
    if len(values) < MIN_HISTORY_FOR_SEASONALITY:
        return False

    series = np.asarray(values, dtype=float)
    centered = series - series.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return False

    lag = min(period, len(centered) - 1)
    if lag < 1:
        return False

    acf = float(np.dot(centered[lag:], centered[:-lag]) / denom)
    return acf > 0.45

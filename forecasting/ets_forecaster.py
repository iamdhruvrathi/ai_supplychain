"""Exponential smoothing (ETS) forecaster for upstream echelons.

Implements Holt's linear trend method when enough history exists; otherwise
falls back to simple exponential smoothing (level only).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from forecasting._common import (
    classify_uncertainty,
    clean_history,
    detect_seasonality,
    detect_trend,
    recent_stats,
)
from forecasting.schemas import ForecastSummary

MODEL_NAME = "ets"
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.1


def _simple_exponential_smoothing(values: List[float], alpha: float = DEFAULT_ALPHA) -> float:
    """Level-only ETS: exponentially weighted moving average of the series."""
    level = values[0]
    for observation in values[1:]:
        level = alpha * observation + (1.0 - alpha) * level
    return level


def _holt_forecast(
    values: List[float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """Holt linear trend: forecast one step ahead from level + trend."""
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0.0

    for observation in values[1:]:
        prev_level = level
        level = alpha * observation + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    return max(0.0, level + trend)


def forecast(history: Sequence[float], echelon: str = "Unknown") -> ForecastSummary:
    """Forecast the next period using exponential smoothing.

    Forecast generation:
        * len < 3  -> simple exponential smoothing (level only)
        * len >= 3 -> Holt's method (level + linear trend)

    Trend detection:
        Independent slope label from ``detect_trend`` for LLM context.

    Uncertainty estimation:
        Trailing-window std mapped to low/medium/high labels.
    """
    values = clean_history(history)
    mean, std = recent_stats(values)

    if len(values) < 3:
        point = _simple_exponential_smoothing(values)
    else:
        point = _holt_forecast(values)

    return ForecastSummary(
        echelon=echelon,
        forecast_model=MODEL_NAME,
        demand_next_period=float(max(0.0, point)),
        demand_uncertainty=classify_uncertainty(std, mean),
        trend=detect_trend(values),
        seasonality_detected=detect_seasonality(values),
        recent_mean=mean,
        recent_std=std,
    )

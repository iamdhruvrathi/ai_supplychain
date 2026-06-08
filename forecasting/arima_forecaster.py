"""Lightweight ARIMA-style forecaster for retailer (echelon 1) demand.

Uses a compact AR(1) fit when enough history is available. For very short
series we fall back to the recent mean so the pipeline never crashes.
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

MODEL_NAME = "arima"


def _fit_ar1(values: List[float]) -> tuple[float, float]:
    """Estimate AR(1): y_t = intercept + phi * y_{t-1} via least squares."""
    y = np.asarray(values[1:], dtype=float)
    x = np.asarray(values[:-1], dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _ar1_forecast(values: List[float]) -> float:
    """One-step-ahead point forecast from an AR(1) model."""
    intercept, phi = _fit_ar1(values)
    last = values[-1]
    return max(0.0, intercept + phi * last)


def forecast(history: Sequence[float], echelon: str = "Retailer") -> ForecastSummary:
    """Forecast the next period from demand/order history.

    Forecast generation:
        * len < 2  -> recent mean (no lag structure identifiable)
        * len >= 2 -> AR(1) one-step forecast (ARIMA(1,0,0) proxy)

    Trend detection:
        Slope of the trailing window via ``detect_trend`` (see ``_common``).

    Uncertainty estimation:
        ``recent_std`` over the trailing window, mapped to low/medium/high
        using coefficient-of-variation thresholds.
    """
    values = clean_history(history)
    mean, std = recent_stats(values)

    if len(values) < 2:
        point = mean
    else:
        point = _ar1_forecast(values)

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

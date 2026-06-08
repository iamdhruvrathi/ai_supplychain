"""Tests for Phase 1 forecasting foundation."""

from __future__ import annotations

import json

from forecasting.arima_forecaster import forecast as arima_forecast
from forecasting.ets_forecaster import forecast as ets_forecast
from forecasting.schemas import ForecastSummary
from tools.forecast_tool import get_forecast, load_echelon_model_map


def test_forecast_summary_to_dict() -> None:
    summary = ForecastSummary(
        echelon="Retailer",
        forecast_model="arima",
        demand_next_period=8.0,
        demand_uncertainty="low",
        trend="stable",
        seasonality_detected=False,
        recent_mean=6.0,
        recent_std=1.0,
    )
    payload = summary.to_dict()
    assert payload["echelon"] == "Retailer"
    json.dumps(payload)


def test_arima_handles_short_history() -> None:
    summary = arima_forecast([], echelon="Retailer")
    assert summary.demand_next_period == 0.0
    assert summary.forecast_model == "arima"


def test_ets_handles_short_history() -> None:
    summary = ets_forecast([5.0], echelon="Wholesaler")
    assert summary.demand_next_period == 5.0
    assert summary.forecast_model == "ets"


def test_echelon_map_and_dispatch() -> None:
    mapping = load_echelon_model_map()
    assert mapping["Retailer"] == "arima"
    assert mapping["Factory"] == "ets"

    retailer = get_forecast("Retailer", [4, 4, 4, 8, 8])
    factory = get_forecast("Factory", [10, 12, 14, 16])
    assert retailer.forecast_model == "arima"
    assert factory.forecast_model == "ets"

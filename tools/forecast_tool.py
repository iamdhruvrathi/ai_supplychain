"""Echelon-aware forecasting tool (Phase 1).

Reads ``configs/forecast_echelon_map.yaml``, dispatches to the configured
forecaster, and returns a structured :class:`ForecastSummary`.

Not integrated with LLM prompts or agents yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from forecasting import arima_forecast, ets_forecast
from forecasting.schemas import ForecastSummary

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "forecast_echelon_map.yaml"

FORECASTERS: Dict[str, Callable[..., ForecastSummary]] = {
    "arima": arima_forecast,
    "ets": ets_forecast,
}


def load_echelon_model_map(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Load echelon -> model name mapping from YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    mapping = data.get("echelon_models", {})
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Invalid echelon_models in {path}")

    return {str(echelon): str(model).lower() for echelon, model in mapping.items()}


def get_forecast(
    echelon: str,
    history: Sequence[float],
    *,
    config_path: Optional[Union[str, Path]] = None,
    model_override: Optional[str] = None,
) -> ForecastSummary:
    """Return a forecast summary for the given echelon and demand history.

    Parameters
    ----------
    echelon:
        Supply-chain role (e.g. ``Retailer``).
    history:
        Past demand or order observations, oldest to newest.
    config_path:
        Optional path to ``forecast_echelon_map.yaml``.
    model_override:
        Force a model name (``arima`` or ``ets``) instead of config lookup.
    """
    if model_override:
        model_name = model_override.lower()
    else:
        mapping = load_echelon_model_map(config_path)
        model_name = mapping.get(echelon, "ets").lower()

    forecaster = FORECASTERS.get(model_name)
    if forecaster is None:
        raise ValueError(
            f"Unknown forecast model '{model_name}' for echelon '{echelon}'. "
            f"Supported: {sorted(FORECASTERS)}"
        )

    return forecaster(history, echelon=echelon)


def forecast_all_echelons(
    histories: Mapping[str, Sequence[float]],
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, ForecastSummary]:
    """Forecast every echelon present in ``histories``."""
    return {
        echelon: get_forecast(echelon, history, config_path=config_path)
        for echelon, history in histories.items()
    }


if __name__ == "__main__":
    # --- Example usage (no LLM / simulator integration) ---
    sample_histories: Dict[str, List[float]] = {
        "Retailer": [4, 4, 4, 4, 8, 8, 8, 8],
        "Wholesaler": [6, 10, 14, 22, 30, 28, 26, 24],
        "Distributor": [8, 12, 18, 24, 32, 30, 28, 26],
        "Factory": [10, 15, 20, 28, 36, 34, 32, 30],
    }

    print("Echelon model map:", load_echelon_model_map())
    print()

    for echelon, history in sample_histories.items():
        summary = get_forecast(echelon, history)
        print(f"{echelon} ({summary.forecast_model}):")
        print(f"  next period : {summary.demand_next_period:.2f}")
        print(f"  uncertainty : {summary.demand_uncertainty}")
        print(f"  trend       : {summary.trend}")
        print(f"  seasonality : {summary.seasonality_detected}")
        print(f"  recent mean/std: {summary.recent_mean:.2f} / {summary.recent_std:.2f}")
        print(f"  dict        : {summary.to_dict()}")
        print()

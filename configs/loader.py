"""Load experiment configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from simulator.config import (
    ConstraintConfig,
    OrchestratorMode,
    RewardConfig,
    SimulationConfig,
)
from simulator.demand import mit_beer_game_demand_path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_from_dict(data: Dict[str, Any]) -> SimulationConfig:
    exp = data.get("experiment", {})
    orch = data.get("orchestrator", {})
    tools = data.get("tools", {})
    cons = data.get("constraints", {})
    rew = data.get("reward", {})

    mode_str = orch.get("mode", "decentralized")
    try:
        mode = OrchestratorMode(mode_str)
    except ValueError:
        mode = OrchestratorMode.DECENTRALIZED

    constraints = ConstraintConfig(
        enabled=bool(cons.get("enabled", False)),
        order_cap=cons.get("order_cap"),
        budget_limit=cons.get("budget_limit"),
        safety_stock_min=cons.get("safety_stock_min"),
        order_smoothing_alpha=cons.get("order_smoothing_alpha"),
        panic_order_threshold=cons.get("panic_order_threshold"),
        panic_max_order=cons.get("panic_max_order"),
    )

    weeks = int(exp.get("weeks", 30))
    fixed_demand_path = exp.get("fixed_demand_path")
    demand_pattern = str(exp.get("demand_pattern", "")).lower()
    if fixed_demand_path is None and demand_pattern in {"mit", "mit_beer_game"}:
        fixed_demand_path = mit_beer_game_demand_path(weeks)

    return SimulationConfig(
        max_weeks=weeks,
        lead_time=int(exp.get("lead_time", 2)),
        initial_inventory=int(exp.get("initial_inventory", 20)),
        demand_seed=exp.get("demand_seed"),
        fixed_demand_path=fixed_demand_path,
        verbose=bool(exp.get("verbose", False)),
        orchestrator_mode=mode,
        demand_history_window=int(orch.get("demand_history_window", 5)),
        use_tool_recommendation=bool(tools.get("use_tool_recommendation", False)),
        constraints=constraints,
        reward=RewardConfig(
            alpha=float(rew.get("alpha", 1.0)),
            beta=float(rew.get("beta", 0.1)),
            gamma=float(rew.get("gamma", 0.5)),
        ),
    )


def load_experiment_config(path: str) -> SimulationConfig:
    return config_from_dict(load_yaml(path))


def load_full_experiment(path: str) -> Dict[str, Any]:
    """Return full YAML dict plus parsed SimulationConfig."""
    raw = load_yaml(path)
    return {
        "raw": raw,
        "simulation": config_from_dict(raw),
        "agents": raw.get("agents", {}),
        "evaluation": raw.get("evaluation", {}),
        "output": raw.get("output", {}),
    }

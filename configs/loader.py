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

    return SimulationConfig(
        max_weeks=int(exp.get("weeks", 30)),
        lead_time=int(exp.get("lead_time", 2)),
        initial_inventory=int(exp.get("initial_inventory", 20)),
        demand_seed=exp.get("demand_seed"),
        verbose=bool(exp.get("verbose", False)),
        orchestrator_mode=mode,
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

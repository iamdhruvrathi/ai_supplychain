"""Repeated-run experiment engine for paper replication (Section 4).

Runs R simulations with identical demand path, environment config, and seeds
for demand — isolating LLM decision stochasticity.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from agents.constraints import apply_constraints
from agents.llm_agent import LLMAgent
from metrics.agent_bullwhip import agent_bullwhip_report, orders_tensor_from_runs
from metrics.bullwhip import bullwhip_per_agent
from metrics.cost_analysis import cost_summary
from metrics.reliability import reliability_summary
from policies.base_stock import base_stock_order
from simulator.beer_game import BeerGame
from simulator.config import OrchestratorMode, SimulationConfig
from simulator.demand import DemandGenerator
from trajectories.schema import standardize_trajectory
from trajectories.writer import TrajectoryWriter

logger = logging.getLogger(__name__)

ECHELONS = ("Retailer", "Wholesaler", "Distributor", "Factory")


def _build_env(config: SimulationConfig) -> BeerGame:
    return BeerGame(
        max_weeks=config.max_weeks,
        verbose=config.verbose,
        alpha=config.reward.alpha,
        beta=config.reward.beta,
        gamma=config.reward.gamma,
        simulation_config=config,
    )


def _apply_orchestrator_states(env: BeerGame) -> Dict[str, Dict]:
    states = {}
    for node in env.nodes:
        local = env.get_agent_state(node.name)
        states[node.name] = local
    return states


def run_single_episode(
    config: SimulationConfig,
    policy_fn: Callable[[BeerGame, str, Dict], int],
    run_id: int = 0,
    llm_agents: Optional[Dict[str, LLMAgent]] = None,
) -> Dict[str, Any]:
    """Run one episode; demand path fixed by config.demand_seed."""
    random.seed(config.demand_seed if config.demand_seed is not None else run_id)

    env = _build_env(config)
    env.reset()

    done = False
    while not done:
        states = _apply_orchestrator_states(env)
        actions = {}
        for name in ECHELONS:
            if llm_agents and name in llm_agents:
                raw = llm_agents[name].generate_order(states[name], fallback=0)
                policy_type = "llm"
            else:
                raw = policy_fn(env, name, states[name])
                policy_type = "heuristic"

            last = states[name].get("last_order", 0)
            actions[name] = apply_constraints(
                raw,
                states[name],
                config.constraints,
                last_order=last,
            )

        _, _, done, info = env.step(actions)

    history = env.get_history()
    bull = bullwhip_per_agent(history)

    return {
        "run_id": run_id,
        "history": history,
        "total_cost": history["total_cost"][-1] if history["total_cost"] else 0.0,
        "bullwhip": bull,
        "trajectories": env.get_trajectories(),
        "metrics": env.compute_metrics() if hasattr(env, "compute_metrics") else {},
        "final_info": info,
    }


def run_repeated_experiment(
    config: SimulationConfig,
    n_runs: int = 30,
    policy_fn: Optional[Callable] = None,
    model_name: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    save_dir: str = "results/repeated_runs",
    export_trajectories: bool = True,
) -> Dict[str, Any]:
    """Execute R repeated runs and compute paper-aligned metrics."""
    os.makedirs(save_dir, exist_ok=True)

    if policy_fn is None:

        def _base_stock(env, name, state):
            return base_stock_order(state, target_inventory=20)

        policy_fn = _base_stock

    llm_agents = None
    policy_type = "heuristic"
    if model_name:
        policy_type = "llm"
        llm_agents = {
            name: LLMAgent(
                agent_name=name,
                model_name=model_name,
                ollama_url=ollama_url,
                max_order=config.constraints.order_cap or 100,
                timeout=120.0,
            )
            for name in ECHELONS
        }

    episodes: List[Dict] = []
    total_costs: List[float] = []

    for r in range(n_runs):
        logger.info("Repeated run %s / %s", r + 1, n_runs)
        ep = run_single_episode(
            config,
            policy_fn=policy_fn,
            run_id=r,
            llm_agents=llm_agents,
        )
        episodes.append(ep)
        total_costs.append(ep["total_cost"])

    histories = [ep["history"] for ep in episodes]
    orders_tensor = orders_tensor_from_runs(histories)
    agent_bw = agent_bullwhip_report(orders_tensor)
    reliability = reliability_summary(total_costs)
    costs = cost_summary(total_costs)

    orders_by_echelon: Dict[str, List[List[int]]] = {e: [] for e in ECHELONS}
    inv_by_echelon: Dict[str, List[List[int]]] = {e: [] for e in ECHELONS}
    bl_by_echelon: Dict[str, List[List[int]]] = {e: [] for e in ECHELONS}
    for h in histories:
        for e in ECHELONS:
            orders_by_echelon[e].append(h["orders"].get(e, []))
            inv_by_echelon[e].append(h["inventory"].get(e, []))
            bl_by_echelon[e].append(h["backlog"].get(e, []))

    reliability_full = reliability_summary(
        total_costs,
        orders_by_echelon=orders_by_echelon,
        inventories_by_echelon=inv_by_echelon,
        backlogs_by_echelon=bl_by_echelon,
    )

    report = {
        "n_runs": n_runs,
        "config": {
            "max_weeks": config.max_weeks,
            "demand_seed": config.demand_seed,
            "orchestrator_mode": config.orchestrator_mode.value,
            "constraints_enabled": config.constraints.enabled,
            "model_name": model_name,
        },
        "cost": costs,
        "reliability": reliability_full,
        "agent_bullwhip": agent_bw,
        "total_costs": total_costs,
    }

    report_path = os.path.join(save_dir, "repeated_runs_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    rows = [
        {"run_id": r, "total_cost": total_costs[r]}
        for r in range(len(total_costs))
    ]
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "run_costs.csv"), index=False)

    if export_trajectories and episodes:
        writer = TrajectoryWriter(os.path.join(save_dir, "trajectories"))
        all_steps = []
        for ep in episodes:
            steps = standardize_trajectory(
                ep["trajectories"],
                policy_type=policy_type,
                model_name=model_name,
                max_weeks=config.max_weeks,
            )
            all_steps.extend(steps)
        writer.write_all(all_steps, stem="rollouts")

    logger.info("Report saved to %s", report_path)

    try:
        from evaluation.plotting import generate_research_plots
        plot_dir = os.path.join("plots", "research")
        generate_research_plots(report, plot_dir=plot_dir)
        logger.info("Research plots saved to %s", plot_dir)
    except ImportError as exc:
        logger.warning("Plotting skipped: %s", exc)

    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Repeated-run reliability experiment")
    parser.add_argument("--weeks", type=int, default=30)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--demand-seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        from configs.loader import load_experiment_config
        config = load_experiment_config(args.config)
        n_runs = args.runs
        model = args.model
    else:
        from simulator.config import SimulationConfig
        config = SimulationConfig(
            max_weeks=args.weeks,
            demand_seed=args.demand_seed,
        )
        n_runs = args.runs
        model = None if args.offline else args.model

    report = run_repeated_experiment(
        config=config,
        n_runs=n_runs,
        model_name=model,
    )
    print(f"Mean cost: {report['cost']['mean']:.2f}, CV: {report['reliability'].get('coefficient_of_variation')}")

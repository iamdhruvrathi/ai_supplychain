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
import time
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
from simulator.demand import DemandGenerator, mit_beer_game_demand_path
from trajectories.schema import standardize_trajectory
from trajectories.writer import TrajectoryWriter

logger = logging.getLogger(__name__)

ECHELONS = ("Retailer", "Wholesaler", "Distributor", "Factory")


def _summarize_consensus(histories: List[Dict[str, Any]]) -> Dict[str, float]:
    gaps = [
        gap
        for history in histories
        for gap in history.get("consensus_gap", [])
        if gap is not None
    ]
    if not gaps:
        return {"mean_consensus_gap": 0.0, "max_consensus_gap": 0.0}
    return {
        "mean_consensus_gap": float(sum(gaps) / len(gaps)),
        "max_consensus_gap": float(max(gaps)),
    }


def _llm_order(
    agent: LLMAgent,
    state: Dict[str, Any],
    fallback_order: int,
    n_samples: int,
) -> int:
    if n_samples > 1:
        return agent.generate_order_majority_vote(
            state,
            n_samples=n_samples,
            fallback=fallback_order,
        )
    return agent.generate_order(state, fallback=fallback_order)


def _metadata_from_agent(agent: LLMAgent) -> Dict[str, Any]:
    return dict(getattr(agent, "last_decision_metadata", {}) or {})


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
    progress: str = "none",
    n_samples: int = 1,
) -> Dict[str, Any]:
    """Run one episode; demand path fixed by config.demand_seed."""
    random.seed(config.demand_seed if config.demand_seed is not None else run_id)

    env = _build_env(config)
    env.reset()

    done = False
    run_start = time.time()
    while not done:
        states = _apply_orchestrator_states(env)
        actions = {}
        action_metadata = {}

        if config.orchestrator_mode == OrchestratorMode.NEGOTIATION and llm_agents:
            proposals = {}
            for name in ECHELONS:
                fallback_order = policy_fn(env, name, states[name])
                proposals[name] = _llm_order(
                    llm_agents[name],
                    states[name],
                    fallback_order,
                    n_samples,
                )

            for name in ECHELONS:
                negotiation_state = dict(states[name])
                negotiation_state["negotiation_round"] = 2
                negotiation_state["negotiation_proposals"] = dict(proposals)
                raw = _llm_order(
                    llm_agents[name],
                    negotiation_state,
                    proposals[name],
                    n_samples,
                )
                metadata = _metadata_from_agent(llm_agents[name])
                metadata["negotiation_proposals"] = dict(proposals)
                action_metadata[name] = metadata

                last = states[name].get("last_order", 0)
                actions[name] = apply_constraints(
                    raw,
                    states[name],
                    config.constraints,
                    last_order=last,
                )
        else:
            for name in ECHELONS:
                if llm_agents and name in llm_agents:
                    fallback_order = policy_fn(env, name, states[name])
                    raw = _llm_order(
                        llm_agents[name],
                        states[name],
                        fallback_order,
                        n_samples,
                    )
                    action_metadata[name] = _metadata_from_agent(llm_agents[name])
                else:
                    raw = policy_fn(env, name, states[name])

                last = states[name].get("last_order", 0)
                actions[name] = apply_constraints(
                    raw,
                    states[name],
                    config.constraints,
                    last_order=last,
                )

        _, _, done, info = env.step(actions, action_metadata=action_metadata)
        if progress == "week":
            action_text = ", ".join(f"{name}={qty}" for name, qty in actions.items())
            print(
                f"[run {run_id + 1}] week {info['week']}/{config.max_weeks} "
                f"demand={info['customer_demand']} cost={info['total_system_cost']:.2f} "
                f"actions: {action_text} elapsed={time.time() - run_start:.1f}s",
                flush=True,
            )

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
    backend: str = "ollama",
    save_dir: str = "results/repeated_runs",
    export_trajectories: bool = True,
    timeout: float = 120.0,
    temperature: float = 0.2,
    num_predict: int = 8,
    progress: str = "run",
    n_samples: int = 1,
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
        print(f"Backend : {backend.capitalize()}")
        print(f"Model   : {model_name}")
        llm_agents = {
            name: LLMAgent(
                agent_name=name,
                model_name=model_name,
                ollama_url=ollama_url,
                max_order=config.constraints.order_cap or 100,
                temperature=temperature,
                timeout=timeout,
                num_predict=num_predict,
                backend=backend,
                use_tool_recommendation=config.use_tool_recommendation,
            )
            for name in ECHELONS
        }

    episodes: List[Dict] = []
    total_costs: List[float] = []
    demand_path = config.fixed_demand_path or DemandGenerator.from_config(config).path

    def write_outputs(status: str) -> Dict[str, Any]:
        histories = [ep["history"] for ep in episodes]
        orders_tensor = orders_tensor_from_runs(histories)
        agent_bw = agent_bullwhip_report(orders_tensor)
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

        consensus = _summarize_consensus(histories)
        report = {
            "n_runs": len(episodes),
            "requested_runs": n_runs,
            "status": status,
            "config": {
                "max_weeks": config.max_weeks,
                "demand_seed": config.demand_seed,
                "fixed_demand_path": demand_path,
                "orchestrator_mode": config.orchestrator_mode.value,
                "use_tool_recommendation": config.use_tool_recommendation,
                "constraints_enabled": config.constraints.enabled,
                "model_name": model_name,
            },
            "cost": costs,
            "reliability": reliability_full,
            "agent_bullwhip": agent_bw,
            "consensus": consensus,
            "mean_consensus_gap": consensus["mean_consensus_gap"],
            "max_consensus_gap": consensus["max_consensus_gap"],
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

        return report

    if demand_path:
        preview = ", ".join(str(x) for x in demand_path[: min(10, len(demand_path))])
        suffix = "..." if len(demand_path) > 10 else ""
        print(f"Fixed demand path ({len(demand_path)} weeks): {preview}{suffix}", flush=True)

    for r in range(n_runs):
        logger.info("Repeated run %s / %s", r + 1, n_runs)
        run_start = time.time()
        if progress != "none":
            print(f"[run {r + 1}/{n_runs}] starting", flush=True)
        try:
            ep = run_single_episode(
                config,
                policy_fn=policy_fn,
                run_id=r,
                llm_agents=llm_agents,
                progress=progress,
                n_samples=n_samples,
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Keeping {len(episodes)} completed run(s).", flush=True)
            if not episodes:
                raise
            return write_outputs("interrupted")
        episodes.append(ep)
        total_costs.append(ep["total_cost"])
        if progress != "none":
            print(
                f"[run {r + 1}/{n_runs}] done total_cost={ep['total_cost']:.2f} "
                f"run_time={time.time() - run_start:.1f}s",
                flush=True,
            )
        write_outputs("partial" if len(episodes) < n_runs else "complete")

    report = write_outputs("complete")

    logger.info("Report saved to %s", os.path.join(save_dir, "repeated_runs_report.json"))

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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=8)
    parser.add_argument("--progress", choices=("none", "run", "week"), default="week")
    parser.add_argument("--output-dir", type=str, default="results/repeated_runs")
    parser.add_argument("--demand-pattern", choices=("mit", "seeded", "random"), default="mit")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument(
        "--orchestrator-mode",
        choices=tuple(mode.value for mode in OrchestratorMode),
        default=None,
    )
    parser.add_argument("--use-tool-recommendation", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("ollama", "groq"),
        default="ollama",
        help="Inference backend to use: ollama (local) or groq (remote)",
    )
    args = parser.parse_args()

    if args.config:
        from configs.loader import load_experiment_config
        config = load_experiment_config(args.config)
        if args.orchestrator_mode:
            config.orchestrator_mode = OrchestratorMode(args.orchestrator_mode)
        if args.use_tool_recommendation:
            config.use_tool_recommendation = True
        n_runs = args.runs
        model = args.model
    else:
        from simulator.config import SimulationConfig
        fixed_demand_path = None
        demand_seed = args.demand_seed
        if args.demand_pattern == "mit":
            fixed_demand_path = mit_beer_game_demand_path(args.weeks)
            demand_seed = None
        elif args.demand_pattern == "random":
            demand_seed = None
        config = SimulationConfig(
            max_weeks=args.weeks,
            demand_seed=demand_seed,
            fixed_demand_path=fixed_demand_path,
            orchestrator_mode=(
                OrchestratorMode(args.orchestrator_mode)
                if args.orchestrator_mode
                else OrchestratorMode.DECENTRALIZED
            ),
            use_tool_recommendation=args.use_tool_recommendation,
        )
        n_runs = args.runs
        model = None if args.offline else args.model

    report = run_repeated_experiment(
        config=config,
        n_runs=n_runs,
        model_name=model,
        backend=args.backend,
        save_dir=args.output_dir,
        timeout=args.timeout,
        temperature=args.temperature,
        num_predict=args.num_predict,
        progress=args.progress,
        n_samples=max(1, args.n_samples),
    )
    print(f"Mean cost: {report['cost']['mean']:.2f}, CV: {report['reliability'].get('coefficient_of_variation')}")

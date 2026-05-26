"""Repeated model evaluation: LLM policies vs classical baselines.

Usage:
    python evaluation/compare_models.py
    python evaluation/compare_models.py --weeks 30 --repeats 10
    python evaluation/compare_models.py --offline   # stub LLM (no Ollama)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from agents.llm_agent import LLMAgent
from evaluation.comparison_plots import generate_all_comparison_plots
from metrics.stability import cumulative_instability, stability_summary
from policies.base_stock import base_stock_order
from simulator.beer_game import BeerGame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ECHELONS = ("Retailer", "Wholesaler", "Distributor", "Factory")

MODELS = {
    "qwen2.5:1.5b": "llm",
    "deepseek-r1:1.5b": "llm",
    "base_stock": "policy",
}


def _mean_backlog(history: Dict) -> float:
    backlogs = history.get("backlog", {})
    if not backlogs:
        return 0.0
    per_agent = [
        sum(vals) / len(vals) if vals else 0.0
        for vals in backlogs.values()
    ]
    return float(sum(per_agent) / len(per_agent))


def _collect_weekly_metrics(env: BeerGame, actions: Dict[str, int]) -> Dict:
    """Build weekly metric dict from env after a completed run."""
    history = env.get_history()
    weeks = len(history.get("demand", []))
    weekly: Dict[str, List] = {
        "reward": list(history.get("reward", [])),
        "bullwhip_overall": [],
        "total_system_cost": list(history.get("step_cost", [])),
    }
    for bw in history.get("bullwhip", []):
        overall = bw.get("overall") if bw else None
        weekly["bullwhip_overall"].append(overall)

    for name in ECHELONS:
        weekly[f"order_{name}"] = history["orders"].get(name, [])
        weekly[f"inventory_{name}"] = history["inventory"].get(name, [])
        weekly[f"backlog_{name}"] = history["backlog"].get(name, [])

    return weekly


def run_base_stock_simulation(
    seed: int,
    max_weeks: int = 30,
    target_inventory: int = 20,
) -> Dict:
    random.seed(seed)
    env = BeerGame(max_weeks=max_weeks, verbose=False)
    env.reset()

    done = False
    while not done:
        actions = {
            node.name: base_stock_order(
                env.get_state_dict(node.name),
                target_inventory=target_inventory,
            )
            for node in env.nodes
        }
        _, _, done, info = env.step(actions)

    history = env.get_history()
    stability = stability_summary(history)
    bull = info.get("bullwhip") or {}

    return {
        "total_cost": history["total_cost"][-1] if history["total_cost"] else 0.0,
        "bullwhip_overall": bull.get("overall"),
        "avg_backlog": _mean_backlog(history),
        "cumulative_reward": sum(history.get("reward", [])),
        "reward_trajectory": history.get("reward", []),
        "instability": stability["cumulative_instability"],
        "weekly_metrics": _collect_weekly_metrics(env, {}),
        "trajectory_count": len(env.get_trajectories()),
    }


def run_llm_simulation(
    seed: int,
    model_name: str,
    max_weeks: int = 30,
    ollama_url: str = "http://localhost:11434",
    max_order: int = 100,
    order_fn: Optional[Callable[[LLMAgent, Dict], int]] = None,
) -> Dict:
    random.seed(seed)
    env = BeerGame(max_weeks=max_weeks, verbose=False)
    env.reset()

    agents = {
        name: LLMAgent(
            agent_name=name,
            model_name=model_name,
            ollama_url=ollama_url,
            max_order=max_order,
            timeout=120.0,
        )
        for name in ECHELONS
    }

    done = False
    while not done:
        states = env.get_all_states()
        if order_fn is not None:
            actions = {
                name: order_fn(agents[name], states[name])
                for name in ECHELONS
            }
        else:
            actions = {
                name: agents[name].generate_order(states[name], fallback=0)
                for name in ECHELONS
            }
        _, _, done, info = env.step(actions)

    history = env.get_history()
    stability = stability_summary(history)
    bull = info.get("bullwhip") or {}

    return {
        "total_cost": history["total_cost"][-1] if history["total_cost"] else 0.0,
        "bullwhip_overall": bull.get("overall"),
        "avg_backlog": _mean_backlog(history),
        "cumulative_reward": sum(history.get("reward", [])),
        "reward_trajectory": history.get("reward", []),
        "instability": stability["cumulative_instability"],
        "weekly_metrics": _collect_weekly_metrics(env, {}),
        "trajectory_count": len(env.get_trajectories()),
    }


def _average_weekly_metrics(
    runs: List[Dict],
) -> Dict[str, List[float]]:
    """Average weekly series across repeated runs."""
    if not runs:
        return {}

    keys = runs[0]["weekly_metrics"].keys()
    averaged: Dict[str, List[float]] = {}

    for key in keys:
        series_list = [r["weekly_metrics"][key] for r in runs]
        min_len = min(len(s) for s in series_list)
        if min_len == 0:
            continue
        averaged[key] = []
        for week_idx in range(min_len):
            values = [
                s[week_idx]
                for s in series_list
                if s[week_idx] is not None
            ]
            if not values:
                averaged[key].append(0.0)
            elif all(isinstance(v, (int, float)) for v in values):
                averaged[key].append(float(sum(values) / len(values)))
            else:
                numeric = [v for v in values if isinstance(v, (int, float))]
                averaged[key].append(
                    float(sum(numeric) / len(numeric)) if numeric else 0.0
                )

    return averaged


def run_comparison(
    repeats: int = 10,
    max_weeks: int = 30,
    out_csv: str = "results/model_comparison.csv",
    ollama_url: str = "http://localhost:11434",
    offline: bool = False,
    plot: bool = True,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows: List[Dict] = []
    weekly_by_model: Dict[str, Dict[str, List[float]]] = {}
    runs_by_model: Dict[str, List[Dict]] = {m: [] for m in MODELS}

    stub_order = (lambda agent, state: 5) if offline else None

    for model_name, model_type in MODELS.items():
        logger.info("Evaluating %s (%s runs)...", model_name, repeats)

        for run_id in range(repeats):
            seed = 1000 + run_id
            try:
                if model_type == "policy":
                    result = run_base_stock_simulation(seed, max_weeks=max_weeks)
                else:
                    result = run_llm_simulation(
                        seed,
                        model_name=model_name,
                        max_weeks=max_weeks,
                        ollama_url=ollama_url,
                        order_fn=stub_order,
                    )
            except Exception as exc:
                logger.error(
                    "Run failed for %s (seed=%s): %s", model_name, seed, exc
                )
                continue

            runs_by_model[model_name].append(result)
            rows.append({
                "model": model_name,
                "run_id": run_id,
                "seed": seed,
                "total_cost": result["total_cost"],
                "bullwhip_overall": result["bullwhip_overall"],
                "avg_backlog": result["avg_backlog"],
                "cumulative_reward": result["cumulative_reward"],
                "instability": result["instability"],
                "trajectory_count": result["trajectory_count"],
                "reward_trajectory": json.dumps(result["reward_trajectory"]),
            })

        if runs_by_model[model_name]:
            weekly_by_model[model_name] = _average_weekly_metrics(
                runs_by_model[model_name]
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Per-run results saved to %s", out_csv)

    if not df.empty:
        summary = df.groupby("model").agg(
            avg_total_cost=("total_cost", "mean"),
            std_total_cost=("total_cost", "std"),
            avg_bullwhip=("bullwhip_overall", "mean"),
            std_bullwhip=("bullwhip_overall", "std"),
            avg_backlog=("avg_backlog", "mean"),
            std_backlog=("avg_backlog", "std"),
            avg_cumulative_reward=("cumulative_reward", "mean"),
            std_cumulative_reward=("cumulative_reward", "std"),
            avg_instability=("instability", "mean"),
        )
        summary_path = out_csv.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path)
        logger.info("Summary saved to %s", summary_path)
        print("\n=== Model Comparison Summary ===")
        print(summary.to_string())

    if plot and weekly_by_model:
        try:
            generate_all_comparison_plots(weekly_by_model)
            logger.info("Comparison plots saved to plots/comparison/")
        except ImportError as exc:
            logger.warning("Skipping plots: %s", exc)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM models and baseline policies"
    )
    parser.add_argument("--weeks", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_comparison.csv",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Stub LLM orders (no Ollama required)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_comparison(
        repeats=args.repeats,
        max_weeks=args.weeks,
        out_csv=args.output,
        ollama_url=args.url,
        offline=args.offline,
        plot=not args.no_plot,
    )

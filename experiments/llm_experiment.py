"""LLM-driven Beer Game experiment runner.

Each supply-chain echelon is controlled by an independent local LLM (Ollama).
No inter-agent communication — decisions use local state only.

Usage:
    python experiments/llm_experiment.py --weeks 30 --model qwen:1.5b
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from agents.llm_agent import LLMAgent
from simulator.beer_game import BeerGame

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ECHELONS = ("Retailer", "Wholesaler", "Distributor", "Factory")


def _log_week(
    week: int,
    actions: Dict[str, int],
    total_cost: float,
    bullwhip: Dict,
) -> None:
    """Print human-readable weekly summary."""
    print(f"\nWeek {week}")
    for name in ECHELONS:
        print(f"{name} ordered: {actions.get(name, 0)}")
    overall = bullwhip.get("overall") if bullwhip else None
    print(f"Total cost: {total_cost:.1f}")
    print(f"Bullwhip: {overall if overall is not None else 'N/A'}")


def run_llm_experiment(
    max_weeks: int = 30,
    model_name: str = "qwen:1.5b",
    ollama_url: str = "http://localhost:11434",
    max_order: int = 10000,
    results_file: str = "results/llm_experiment_results.csv",
    use_tool_recommendation: bool = False,
) -> pd.DataFrame:
    """Run a complete LLM-driven Beer Game experiment."""
    logger.info("Starting LLM experiment with model: %s", model_name)
    logger.info("Max weeks: %s, Max order: %s", max_weeks, max_order)

    env = BeerGame(max_weeks=max_weeks, verbose=False)
    env.reset()

    print(f"Backend : Ollama")
    print(f"Model   : {model_name}")

    agents = {
        name: LLMAgent(
            agent_name=name,
            model_name=model_name,
            ollama_url=ollama_url,
            max_order=max_order,
            temperature=0.2,
            use_tool_recommendation=use_tool_recommendation,
        )
        for name in ECHELONS
    }

    metrics: List[Dict] = []
    done = False

    while not done:
        states = env.get_all_states()
        actions = {
            name: agents[name].generate_order(states[name], fallback=0)
            for name in ECHELONS
        }
        action_metadata = {
            name: dict(agents[name].last_decision_metadata)
            for name in ECHELONS
        }

        _, reward, done, info = env.step(actions, action_metadata=action_metadata)

        bullwhip = info.get("bullwhip") or {}
        _log_week(
            info["week"],
            actions,
            info["total_system_cost"],
            bullwhip,
        )

        metric_row = {
            "week": info["week"],
            "customer_demand": info["customer_demand"],
            "total_system_cost": info["total_system_cost"],
            "reward": reward,
            "bullwhip_overall": bullwhip.get("overall"),
            "bullwhip_retailer": bullwhip.get("Retailer"),
            "bullwhip_wholesaler": bullwhip.get("Wholesaler"),
            "bullwhip_distributor": bullwhip.get("Distributor"),
            "bullwhip_factory": bullwhip.get("Factory"),
        }
        for agent in env.nodes:
            metric_row[f"order_{agent.name}"] = actions[agent.name]
            metric_row[f"inventory_{agent.name}"] = agent.inventory
            metric_row[f"backlog_{agent.name}"] = agent.backlog
            metadata = action_metadata.get(agent.name, {})
            metric_row[f"tool_order_{agent.name}"] = metadata.get("tool_order")
            metric_row[f"llm_order_{agent.name}"] = metadata.get("llm_order")
            metric_row[f"difference_{agent.name}"] = metadata.get("difference")
        metric_row["consensus_gap"] = info.get("consensus_gap")

        metrics.append(metric_row)

    df = pd.DataFrame(metrics)

    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    df.to_csv(results_file, index=False)
    logger.info("Results saved to %s", results_file)

    plot_dir = os.path.join("plots")
    os.makedirs(plot_dir, exist_ok=True)

    if plt is None:
        logger.warning(
            "matplotlib is not installed; skipping plot generation."
        )
    else:
        env.plot_orders_vs_demand(
            save_path=os.path.join(plot_dir, "llm_orders_vs_demand.png")
        )
        env.plot_inventory(
            save_path=os.path.join(plot_dir, "llm_inventory_trajectories.png")
        )
        env.plot_backlog(
            save_path=os.path.join(plot_dir, "llm_backlog_trajectories.png")
        )
        env.plot_inventory_and_backlog(
            save_path=os.path.join(plot_dir, "llm_inventory_backlog.png")
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["week"], df["bullwhip_overall"], label="Overall", marker="o")
        ax.plot(df["week"], df["bullwhip_retailer"], label="Retailer")
        ax.plot(df["week"], df["bullwhip_wholesaler"], label="Wholesaler")
        ax.plot(df["week"], df["bullwhip_distributor"], label="Distributor")
        ax.plot(df["week"], df["bullwhip_factory"], label="Factory")
        ax.set_title("LLM Experiment Bullwhip Metrics")
        ax.set_xlabel("Week")
        ax.set_ylabel("Bullwhip Ratio")
        ax.legend()
        ax.grid(True)
        fig.savefig(
            os.path.join(plot_dir, "llm_bullwhip_metrics.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df["week"],
            df["total_system_cost"].cumsum(),
            marker="o",
            color="tab:red",
        )
        ax.set_title("LLM Experiment Cumulative System Cost")
        ax.set_xlabel("Week")
        ax.set_ylabel("Cumulative Cost")
        ax.grid(True)
        fig.savefig(
            os.path.join(plot_dir, "llm_cumulative_cost.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.info("Plots saved to %s", plot_dir)

    print("\n=== Experiment Summary ===")
    print(f"Total weeks: {len(metrics)}")
    print(f"Average system cost: {df['total_system_cost'].mean():.2f}")
    print(f"Total system cost: {df['total_system_cost'].sum():.2f}")
    if df["bullwhip_overall"].notna().any():
        print(f"Average bullwhip: {df['bullwhip_overall'].mean():.2f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-driven Beer Game experiment"
    )
    parser.add_argument("--weeks", type=int, default=30)
    parser.add_argument("--model", type=str, default="qwen:1.5b")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--max-order", type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_experiment_results.csv",
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "groq", "vllm"),
        default="ollama",
        help="Inference backend to use (ollama or groq)",
    )
    parser.add_argument("--use-tool-recommendation", action="store_true")
    args = parser.parse_args()
    try:
        run_llm_experiment(
            max_weeks=args.weeks,
            model_name=args.model,
            ollama_url=args.url,
            max_order=args.max_order,
            results_file=args.output,
            use_tool_recommendation=args.use_tool_recommendation,
        )
        logger.info("Experiment completed successfully.")
    except Exception as exc:
        logger.error("Experiment failed: %s", exc, exc_info=True)
        sys.exit(1)

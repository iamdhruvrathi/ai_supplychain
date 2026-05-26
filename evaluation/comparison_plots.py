"""Comparative visualization for multi-model Beer Game evaluation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _require_plt():
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with `pip install matplotlib`."
        )


def plot_orders_comparison(
    weekly_by_model: Dict[str, Dict[str, List[float]]],
    save_path: str,
    agent: str = "Retailer",
) -> None:
    """Plot mean orders per week for each model (one echelon)."""
    _require_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, series in weekly_by_model.items():
        key = f"order_{agent}"
        if key not in series:
            continue
        weeks = range(1, len(series[key]) + 1)
        ax.plot(weeks, series[key], label=model, marker="o", markersize=3)

    ax.set_title(f"Orders Comparison — {agent}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Order Quantity")
    ax.legend()
    ax.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_rewards(
    weekly_by_model: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Plot cumulative shaped reward trajectories by model."""
    _require_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, series in weekly_by_model.items():
        rewards = series.get("reward", [])
        if not rewards:
            continue
        cumulative = []
        total = 0.0
        for r in rewards:
            total += r
            cumulative.append(total)
        weeks = range(1, len(cumulative) + 1)
        ax.plot(weeks, cumulative, label=model, marker="o", markersize=3)

    ax.set_title("Cumulative Shaped Reward by Model")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_bullwhip_over_time(
    weekly_by_model: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Plot overall bullwhip ratio over time for each model."""
    _require_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, series in weekly_by_model.items():
        bullwhip = series.get("bullwhip_overall", [])
        if not bullwhip:
            continue
        weeks = range(1, len(bullwhip) + 1)
        ax.plot(weeks, bullwhip, label=model, marker="o", markersize=3)

    ax.set_title("Bullwhip Ratio Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Bullwhip (overall)")
    ax.legend()
    ax.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_inventory_oscillations(
    weekly_by_model: Dict[str, Dict[str, List[float]]],
    save_path: str,
    agent: str = "Retailer",
) -> None:
    """Plot inventory trajectories by model for one echelon."""
    _require_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, series in weekly_by_model.items():
        key = f"inventory_{agent}"
        if key not in series:
            continue
        weeks = range(1, len(series[key]) + 1)
        ax.plot(weeks, series[key], label=model, marker="o", markersize=3)

    ax.set_title(f"Inventory Oscillations — {agent}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Inventory")
    ax.legend()
    ax.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_comparison_plots(
    weekly_by_model: Dict[str, Dict[str, List[float]]],
    plot_dir: str = "plots/comparison",
) -> None:
    """Generate the full comparative plot suite."""
    os.makedirs(plot_dir, exist_ok=True)
    plot_orders_comparison(
        weekly_by_model,
        os.path.join(plot_dir, "orders_qwen_vs_deepseek.png"),
    )
    plot_cumulative_rewards(
        weekly_by_model,
        os.path.join(plot_dir, "cumulative_rewards.png"),
    )
    plot_bullwhip_over_time(
        weekly_by_model,
        os.path.join(plot_dir, "bullwhip_over_time.png"),
    )
    plot_inventory_oscillations(
        weekly_by_model,
        os.path.join(plot_dir, "inventory_oscillations.png"),
    )

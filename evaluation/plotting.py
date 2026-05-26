"""Research-quality plots for repeated-run and agent-bullwhip analysis."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


def _require_plt():
    if plt is None:
        raise ImportError("matplotlib required: pip install matplotlib")


def plot_repeated_run_boxplot(
    costs_by_label: Dict[str, List[float]],
    save_path: str,
) -> None:
    _require_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(costs_by_label.keys())
    data = [costs_by_label[l] for l in labels]
    ax.boxplot(data, labels=labels)
    ax.set_title("Total Cost Distribution Across Repeated Runs")
    ax.set_ylabel("Total Cost")
    ax.grid(True, axis="y")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_agent_bullwhip_heatmap(
    sigma_squared: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Heatmap of σ²(k,t) across echelons and weeks."""
    _require_plt()
    echelons = list(sigma_squared.keys())
    if not echelons:
        return
    matrix = np.array([sigma_squared[e] for e in echelons])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(echelons)))
    ax.set_yticklabels(echelons)
    ax.set_xlabel("Week")
    ax.set_title("Run-to-Run Order Variance σ²(k,t)")
    fig.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_psi_phi_over_time(
    psi: Dict[str, List[Union[float, None]]],
    phi: Dict[str, List[Union[float, None]]],
    save_path: str,
) -> None:
    _require_plt()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for echelon, vals in psi.items():
        weeks = range(1, len(vals) + 1)
        numeric = [v if v is not None else np.nan for v in vals]
        axes[0].plot(weeks, numeric, label=echelon, marker="o", markersize=2)
    axes[0].axhline(1.0, color="gray", linestyle="--", label="Ψ=1")
    axes[0].set_title("Cross-Echelon Amplification Ψ_k(t)")
    axes[0].legend()
    axes[0].grid(True)

    for echelon, vals in phi.items():
        weeks = range(1, len(vals) + 1)
        numeric = [v if v is not None else np.nan for v in vals]
        axes[1].plot(weeks, numeric, label=echelon, marker="o", markersize=2)
    axes[1].axhline(1.0, color="gray", linestyle="--", label="Φ=1")
    axes[1].set_title("Intertemporal Amplification Φ_k(t)")
    axes[1].set_xlabel("Week")
    axes[1].legend()
    axes[1].grid(True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_research_plots(
    report: Dict,
    plot_dir: str = "plots/research",
) -> None:
    """Generate standard research plot suite from repeated_runs report."""
    os.makedirs(plot_dir, exist_ok=True)
    costs = report.get("total_costs", [])
    if costs:
        plot_repeated_run_boxplot(
            {"policy": costs},
            os.path.join(plot_dir, "cost_boxplot.png"),
        )

    ab = report.get("agent_bullwhip", {})
    sigma = ab.get("sigma_squared", {})
    if sigma:
        plot_agent_bullwhip_heatmap(
            sigma,
            os.path.join(plot_dir, "agent_bullwhip_heatmap.png"),
        )
    psi = ab.get("psi", {})
    phi = ab.get("phi", {})
    if psi or phi:
        plot_psi_phi_over_time(
            psi,
            phi,
            os.path.join(plot_dir, "psi_phi_over_time.png"),
        )

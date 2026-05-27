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


def generate_bullwhip_boxplots(results_dir: str, output_dir: str) -> None:
    """Generate Figure 2-style order boxplots from rollout JSONL files."""
    _require_plt()
    import json
    from pathlib import Path

    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rollouts = list(results_path.rglob("*rollouts*.jsonl"))
    if not rollouts:
        rollouts = list(results_path.rglob("*.jsonl"))
    if not rollouts:
        raise FileNotFoundError(f"No trajectory JSONL files found under {results_dir}")

    echelons = ("Retailer", "Wholesaler", "Distributor", "Factory")
    orders = {e: {} for e in echelons}

    for path in rollouts:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                agent = rec.get("agent") or rec.get("agent_role") or rec.get("role")
                action = rec.get("action")
                if "week" in rec:
                    week = rec.get("week")
                elif "t" in rec:
                    week = rec.get("t")
                else:
                    week = rec.get("time")

                if agent is None or action is None or week is None:
                    continue

                agent_name = str(agent).strip().capitalize()
                if agent_name not in echelons:
                    short = {
                        "r": "Retailer",
                        "w": "Wholesaler",
                        "d": "Distributor",
                        "f": "Factory",
                    }
                    agent_name = short.get(str(agent).strip().lower(), agent_name)
                if agent_name not in echelons:
                    continue

                try:
                    week_num = int(float(week))
                    order_qty = int(float(action))
                except (TypeError, ValueError):
                    continue

                orders[agent_name].setdefault(week_num, []).append(order_qty)

    raw_weeks = sorted({w for by_week in orders.values() for w in by_week})
    if not raw_weeks:
        raise RuntimeError("No order data found in rollout files")

    display_offset = 1 if min(raw_weeks) == 0 else 0
    positions = list(range(1, len(raw_weeks) + 1))
    labels = [str(w + display_offset) for w in raw_weeks]

    fig, axes = plt.subplots(len(echelons), 1, figsize=(12, 3 * len(echelons)), sharex=True)

    for idx, echelon in enumerate(echelons):
        ax = axes[idx]
        data = [orders[echelon].get(week, []) for week in raw_weeks]
        data_for_plot = [values if values else [np.nan] for values in data]
        ax.boxplot(
            data_for_plot,
            positions=positions,
            widths=0.6,
            showfliers=True,
            patch_artist=True,
        )
        ax.set_ylabel(f"{echelon}\nOrder Qty")
        ax.grid(True, axis="y")
        if idx == 0:
            ax.set_title("Figure 2: Order Quantity Distributions by Week and Echelon")

    axes[-1].set_xlabel("Week")
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()

    png_path = out_path / "figure2_bullwhip_boxplots.png"
    pdf_path = out_path / "figure2_bullwhip_boxplots.pdf"
    fig.savefig(str(png_path), bbox_inches="tight", dpi=200)
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

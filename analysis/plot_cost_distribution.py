#!/usr/bin/env python3
"""Generate cost distribution boxplots for the four experiment conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.plotting import plot_repeated_run_boxplot

EXPERIMENTS = [
    ("Baseline", "results/exp1_baseline/repeated_runs_report.json"),
    ("Tool", "results/exp2_tool/repeated_runs_report.json"),
    ("Negotiation", "results/exp3_negotiation/repeated_runs_report.json"),
    ("Tool+Negotiation", "results/exp4_tool_negotiation/repeated_runs_report.json"),
]

OUTPUT_PATH = REPO_ROOT / "plots" / "research" / "cost_distribution.png"


def load_total_costs(report_path: Path) -> list[float]:
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    costs = report.get("total_costs")
    if costs is None:
        raise ValueError(f"Missing total_costs in {report_path}")
    return [float(c) for c in costs]


def main() -> None:
    costs_by_label: dict[str, list[float]] = {}
    for label, rel_path in EXPERIMENTS:
        report_path = REPO_ROOT / rel_path
        costs_by_label[label] = load_total_costs(report_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_repeated_run_boxplot(costs_by_label, str(OUTPUT_PATH))
    print(f"Saved cost distribution figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

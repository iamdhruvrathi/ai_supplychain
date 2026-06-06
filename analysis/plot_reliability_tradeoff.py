#!/usr/bin/env python3
"""Create a reliability vs performance tradeoff scatter plot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("Baseline", REPO_ROOT / "results/exp1_baseline/repeated_runs_report.json"),
    ("Tool", REPO_ROOT / "results/exp2_tool/repeated_runs_report.json"),
    ("Negotiation", REPO_ROOT / "results/exp3_negotiation/repeated_runs_report.json"),
    ("Tool+Negotiation", REPO_ROOT / "results/exp4_tool_negotiation/repeated_runs_report.json"),
]
OUTPUT_PATH = REPO_ROOT / "plots" / "research" / "reliability_tradeoff.png"


def load_metrics(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    cost = float(report["cost"]["mean"])
    std = float(report["cost"]["std"])
    if cost == 0:
        cv = 0.0
    else:
        cv = std / cost
    return cost, cv


def main() -> None:
    labels = []
    costs = []
    cvs = []
    for label, path in EXPERIMENTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing report file: {path}")
        cost, cv = load_metrics(path)
        labels.append(label)
        costs.append(cost)
        cvs.append(cv)

    plt.figure(figsize=(10, 6))
    plt.scatter(cvs, costs, s=100, color="#4C78A8")
    for label, x, y in zip(labels, cvs, costs):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")

    plt.title("Reliability vs Performance Tradeoff")
    plt.xlabel("Coefficient of Variation (CV)")
    plt.ylabel("Mean Total Cost")
    plt.grid(True, axis="both", linestyle="--", alpha=0.4)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_PATH), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved reliability tradeoff figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

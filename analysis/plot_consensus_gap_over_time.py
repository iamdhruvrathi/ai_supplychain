#!/usr/bin/env python3
"""Plot mean consensus gap over time for all four experimental conditions."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import isnan
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("Baseline", REPO_ROOT / "results/exp1_baseline/trajectories/rollouts.jsonl"),
    ("Tool", REPO_ROOT / "results/exp2_tool/trajectories/rollouts.jsonl"),
    ("Negotiation", REPO_ROOT / "results/exp3_negotiation/trajectories/rollouts.jsonl"),
    ("Tool+Negotiation", REPO_ROOT / "results/exp4_tool_negotiation/trajectories/rollouts.jsonl"),
]
OUTPUT_PATH = REPO_ROOT / "plots" / "research" / "consensus_gap_over_time.png"


def load_consensus_gap_by_week(path: Path) -> dict[int, list[float]]:
    gap_by_week: dict[int, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            week = record.get("week")
            gap = record.get("consensus_gap")
            if gap is None:
                continue
            try:
                week_num = int(week)
                gap_value = float(gap)
            except (TypeError, ValueError):
                continue
            gap_by_week[week_num].append(gap_value)
    return gap_by_week


def compute_summary(gaps: dict[int, list[float]]) -> tuple[list[int], list[float], list[float], list[float]]:
    weeks = sorted(gaps)
    means = []
    lower = []
    upper = []
    for week in weeks:
        values = [v for v in gaps[week] if not isnan(v)]
        if values:
            mean = float(sum(values)) / len(values)
            std = float(np.std(values, ddof=0))
        else:
            mean = float("nan")
            std = 0.0
        means.append(mean)
        lower.append(mean - std)
        upper.append(mean + std)
    return weeks, means, lower, upper


def main() -> None:
    plt.figure(figsize=(10, 6))
    for label, path in EXPERIMENTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory file: {path}")
        gap_by_week = load_consensus_gap_by_week(path)
        weeks, means, lower, upper = compute_summary(gap_by_week)
        plt.plot(weeks, means, marker="o", label=label)
        plt.fill_between(weeks, lower, upper, alpha=0.15)

    plt.title("Consensus Gap Over Time")
    plt.xlabel("Week")
    plt.ylabel("Consensus Gap")
    plt.legend()
    plt.grid(True, axis="both", linestyle="--", alpha=0.4)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_PATH), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved consensus gap over time figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

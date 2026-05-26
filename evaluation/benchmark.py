"""Unified benchmark entry point (paper replication)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.loader import load_full_experiment
from evaluation.repeated_runs import run_repeated_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_from_config(config_path: str) -> dict:
    loaded = load_full_experiment(config_path)
    config = loaded["simulation"]
    eval_cfg = loaded["evaluation"]
    agents = loaded["agents"]

    model_name = None
    if agents:
        models = set(agents.values())
        if len(models) == 1 and "base_stock" not in models:
            model_name = models.pop()

    n_runs = int(loaded["raw"].get("experiment", {}).get("runs", 30))
    save_dir = loaded["output"].get("results_dir", "results/repeated_runs")

    return run_repeated_experiment(
        config=config,
        n_runs=n_runs,
        model_name=model_name,
        ollama_url=eval_cfg.get("ollama_url", "http://localhost:11434"),
        save_dir=save_dir,
        export_trajectories=eval_cfg.get("export_trajectories", True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper-aligned benchmark from YAML")
    parser.add_argument(
        "--config",
        default="configs/default_experiment.yaml",
    )
    args = parser.parse_args()
    report = run_from_config(args.config)
    print(f"Completed {report['n_runs']} runs. Mean cost: {report['cost']['mean']:.2f}")

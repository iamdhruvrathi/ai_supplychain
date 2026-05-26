#!/usr/bin/env python
"""
Beer Game LLM Experiment - Main Entry Point

This script provides a simple CLI to run experiments and tests.

Usage:
    python main.py test-llm          # Run LLM agent tests (no Ollama needed)
    python main.py test-state        # Run state API tests
    python main.py baseline          # Run baseline policy experiments
    python main.py llm [--weeks N]   # Run LLM experiments (requires Ollama)
    python main.py compare           # Compare models (LLM + baselines)
    python main.py repeated          # Paper-aligned repeated-run analysis
    python main.py benchmark         # Run experiment from YAML config
    python main.py demo              # Quick demo simulation
    python main.py help              # Show this help message
"""

import sys
import os
import subprocess
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_test_llm():
    """Run LLM agent unit tests."""
    print("=" * 60)
    print("Running LLM Agent Tests")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "experiments/test_llm_agent.py"],
        cwd=project_root
    )
    return result.returncode


def run_test_state():
    """Run state API tests."""
    print("=" * 60)
    print("Running State API Tests")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "experiments/test_state_api.py"],
        cwd=project_root
    )
    return result.returncode


def run_baseline():
    """Run baseline policy experiments."""
    print("=" * 60)
    print("Running Baseline Policy Experiments")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "experiments/baseline_experiment.py"],
        cwd=project_root
    )
    return result.returncode


def run_repeated(weeks: int = 30, runs: int = 10, offline: bool = False):
    """Run repeated-run reliability experiment (paper Section 4)."""
    print("=" * 60)
    print("Running Repeated-Run Experiment")
    print("=" * 60)
    cmd = [
        sys.executable,
        "evaluation/repeated_runs.py",
        "--weeks", str(weeks),
        "--runs", str(runs),
    ]
    if offline:
        cmd.append("--offline")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_benchmark(config: str = "configs/default_experiment.yaml"):
    """Run benchmark from YAML configuration."""
    print("=" * 60)
    print("Running YAML Benchmark")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "evaluation/benchmark.py", "--config", config],
        cwd=project_root,
    )
    return result.returncode


def run_compare(weeks: int = 30, repeats: int = 10, offline: bool = False):
    """Run multi-model comparison evaluation."""
    print("=" * 60)
    print("Running Model Comparison")
    print("=" * 60)
    cmd = [
        sys.executable,
        "evaluation/compare_models.py",
        "--weeks", str(weeks),
        "--repeats", str(repeats),
    ]
    if offline:
        cmd.append("--offline")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_llm_experiment(weeks: int = 30, model: str = "qwen:1.5b"):
    """Run LLM-driven experiment."""
    print("=" * 60)
    print("Running LLM Experiment")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "experiments/llm_experiment.py", 
         "--weeks", str(weeks), "--model", model],
        cwd=project_root
    )
    return result.returncode


def run_demo():
    """Quick demo simulation."""
    print("=" * 60)
    print("Beer Game Simulator - Quick Demo")
    print("=" * 60)

    from simulator.beer_game import BeerGame
    from policies.base_stock import base_stock_order
    import random

    print("\nInitializing Beer Game (10 weeks, base-stock policy)...")

    env = BeerGame(max_weeks=10, verbose=False)
    env.reset()

    random.seed(42)

    done = False
    week = 0

    while not done:
        week += 1

        # Use base-stock policy for all nodes
        actions = {}

        for node in env.nodes:
            state = env.get_state_dict(node.name)
            action = base_stock_order(state, target_inventory=20)
            actions[node.name] = action

        _, _, done, info = env.step(actions)

        # Safe bullwhip extraction
        bullwhip_data = info.get("bullwhip")

        if isinstance(bullwhip_data, dict):
            overall_bullwhip = bullwhip_data.get("overall", 0.0)
        elif bullwhip_data is None:
            overall_bullwhip = 0.0
        else:
            overall_bullwhip = bullwhip_data

        print(
            f"Week {info['week']}: "
            f"Demand={info['customer_demand']:.0f}, "
            f"Cost=${info['total_system_cost']:.2f}, "
            f"Bullwhip={overall_bullwhip:.3f}"
        )

    print("\n✓ Demo completed!")
    print(f"  Total system cost: ${info['total_system_cost']:.2f}")
    print(f"  Final bullwhip: {overall_bullwhip:.3f}")

    return 0


def show_help():
    """Show help message."""
    print(__doc__)
    print("\nAvailable Commands:")
    print("  test-llm        Run LLM agent unit tests (no Ollama needed)")
    print("  test-state      Run state API validation tests")
    print("  baseline        Run classical policy baselines")
    print("  llm [--weeks N] Run LLM-driven experiment (requires Ollama running)")
    print("  compare         Compare qwen / deepseek / base-stock (10 runs each)")
    print("  repeated        Repeated-run reliability + agent bullwhip (paper Sec 4)")
    print("  benchmark       Run experiment from configs/default_experiment.yaml")
    print("  demo            Quick demo with base-stock policy")
    print("  help            Show this help message")
    print("\nExamples:")
    print("  python main.py test-llm")
    print("  python main.py llm --weeks 50")
    print("  python main.py demo")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        show_help()
        return 0
    
    command = sys.argv[1].lower()
    
    if command == "test-llm":
        return run_test_llm()
    elif command == "test-state":
        return run_test_state()
    elif command == "baseline":
        return run_baseline()
    elif command == "llm":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--weeks", type=int, default=30)
        parser.add_argument("--model", type=str, default="qwen:1.5b")
        args, _ = parser.parse_known_args(sys.argv[2:])
        return run_llm_experiment(weeks=args.weeks, model=args.model)
    elif command == "repeated":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--weeks", type=int, default=30)
        parser.add_argument("--runs", type=int, default=10)
        parser.add_argument("--offline", action="store_true")
        args, _ = parser.parse_known_args(sys.argv[2:])
        return run_repeated(weeks=args.weeks, runs=args.runs, offline=args.offline)
    elif command == "benchmark":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default="configs/default_experiment.yaml")
        args, _ = parser.parse_known_args(sys.argv[2:])
        return run_benchmark(config=args.config)
    elif command == "compare":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--weeks", type=int, default=30)
        parser.add_argument("--repeats", type=int, default=10)
        parser.add_argument("--offline", action="store_true")
        args, _ = parser.parse_known_args(sys.argv[2:])
        return run_compare(
            weeks=args.weeks,
            repeats=args.repeats,
            offline=args.offline,
        )
    elif command == "demo":
        return run_demo()
    elif command == "help":
        show_help()
        return 0
    else:
        print(f"Unknown command: {command}")
        print("Use 'python main.py help' for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

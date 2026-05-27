"""Run repeated Beer Game experiments with majority-vote LLM decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run majority-vote repeated runs")
    parser.add_argument("--weeks", type=int, default=25)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--progress", choices=("none", "run", "week"), default="run")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=8)
    args = parser.parse_args(argv)

    from evaluation.repeated_runs import run_repeated_experiment
    from simulator.config import SimulationConfig
    from simulator.demand import mit_beer_game_demand_path

    config = SimulationConfig(
        max_weeks=args.weeks,
        fixed_demand_path=mit_beer_game_demand_path(args.weeks),
    )

    report = run_repeated_experiment(
        config=config,
        n_runs=args.runs,
        model_name=None if args.offline else args.model,
        save_dir=args.output_dir,
        timeout=args.timeout,
        temperature=args.temperature,
        num_predict=args.num_predict,
        progress=args.progress,
        n_samples=max(1, args.n_samples),
    )

    print(
        f"Completed {report['n_runs']} run(s), "
        f"n_samples={args.n_samples}, mean_cost={report['cost']['mean']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

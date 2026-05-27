"""Generate Figure 2-style bullwhip boxplots from repeated-run results.

Usage:
    python experiments/run_figure2.py --results results/qwen25_debug --output plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate Figure 2 boxplots")
    parser.add_argument("--results", type=str, default="results/repeated_runs")
    parser.add_argument("--output", type=str, default="plots/")
    args = parser.parse_args(argv)

    try:
        from evaluation.plotting import generate_bullwhip_boxplots

        generate_bullwhip_boxplots(args.results, args.output)
    except Exception as exc:
        print(f"Failed to generate Figure 2: {exc}", file=sys.stderr)
        return 1

    print(f"Figure 2 saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

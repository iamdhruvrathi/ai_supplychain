"""Generate Figure 3-style plots from two majority-vote result folders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate Figure 3 majority-vote boxplots")
    parser.add_argument("--results-10", type=str, required=True)
    parser.add_argument("--results-100", type=str, required=True)
    parser.add_argument("--output", type=str, default="plots/")
    args = parser.parse_args(argv)

    try:
        from evaluation.plotting import generate_figure3_boxplots

        generate_figure3_boxplots(args.results_10, args.results_100, args.output)
    except Exception as exc:
        print(f"Failed to generate Figure 3: {exc}", file=sys.stderr)
        return 1

    print(f"Figure 3 saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

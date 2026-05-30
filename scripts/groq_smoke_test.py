"""Standalone Groq smoke-test script.

Usage:
  python scripts/groq_smoke_test.py --model qwen/qwen3-4b --prompt "Hello"

This script requires the `groq` Python SDK and `GROQ_API_KEY` in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from groq import Groq
except Exception:
    Groq = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Groq smoke test")
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--prompt", type=str, default="Say hi")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args(argv)

    if Groq is None:
        print("groq SDK not installed. Install with: pip install groq")
        return 2

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set in environment. Export it and retry.")
        return 2

    try:
        client = Groq(api_key=api_key)
    except Exception as exc:
        print(f"Failed to initialize Groq client: {exc}")
        return 2

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("Raw response:")
        print(resp)
        return 0
    except Exception as exc:
        print(f"Groq API call failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

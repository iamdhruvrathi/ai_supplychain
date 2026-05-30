"""Smoke test for GroqBackend post-processing.

Sends the prompt: "Respond only with the number 42." and verifies the post-processed
response equals exactly "42".
"""

from __future__ import annotations

import argparse
import os
import sys

from agents.llm_backends import GroqBackend


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GroqBackend post-process smoke test")
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--prompt", type=str, default="Respond only with the number 42.")
    args = parser.parse_args(argv)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set in environment. Set it and re-run.")
        return 2

    backend = GroqBackend(model_name=args.model, api_key=api_key)
    resp = backend.generate(args.prompt, max_tokens=16, temperature=0.0)
    print("Post-processed response:", repr(resp))
    if resp == "42":
        print("Smoke test passed: output is exactly '42'")
        return 0
    else:
        print("Smoke test FAILED: expected '42'")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

GROQ + Qwen3-4B Setup

This document explains how to run the Beer Game experiments using Groq-hosted Qwen3-4B.

Prerequisites

- Python environment with the project's `requirements.txt` installed.
- A valid Groq API key with access to `qwen/qwen3-4b`.

Environment

- Set the API key in the environment:

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "your_groq_api_key"
```

Optional: set a custom API URL with `GROQ_API_URL`.

Supported models

- `qwen/qwen3-4b` (recommended for Groq runs)
- Ollama local models (unchanged): e.g. `qwen2.5:1.5b`, `qwen:1.5b`.

How it works

- Add `--backend groq` and `--model qwen/qwen3-4b` to your existing commands.
- The code now supports two backends:
  - `ollama` (default): local Ollama server at `--url`.
  - `groq`: remote Groq API via `GROQ_API_KEY`.

Example: Figure 2 (repeated runs)

```powershell
python evaluation/repeated_runs.py \
  --backend groq \
  --model qwen/qwen3-4b \
  --weeks 20 \
  --runs 30 \
  --output-dir results/qwen3_groq_fig2
```

Example: Figure 3 majority vote (N=10)

```powershell
python experiments/run_majority_vote.py \
  --backend groq \
  --model qwen/qwen3-4b \
  --weeks 20 \
  --runs 10 \
  --n-samples 10 \
  --output-dir results/qwen3_groq_n10
```

Troubleshooting

- If you see `GROQ_API_KEY not found` ensure the environment variable is set in the same shell where you run Python.
- Network/timeouts: increase `--timeout` on the CLI if requests are timing out.
- For debugging, set `LOGLEVEL=DEBUG` and run with Python to get more details.

Notes

- The Groq backend uses a permissive JSON parsing strategy: it tries several common response shapes and falls back to raw text.
- Retries and timeouts are implemented in the Groq client (via requests + urllib3 Retry).

If you want help validating a run, I can run a quick smoke test command for you (you must have `GROQ_API_KEY` set).

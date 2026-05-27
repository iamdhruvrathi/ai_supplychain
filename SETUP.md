# Setup For Qwen2.5

This project is currently configured for:

```text
qwen2.5:1.5b
```

Qwen3 was removed from the recommended path because its reasoning output is slow for repeated simulations.

## Install

```powershell
cd D:\GitHub\ai_supplychain
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ollama

```powershell
ollama serve
ollama pull qwen2.5:1.5b
ollama list
```

## Verify Python

```powershell
python main.py test-llm
python main.py test-state
```

## Verify Pipeline Without Ollama

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --offline --progress week --output-dir results/offline_debug
python experiments/run_figure2.py --results results/offline_debug --output plots/
```

## Verify Qwen2.5

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --model qwen2.5:1.5b --progress week --output-dir results/qwen25_debug
python experiments/run_figure2.py --results results/qwen25_debug --output plots/
```

## Faster Development Settings

```powershell
# Tiny
python evaluation/repeated_runs.py --weeks 5 --runs 2 --model qwen2.5:1.5b --progress week --output-dir results/qwen25_debug

# Medium
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs

# Full
python evaluation/repeated_runs.py --weeks 30 --runs 30 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_30runs
```

## GPU

Python does not directly use CUDA. Ollama uses the GPU if your install supports it.

Check while a model is running:

```powershell
nvidia-smi
```

## If Output Looks Flat

Offline runs are deterministic, so the boxes may be flat. Use `--model qwen2.5:1.5b` to see LLM decision variability.

# Setup

Use this guide only to get the project running. For the research logic, read `docs/REPLICATION_PLAN.md`.

## 1. Python Environment

From the repo root:

```powershell
cd D:\GitHub\ai_supplychain
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check that the main packages import:

```powershell
python -c "import pandas, matplotlib, requests; print('OK')"
```

## 2. No-LLM Smoke Test

Run this first:

```powershell
python experiments/run_smoke_tests.py
```

This should create or update:

```text
results/qwen25_mit_10runs/repeated_runs_report.json
results/qwen25_mit_10runs/run_costs.csv
results/qwen25_mit_10runs/trajectories/rollouts.jsonl
plots/figure2_bullwhip_boxplots.png
results/table1.csv
```

This test uses a non-LLM policy. It proves the pipeline works, but it does not reproduce the paper's LLM variability.

## 3. Ollama Setup For LLM Experiments

Install Ollama from:

```text
https://ollama.com
```

Start the server:

```powershell
ollama serve
```

Pull a small model:

```powershell
ollama pull qwen2.5:1.5b
```

Optional second model:

```powershell
ollama pull deepseek-r1:1.5b
```

Check Ollama is reachable:

```powershell
curl http://localhost:11434/api/tags
```

## 4. Short LLM Trial

Before doing 30 full runs, try something small:

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 3 --model qwen2.5:1.5b
python experiments/run_figure2.py --results results/repeated_runs --output plots/
```

Then open:

```text
plots/figure2_bullwhip_boxplots.png
```

## 5. Full Figure 2-Style Run

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

Outputs:

```text
results/qwen25_mit_10runs/repeated_runs_report.json
results/qwen25_mit_10runs/trajectories/rollouts.jsonl
plots/figure2_bullwhip_boxplots.png
plots/figure2_bullwhip_boxplots.pdf
```

## 6. Troubleshooting

### Python command is not found

Try:

```powershell
py -m venv venv
```

or use the full path to your Python executable.

### Ollama connection fails

Make sure this is running in another terminal:

```powershell
ollama serve
```

Then test:

```powershell
curl http://localhost:11434/api/tags
```

### Model is not found

Pull it:

```powershell
ollama pull qwen2.5:1.5b
```

### LLM runs are slow

Use fewer weeks and runs while learning:

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 3 --model qwen2.5:1.5b
```

Full 30-run experiments can take a long time because each week makes four model calls.

### Box plots are flat

If you used `--offline`, flat plots are expected. Offline/base-stock behavior is mostly deterministic. Use `--model qwen2.5:1.5b` to study LLM run-to-run variability.s

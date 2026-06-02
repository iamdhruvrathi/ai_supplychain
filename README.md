# AI Supply Chain Beer Game Replication

This repository implements a local Beer Game replication framework with offline classical baselines, local LLM agents, repeated-run reliability experiments, and plot generation.

The current codebase is built around a locally served Ollama backend. The recommended model for repeated experiments is `qwen2.5:1.5b`, but some scripts default to `qwen:1.5b`.

## Quick Setup

```powershell
cd D:\GitHub\ai_supplychain
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start Ollama

```powershell
ollama serve
ollama pull qwen2.5:1.5b
```

## Verify the codebase

```powershell
python main.py test-llm
python main.py test-state
python experiments/smoke_test.py
python main.py demo
```

## Offline pipeline check

Use this to verify the simulator and output generation without Ollama.

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --offline --progress week --output-dir results/offline_debug
python experiments/run_figure2.py --results results/offline_debug --output plots/
```

## Recommended LLM experiment flow

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

## Majority-vote experiment flow

```powershell
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 10 --output-dir results/fig3_qwen25_n10_debug --progress run
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 100 --output-dir results/fig3_qwen25_n100_debug --progress run
python experiments/run_figure3.py --results-10 results/fig3_qwen25_n10_debug --results-100 results/fig3_qwen25_n100_debug --output plots/
```

## Core CLI wrappers

The repository ships `main.py` for convenience:

```powershell
python main.py test-llm
python main.py test-state
python main.py baseline
python main.py llm --weeks 10 --model qwen2.5:1.5b
python main.py repeated --weeks 10 --runs 3 --model qwen2.5:1.5b
python main.py benchmark --config configs/default_experiment.yaml
python main.py compare --weeks 10 --repeats 3
python main.py demo
```

## Important notes

- `experiments/smoke_test.py` is the minimal smoke test script. There is no `experiments/run_smoke_tests.py` file.
- The repository does not include `experiments/run_table1.py`.
- `evaluation/compare_models.py --offline` runs stub LLM order generation rather than real Ollama inference.
- Some experiments can be run with the `--backend groq` option, but core validation is centered on Ollama.
- `plots/` contains generated figures, and `results/` holds experiment outputs.

## Folder map

| Folder         | Purpose                                             |
| -------------- | --------------------------------------------------- |
| `simulator/`   | Beer Game environment and state progression         |
| `agents/`      | LLM agent wrappers and backend adapters             |
| `policies/`    | Classical policies and baseline order rules         |
| `evaluation/`  | Repeated-run analysis, comparisons, plotting        |
| `experiments/` | Script wrappers for experiments and validation      |
| `metrics/`     | Bullwhip, stability, reliability, and cost metrics   |
| `configs/`     | Experiment YAML settings and loader utilities       |
| `results/`     | Generated experiment output                          |
| `plots/`       | Generated figure images                             |
| `docs/`        | Replication notes and backend setup documentation    |

## Project goal

The repository is designed to explore agent bullwhip and repeated-run variability in a Beer Game environment. The immediate replication target is the repeated-run artifact shape used for Figure 2.

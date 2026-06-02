# Setup For Qwen2.5

This project is configured around local Ollama inference. The recommended model for repeated LLM experiments is `qwen2.5:1.5b`.

## Install

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

If you want to use Groq, set up the Groq backend separately and pass `--backend groq` to applicable scripts.

## Verify Python environment

```powershell
python -c "import pandas, matplotlib, requests, numpy, yaml; print('OK')"
```

## Basic validation

```powershell
python main.py test-llm
python main.py test-state
python experiments/smoke_test.py
```

## Offline pipeline validation

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --offline --progress week --output-dir results/offline_debug
python experiments/run_figure2.py --results results/offline_debug --output plots/
```

## Small Qwen2.5 trial

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --model qwen2.5:1.5b --progress week --output-dir results/qwen25_debug
python experiments/run_figure2.py --results results/qwen25_debug --output plots/
```

## Full repeated-run experiment

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

## Majority-vote debugging

```powershell
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 10 --output-dir results/fig3_qwen25_n10_debug --progress run
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 100 --output-dir results/fig3_qwen25_n100_debug --progress run
python experiments/run_figure3.py --results-10 results/fig3_qwen25_n10_debug --results-100 results/fig3_qwen25_n100_debug --output plots/
```

## Important notes

- `experiments/smoke_test.py` is the repository's smoke test entry point. There is no `experiments/run_smoke_tests.py` file.
- `evaluation/compare_models.py --offline` uses a stub order function for LLM models.
- There is no training pipeline, no RL training wrapper in `train/`, and no dedicated Gymnasium environment.

## Recommended commands via `main.py`

```powershell
python main.py test-llm
python main.py test-state
python main.py baseline
python main.py llm --weeks 30 --model qwen:1.5b
python main.py repeated --weeks 30 --runs 10 --model qwen2.5:1.5b
python main.py compare --weeks 10 --repeats 3
python main.py benchmark --config configs/default_experiment.yaml
python main.py demo
```

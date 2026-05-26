# Setup Guide

This guide covers environment setup, Ollama configuration, running experiments, and verifying outputs. For system design, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Requirements

| Component      | Version / notes                                                    |
| -------------- | ------------------------------------------------------------------ |
| Python         | 3.10+ recommended                                                  |
| pip            | Latest stable                                                      |
| Ollama         | Required for live LLM experiments ([ollama.ai](https://ollama.ai)) |
| GPU (optional) | Accelerates Ollama inference when supported by your hardware       |

Python dependencies (`requirements.txt`):

```
pandas
matplotlib
requests
```

---

## 1. Python Environment

### Windows (PowerShell)

```powershell
cd D:\GitHub\ai_supplychain
python -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd ai_supplychain
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Python packages

```powershell
python -c "import pandas, matplotlib, requests; print('OK')"
```

---

## 2. Ollama Installation

Ollama runs LLM inference locally and exposes an HTTP API consumed by `agents/llm_agent.py`.

1. Download and install from [https://ollama.ai](https://ollama.ai).
2. Start the server (keep this terminal open):

```powershell
ollama serve
```

3. Confirm the API is reachable:

```powershell
curl http://localhost:11434/api/tags
```

You should receive a JSON list of installed models (possibly empty before pulling).

---

## 3. Supported Models

The framework uses any model available in your local Ollama library. Documented benchmarks:

| Model tag          | Type          | Notes                                                    |
| ------------------ | ------------- | -------------------------------------------------------- |
| `qwen2.5:1.5b`     | Non-reasoning | Default in model comparison; fast, direct answers        |
| `deepseek-r1:1.5b` | Reasoning     | May emit chain-of-thought; parser extracts final integer |
| `qwen:1.5b`        | Non-reasoning | Default in `llm_experiment.py` and `main.py llm`         |

Pull models before running experiments:

```powershell
ollama pull qwen2.5:1.5b
ollama pull deepseek-r1:1.5b
ollama pull qwen:1.5b
```

List installed models:

```powershell
ollama list
```

Use a custom model in experiments:

```powershell
python experiments/llm_experiment.py --model phi3:mini --weeks 20
```

---

## 4. CUDA / GPU Usage

This project does **not** implement custom CUDA kernels. GPU acceleration is handled entirely by **Ollama**:

- On NVIDIA systems, Ollama uses GPU layers when drivers and CUDA are properly installed.
- On Apple Silicon, Ollama uses Metal automatically.
- On CPU-only machines, models run on CPU (slower but functional).

**Check GPU usage while a model runs:**

```powershell
# Windows Task Manager → Performance → GPU
# Or NVIDIA:
nvidia-smi
```

**If inference is slow:**

- Use smaller models (`1.5b` / `3b` class).
- Reduce simulation length: `--weeks 10`.
- Ensure no other heavy GPU processes are running.

**Timeout:** `LLMAgent` defaults to `timeout=120` seconds per request. Reasoning models (DeepSeek-R1) may need the full window on CPU.

---

## 5. Running Experiments

### 5.1 Validation (no Ollama)

```powershell
python main.py test-llm      # parsing, trajectories, reward shaping
python main.py test-state    # state API
python experiments/smoke_test.py
```

### 5.2 Quick demo (base-stock, no Ollama)

```powershell
python main.py demo
```

### 5.3 Classical baselines

```powershell
python main.py baseline
# or
python experiments/baseline_experiment.py
```

**Output:** `results/baseline_results.csv`

### 5.4 Single LLM experiment

Requires `ollama serve` and a pulled model.

```powershell
python experiments/llm_experiment.py --weeks 30 --model qwen2.5:1.5b
```

| Flag          | Default                              | Description                   |
| ------------- | ------------------------------------ | ----------------------------- |
| `--weeks`     | 30                                   | Simulation horizon            |
| `--model`     | `qwen:1.5b`                          | Ollama model tag              |
| `--url`       | `http://localhost:11434`             | Ollama base URL               |
| `--max-order` | 100                                  | Upper bound on order quantity |
| `--output`    | `results/llm_experiment_results.csv` | Results CSV                   |

**Outputs:**

- `results/llm_experiment_results.csv`
- `plots/llm_*.png` (if matplotlib is installed)

### 5.5 Repeated-run reliability (paper Section 4)

Runs R episodes with **identical demand path** (`demand_seed`) to measure agent bullwhip and CV:

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 30 --demand-seed 42 --offline
python main.py repeated --runs 30 --offline
```

Outputs: `results/repeated_runs/repeated_runs_report.json`, `run_costs.csv`, `trajectories/rollouts.jsonl`

### 5.6 YAML benchmark

```powershell
pip install pyyaml numpy
python evaluation/benchmark.py --config configs/default_experiment.yaml
python main.py benchmark
```

### 5.7 Multi-model comparison

```powershell
python evaluation/compare_models.py --weeks 30 --repeats 10
# or
python main.py compare
```

**Offline mode** (stubs LLM orders; no Ollama):

```powershell
python evaluation/compare_models.py --offline --repeats 2 --weeks 5
```

**Outputs:**

| File                                   | Description                                |
| -------------------------------------- | ------------------------------------------ |
| `results/model_comparison.csv`         | Per-run metrics + JSON reward trajectories |
| `results/model_comparison_summary.csv` | Mean / std by model                        |
| `plots/comparison/*.png`               | Comparative visualizations                 |

---

## 6. Verifying Results

### CSV files

```powershell
# LLM single run
Import-Csv results\llm_experiment_results.csv | Select-Object -First 5

# Model comparison summary
Import-Csv results\model_comparison_summary.csv
```

Expected columns in `llm_experiment_results.csv`:

- `week`, `customer_demand`, `total_system_cost`, `reward`
- `bullwhip_overall`, per-echelon bullwhip
- `order_*`, `inventory_*`, `backlog_*` for each echelon

### Plots

After an LLM experiment with matplotlib installed:

```
plots/
  llm_orders_vs_demand.png
  llm_inventory_trajectories.png
  llm_backlog_trajectories.png
  llm_inventory_backlog.png
  llm_bullwhip_metrics.png
  llm_cumulative_cost.png

plots/comparison/          # after compare_models.py
  orders_qwen_vs_deepseek.png
  cumulative_rewards.png
  bullwhip_over_time.png
  inventory_oscillations.png
```

### Metrics in Python

```python
from simulator.beer_game import BeerGame
from metrics.stability import stability_summary

env = BeerGame(max_weeks=10)
env.reset()
# ... run simulation ...
print(stability_summary(env.get_history()))
print(len(env.get_trajectories()), "transitions logged")
```

---

## 7. Troubleshooting

### `Failed to connect to Ollama at http://localhost:11434`

- Start Ollama: `ollama serve`
- Confirm port: `curl http://localhost:11434/api/tags`
- Firewall: allow localhost connections on port 11434

### `Model 'qwen2.5:1.5b' not found`

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

### Request timeout (especially DeepSeek-R1)

- Reasoning models generate longer outputs; default timeout is **120 s**.
- Use GPU if available, or a shorter horizon (`--weeks 10`).
- Test with a smaller model first (`qwen:1.5b`).

### `No integer found in LLM response`

- The parser in `agents/llm_agent.py` handles verbose and reasoning outputs.
- If failures persist, try lower temperature (already `0.2`) or a different model.
- Check logs for the raw response snippet.

### `matplotlib is not installed; skipping plot generation`

```powershell
pip install matplotlib
```

### Baseline experiment uses wrong state format

Classical policies in `baseline_experiment.py` call `env.get_state()[node]`, which returns the **raw node dict** (list pipeline). Policies in `policies/base_stock.py` accept both list and summed pipeline. For new code, prefer `env.get_state_dict(agent_name)`.

### Reproducibility

- `baseline_experiment.py` and `compare_models.py` set `random.seed(seed)` per run.
- Customer demand in `BeerGame.generate_customer_demand()` is `random.randint(2, 8)` each week.

---

## 8. Optional: Live Ollama Integration Test

```powershell
python experiments/test_llm_agent.py --ollama
```

Confirms end-to-end prompt → Ollama → parsed order for a single agent call.

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for simulator step order, APIs, and data flow.
- Run `python main.py compare` after pulling both benchmark models.
- Use `env.get_trajectories()` exports for custom analysis or future RL training scripts.

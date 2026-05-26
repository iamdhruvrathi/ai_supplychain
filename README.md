# AI Supply Chain — Beer Game Research Framework

A research-grade Python framework for simulating the classic **Beer Game**, benchmarking **autonomous LLM agents** via [Ollama](https://ollama.ai), and analyzing **bullwhip dynamics** and **agent reliability**—aligned with *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long et al.). The codebase supports paper replication, repeated-run analysis, and future PPO/GRPO post-training.

## Research Motivation

Decentralized supply chains amplify demand variability upstream—the **bullwhip effect**. Human players in the Beer Game notoriously over-order and oscillate. This project asks:

- Can **local LLM policies** stabilize a four-echelon chain?
- How do **reasoning models** (e.g. DeepSeek-R1) compare to **direct models** (e.g. Qwen) on cost, backlog, and bullwhip?
- Can shaped rewards and trajectory logging support future **RL fine-tuning** (PPO/GRPO)?

## Current Capabilities

| Area | Status |
|------|--------|
| Multi-echelon simulator (Retailer → Factory) | ✓ |
| FIFO shipment pipelines, inventory, backlog, costs | ✓ |
| LLM agents via Ollama (decentralized, local state only) | ✓ |
| Bullwhip + stability metrics | ✓ |
| Trajectory logging `(s, a, r, s')` per agent | ✓ |
| Shaped reward (cost + bullwhip + backlog) | ✓ |
| Classical baselines (base-stock, moving average, random) | ✓ |
| Multi-model benchmarking pipeline | ✓ |
| **Agent bullwhip** (Ψ, Φ, σ² across runs) | ✓ |
| **Repeated-run reliability** (CV, tails, instability) | ✓ |
| Orchestrator modes + policy constraints | ✓ |
| YAML experiment configs | ✓ |
| Trajectory export (JSONL / CSV / parquet) | ✓ |
| CSV results + comparative plots | ✓ |
| PPO / GRPO training | Planned |
| Gymnasium wrapper | Planned |

## LLM Benchmarking

The evaluation pipeline compares **reasoning** and **non-reasoning** small models against a **base-stock** baseline over repeated seeds:

- `qwen2.5:1.5b` — compact instruction-following model
- `deepseek-r1:1.5b` — reasoning-oriented model (longer responses; robust parsing)
- `base_stock` — classical replenishment policy

Metrics include total cost, bullwhip ratio, backlog, shaped reward trajectories, and cumulative instability. See [SETUP.md](SETUP.md) for installation and [ARCHITECTURE.md](ARCHITECTURE.md) for technical detail.

## Quick Start

```powershell
# 1. Environment
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify (no Ollama required)
python main.py test-llm
python main.py demo

# 3. LLM experiment (requires Ollama — see SETUP.md)
ollama serve
ollama pull qwen2.5:1.5b
python main.py llm --weeks 30 --model qwen2.5:1.5b

# 4. Full model comparison
python main.py compare

# 5. Paper-aligned repeated runs (agent bullwhip + reliability)
python main.py repeated --runs 30 --offline   # offline = base-stock only

# 6. YAML-driven benchmark
python main.py benchmark --config configs/default_experiment.yaml
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py test-llm` | Unit tests (parsing, trajectories, reward shaping) |
| `python main.py test-state` | RL state API validation |
| `python main.py demo` | 10-week base-stock simulation |
| `python main.py baseline` | Classical policy comparison |
| `python main.py llm` | Single-model LLM experiment |
| `python main.py compare` | Qwen vs DeepSeek vs base-stock (10 runs each) |

## Documentation

| Document | Contents |
|----------|----------|
| [SETUP.md](SETUP.md) | Python, Ollama, models, GPU, experiments, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Simulator design, APIs, data flow, metrics, RL roadmap |
| [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) | Paper alignment audit, gaps, technical debt |
| [docs/REPLICATION_PLAN.md](docs/REPLICATION_PLAN.md) | Replication checklist and roadmap |
| [docs/METRICS.md](docs/METRICS.md) | Equations: classical & agent bullwhip, reliability |

## Project Layout

```
simulator/     BeerGame environment + SupplyChainNode
agents/        LLMAgent (Ollama integration)
policies/      Classical replenishment policies
metrics/       Bullwhip and stability analysis
evaluation/    Multi-model comparison + plots
experiments/   Experiment runners and tests
results/       CSV outputs
plots/         Generated figures
main.py        CLI entry point
```

## Future Roadmap (PPO / GRPO)

1. **Benchmark** — LLM and classical baselines with shared trajectories and metrics *(current phase)*.
2. **PPO** — Train per-echelon policies on `get_trajectories()` rollouts with shaped rewards.
3. **GRPO** — Group-relative optimization for aligning LLM ordering behavior.
4. **Gymnasium** — Standard `reset` / `step` wrapper for RL libraries.
5. **MARL** — Optional inter-agent communication on top of local observations.

## Citation

```bibtex
@software{beer_game_llm_2026,
  title  = {AI Supply Chain Beer Game Simulator},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/yourusername/ai_supplychain}
}
```

## References

- Sterman, J. D. (1989). Modeling managerial behavior in dynamic decision making experiments. *Management Science*.
- Lee, H. L., Padmanabhan, V., & Whang, S. (1997). The bullwhip effect in supply chains. *Sloan Management Review*.

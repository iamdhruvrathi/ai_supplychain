# Research Replication Status

## Repository Overview

This repository implements a Beer Game replication framework around local LLM agents, repeated-run evaluation, and bullwhip analysis.

Key components:

- `simulator/beer_game.py`: Beer Game environment with 4 echelons, lead time, inventory, backlog, pipeline inventory, and cost accounting.
- `agents/llm_agent.py`: LLM prompt construction, output parsing, decision generation, tool recommendation support, and majority-vote sampling.
- `agents/llm_backends.py`: Backend abstraction with `OllamaBackend` and `GroqBackend`.
- `evaluation/repeated_runs.py`: Fixed-demand repeated-run experiment engine, metrics export, and trajectory writer.
- `evaluation/plotting.py`: Figure 2/3 plotting helpers and exported PNG/PDF figures.
- `metrics/agent_bullwhip.py`: Agent bullwhip metrics, including sigma-squared, psi, and phi computations.
- `experiments/run_figure2.py`: Figure 2-style box plot wrapper.
- `experiments/run_majority_vote.py`: Majority-vote repeated-run wrapper.
- `experiments/smoke_test.py`: Minimal integration smoke test.
- `docs/GROQ_QWEN3_SETUP.md`: Groq + Qwen3 backend setup documentation.

## Paper Replication Checklist

| Component                                          | Status      | Evidence                                                                                                     | Notes                                                                                          |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| MIT Beer Game environment                          | ✅ Complete | `simulator/beer_game.py`, `simulator/config.py`                                                              | 4-echelon chain, lead time, inventory/backlog, shipments, and cost tracking are implemented.   |
| 4-echelon supply chain                             | ✅ Complete | `BeerGame.__init__`, node classes                                                                            | Retailer/Wholesaler/Distributor/Factory present; factory production modeled.                    |
| Lead times, inventory, backlog, pipeline inventory | ✅ Complete | `simulator/beer_game.py`, `simulator/node.py`                                                                | `incoming_shipments`, `pipeline_inventory`, `inventory`, and `backlog` are tracked per node.   |
| Cost calculations                                  | ✅ Complete | `simulator/beer_game.py`, `simulator/node.py`, `metrics/cost_analysis.py`                                    | Per-step and total costs are recorded.                                                         |
| Local decentralized LLM agents                     | ✅ Complete | `agents/llm_agent.py`, `evaluation/repeated_runs.py`                                                         | Agents make decentralized decisions with local state.                                          |
| Prompt engineering / orchestrator modes            | ✅ Partial  | `agents/llm_agent.py`, `simulator/orchestrator.py`, `simulator/config.py`                                    | Orchestrator and shared-state modes exist; paper-style ablation studies are not documented.    |
| Majority voting implementation                     | ✅ Partial  | `agents/llm_agent.py`, `experiments/run_majority_vote.py`, `experiments/run_figure3.py`                      | Majority-vote logic is implemented.                                                            |
| Fixed demand repeated runs                         | ✅ Complete | `evaluation/repeated_runs.py`, `simulator/demand.py`                                                         | Fixed-demand path and deterministic run support are present.                                   |
| Agent bullwhip metrics                             | ✅ Complete | `metrics/agent_bullwhip.py`, `evaluation/repeated_runs.py`                                                   | Metrics include per-agent and across-run bullwhip analysis.                                   |
| Figure 2 boxplot generation                        | ✅ Complete | `experiments/run_figure2.py`, `evaluation/plotting.py`, `plots/figure2_bullwhip_boxplots.png`                | Figure 2 artifact files are present.                                                           |
| Figure 3 majority-vote plots                       | ✅ Partial  | `experiments/run_majority_vote.py`, `experiments/run_figure3.py`, `plots/figure3_majority_vote_boxplots.png` | Plot generation exists; result folder validation is partial.                                   |
| Groq / Qwen3 remote backend support                | ✅ Partial  | `agents/llm_backends.py`, `docs/GROQ_QWEN3_SETUP.md`, `scripts/groq_smoke_test.py`                           | Groq backend adapter and docs exist; end-to-end validation is not fully confirmed.             |
| Reliability metrics                                | ✅ Complete | `metrics/reliability.py`, `evaluation/repeated_runs.py`                                                      | CV, instability, and backlog metrics are produced.                                             |
| Law of Total Variance / decomposition              | ❌ Missing  | No code found                                                                                                | Only bullwhip metric calculations are implemented.                                             |
| GRPO training / PPO                                | ❌ Missing  | No training code                                                                                             | No training or policy optimization pipeline is present.                                       |
| Figure 4 post-training reliability                 | ❌ Missing  | No scripts or results                                                                                        | No post-training reliability workflow exists.                                                 |
| Figure 5 post-training cost comparison             | ❌ Missing  | No scripts or results                                                                                        | No post-training cost-comparison figure implementation.                                       |
| Human baseline comparison                          | ❌ Missing  | No baseline dataset or loader                                                                                | No human baseline data or explicit comparison pipeline exists.                                |

## Figures and artifacts

### Figure 1

- Status: 🟡 Partial
- Evidence: `evaluation/compare_models.py`, `experiments/llm_experiment.py`
- Notes: Multi-model comparison support exists, but human baseline and exact paper dataset are absent.

### Figure 2

- Status: ✅ Complete
- Evidence: `evaluation/repeated_runs.py`, `experiments/run_figure2.py`, `plots/figure2_bullwhip_boxplots.png`, `plots/figure2_bullwhip_boxplots.pdf`
- Notes: Repeated-run experiment outputs are available under `results/`.

### Figure 3

- Status: ⏳ Partial
- Evidence: `experiments/run_majority_vote.py`, `experiments/run_figure3.py`, `plots/figure3_majority_vote_boxplots.png`, `plots/figure3_majority_vote_boxplots.pdf`
- Notes: Plot generation exists, but the result folder inventory and validation are incomplete.

### Figure 4 / Figure 5

- Status: ❌ Missing
- Notes: No training / post-training reliability or cost-comparison scripts are implemented.

## Confirmed outputs

- `results/qwen25_30runs/` — repeated-run experiment outputs.
- `results/qwen3_32b_fig2/` — experiment artifacts for a qwen3-style run.
- `results/qwen3_32b_fig3_n10/` — majority-vote run result folder.
- `results/figure3_n10/` and `results/figure3_n100/` — additional result directories.
- `plots/` contains generated figure PNG/PDF artifacts.

## Backend support

- Ollama is the primary supported LLM backend.
- `agents/llm_backends.py` exposes a Groq backend adapter.
- `experiments/run_majority_vote.py` and `experiments/llm_experiment.py` accept `--backend`.
- `evaluation/compare_models.py --offline` uses a stubbed LLM order function instead of real inference.

## Missing components and gaps

1. No RL training pipeline or GRPO/PPO implementation.
2. No explicit human baseline data or comparison pipeline.
3. No Law of Total Variance decomposition implementation.
4. No dedicated Figure 4 or Figure 5 post-training workflows.
5. `experiments/smoke_test.py` is the smoke test entry, but there is no `run_smoke_tests.py`.
6. `experiments/run_table1.py` does not exist.

## Recommended next steps

1. Validate the existing Figure 2 workflow using `results/qwen25_30runs`.
2. Re-run majority-vote experiments for `n=10` and `n=100` and confirm `results/figure3_*` outputs.
3. Add human-baseline or synthetic baseline data to support paper comparison claims.
4. Add a training wrapper and Gym-compatible RL environment if post-training replication is required.

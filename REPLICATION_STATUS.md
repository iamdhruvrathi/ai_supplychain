# Research Replication Status

## Repository Overview

This repository implements a Beer Game replication framework centered on a local LLM agent workflow, repeated-run evaluation, agent bullwhip metrics, and figure generation.

Key components:

- `simulator/beer_game.py`: Beer Game environment with 4 echelons, lead time, inventory, backlog, pipeline inventory, and cost accounting.
- `agents/llm_agent.py`: LLM prompt construction, parse logic, agent decision generation, and majority-vote sampling.
- `agents/llm_backends.py`: Backend abstraction with `OllamaBackend` and `GroqBackend`.
- `evaluation/repeated_runs.py`: Fixed-demand repeated-run experiment engine, agent metrics export, and trajectory writer.
- `evaluation/plotting.py`: Figure 2/3 plotting helpers and exported PDF/PNG figures.
- `metrics/agent_bullwhip.py`: Agent bullwhip metrics including σ², Ψ, Φ and per-echelon summaries.
- `experiments/run_figure2.py`: Figure 2-style box plot wrapper.
- `experiments/run_majority_vote.py`: Majority-vote experiment wrapper.
- `experiments/run_figure3.py`: Figure 3-style majority-vote plot wrapper.
- `docs/GROQ_QWEN3_SETUP.md`: Groq + Qwen3 remote backend setup documentation.
- `scripts/groq_smoke_test.py`: Groq SDK smoke test.
- `scripts/groq_smoke_test_postproc.py`: Groq post-processing validation.

## Paper Replication Checklist

| Component                                          | Status      | Evidence                                                                                                     | Notes                                                                                            |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| MIT Beer Game environment                          | ✅ Complete | `simulator/beer_game.py`, `simulator/config.py`                                                              | 4-echelon chain, lead time, inventory/backlog, shipments, and cost tracking are implemented.     |
| 4-echelon supply chain                             | ✅ Complete | `BeerGame.__init__`, node classes                                                                            | Retailer/Wholesaler/Distributor/Factory present; factory production modelled.                    |
| Lead times, inventory, backlog, pipeline inventory | ✅ Complete | `simulator/beer_game.py`, `simulator/node.py`                                                                | `incoming_shipments`, `pipeline_inventory`, `inventory`, and `backlog` are tracked per node.     |
| Cost calculations                                  | ✅ Complete | `simulator/beer_game.py`, `simulator/node.py`, `metrics/cost_analysis.py`                                    | Per-step and total costs are recorded.                                                           |
| Local decentralized LLM agents                     | ✅ Complete | `agents/llm_agent.py`, `evaluation/repeated_runs.py`                                                         | Agents make decentralized decisions using local model inference.                                 |
| Prompt engineering / orchestrator modes            | ✅ Partial  | `agents/llm_agent.py`, `simulator/orchestrator.py`, `simulator/config.py`                                    | Orchestrator and prompt logic exist; paper-style prompt ablations are not explicitly documented. |
| Majority voting implementation                     | ✅ Partial  | `agents/llm_agent.py`, `experiments/run_majority_vote.py`, `experiments/run_figure3.py`                      | Majority-vote logic is implemented.                                                              |
| Fixed demand repeated runs                         | ✅ Complete | `evaluation/repeated_runs.py`, `simulator/demand.py`                                                         | Fixed-demand path and deterministic run support are present.                                     |
| Agent bullwhip metrics                             | ✅ Complete | `metrics/agent_bullwhip.py`, `evaluation/repeated_runs.py`                                                   | Metrics include `sigma_squared`, `psi`, `phi`, and per-echelon summaries.                        |
| Figure 2 boxplot generation                        | ✅ Complete | `experiments/run_figure2.py`, `evaluation/plotting.py`, `plots/figure2_bullwhip_boxplots.png`                | Figure 2 artifacts are present.                                                                  |
| Figure 3 majority-vote plots                       | ✅ Partial  | `experiments/run_majority_vote.py`, `experiments/run_figure3.py`, `plots/figure3_majority_vote_boxplots.png` | Plot generation exists; some result folders require validation.                                  |
| Groq / Qwen3 remote backend support                | ✅ Partial  | `agents/llm_backends.py`, `docs/GROQ_QWEN3_SETUP.md`, `scripts/groq_smoke_test.py`                           | Backend support and docs exist; completed Groq experiment evidence is not fully validated.       |
| Reliability metrics                                | ✅ Complete | `metrics/reliability.py`, `evaluation/repeated_runs.py`                                                      | CV, instability, tail-event, and failure count metrics are produced.                             |
| Law of Total Variance / decomposition              | ❌ Missing  | No code found                                                                                                | Only bullwhip metric calculations are implemented.                                               |
| GRPO training / PPO                                | ❌ Missing  | No training code                                                                                             | No training pipeline is present.                                                                 |
| Figure 4 post-training reliability                 | ❌ Missing  | No scripts or results                                                                                        | Lacks trained-agent reliability figure generation.                                               |
| Figure 5 post-training cost comparison             | ❌ Missing  | No scripts or results                                                                                        | Lacks post-training cost comparison artifacts.                                                   |
| Human baseline comparison                          | ❌ Missing  | No baseline data or loader                                                                                   | No explicit human baseline integration.                                                          |

## Figures Reproduced

### Figure 1

- Status: 🟡 Partial
- Evidence: `evaluation/compare_models.py` and `experiments/llm_experiment.py` support model comparisons.
- Notes: No explicit human baseline figure or dataset. Paper-style Figure 1 is not fully reproduced.

### Figure 2

- Status: ✅ Complete
- Evidence: `evaluation/repeated_runs.py` and `experiments/run_figure2.py`; `plots/figure2_bullwhip_boxplots.png` and `.pdf` exist.
- Notes: Current results include `results/qwen25_mit_10runs` and `results/qwen3_32b_fig2`, showing repeated-run experiment artifacts.

### Figure 3

- Status: ✅ Partial
- Evidence: Majority-vote experiment code and `plots/figure3_majority_vote_boxplots.png` / `.pdf`.
- Notes: Plot artifacts exist, but repository evidence for fully validated N=10/N=100 experiment folders is partial. |

### Figure 4

- Status: ❌ Missing
- Evidence: none in code.
- Notes: No training/post-training reliability workflow implemented. |

### Figure 5

- Status: ❌ Missing
- Evidence: none in code.
- Notes: No post-training cost-comparison figure implementation. |

## Experiments and Artifacts

### Confirmed outputs

- `results/qwen25_mit_10runs/` — repeated-run report, `run_costs.csv`, `trajectories/rollouts.jsonl`.
- `results/qwen3_32b_fig2/` — generated result folder with trajectories.
- `results/figure3_test_n10/` — folder exists but is currently empty.
- `results/figure3_test_n100/` — folder exists but is currently empty.

### Plot artifacts

- `plots/figure2_bullwhip_boxplots.png`
- `plots/figure2_bullwhip_boxplots.pdf`
- `plots/figure3_majority_vote_boxplots.png`
- `plots/figure3_majority_vote_boxplots.pdf`
- `plots/llm_orders_vs_demand.png`
- `plots/llm_inventory_trajectories.png`
- `plots/llm_backlog_trajectories.png`
- `plots/llm_inventory_backlog.png`
- `plots/llm_bullwhip_metrics.png`
- `plots/llm_cumulative_cost.png`

## Groq / Qwen3 Support

- `agents/llm_backends.py` implements a `GroqBackend` in addition to Ollama.
- `agents/llm_agent.py` supports `--backend ollama` and `--backend groq`.
- `evaluation/repeated_runs.py` and `experiments/run_majority_vote.py` include backend selection support.
- `docs/GROQ_QWEN3_SETUP.md` documents how to run remote Groq `qwen/qwen3-4b` experiments.
- `scripts/groq_smoke_test.py` and `scripts/groq_smoke_test_postproc.py` validate Groq API connectivity and output parsing.
- `scripts/list_model.py` demonstrates Groq SDK usage.

## Missing Components and Gaps

1. No GRPO/PPO training pipeline or training code.
2. No human baseline dataset, loader, or explicit human comparison figure.
3. No Law of Total Variance decomposition implementation.
4. No dedicated Figure 4 or Figure 5 training/post-training artifacts.
5. Current majority-vote result folders are partially present, but some are empty.
6. Model mapping remains local/Ollama-first; cloud model provenance is not fully established.

## Recommended Next Steps

1. Re-run Figure 2 with `results/qwen25_mit_10runs` and verify the generated plot.
2. Re-run majority-vote experiments for `n=10` and `n=100` and confirm `results/figure3_*` contents.
3. Add a baseline loader or synthetic human reference to support Figure 1-style comparisons.
4. Add a training wrapper and Gym adapter if GRPO/PPO replication is required.

## Overall Replication Summary

- Core simulation and repeated-run framework: strong.
- LLM inference and majority-vote support: present.
- Figure 2 artifact reproduction: confirmed.
- Figure 3 support: implemented but only partially validated.
- Training/post-training components: absent.
- Full paper replication: incomplete.

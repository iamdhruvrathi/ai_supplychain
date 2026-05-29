# Research Replication Status

## Repository Overview

This repository contains a Beer Game replication framework with a local Ollama-based LLM agent, repeated-run evaluation, agent bullwhip metrics, and plotting tools.

Key components:
- `simulator/beer_game.py`: Beer Game environment, 4 echelons, lead time, inventory, backlog, shipments, and cost accounting.
- `agents/llm_agent.py`: Ollama LLM interface, prompt builder, parse logic, and majority-vote sampling.
- `evaluation/repeated_runs.py`: Fixed-demand repeated-run experiment engine, repeated-run metrics, trajectory export, and plot generation.
- `evaluation/plotting.py`: Figure 2 and Figure 3 plotting helpers plus research plot generation.
- `metrics/agent_bullwhip.py`: Agent bullwhip metrics (σ², Ψ, Φ, summary) aligned with paper definitions.
- `metrics/reliability.py`: Cost CV, run-to-run instability, tail event rate, order spikes, inventory/backlog failure events.
- `experiments/run_figure2.py`: Generate Figure 2-style box plots from repeated-run trajectories.
- `experiments/run_majority_vote.py`: Execute majority-vote repeated runs.
- `experiments/run_figure3.py`: Generate Figure 3-style majority-vote boxplots.
- `docs/REPLICATION_PLAN.md`: replication checklist and gap analysis.

## Paper Replication Checklist

| Component | Status | Evidence | Notes |
| --- | --- | --- | --- |
| MIT Beer Game environment | ✅ Complete | `simulator/beer_game.py`, `simulator/config.py` | 4 echelons, lead time, inventory, backlog, pipeline inventory, costs present. |
| 4-echelon supply chain | ✅ Complete | `BeerGame.__init__`, nodes Retailer/Wholesaler/Distributor/Factory | Factory uses upstream orders as production. |
| Lead times, inventory, backlog, pipeline inventory | ✅ Complete | `simulator/beer_game.py`, `SupplyChainNode` state | `incoming_shipments`, `pipeline_inventory`, `inventory`, `backlog` are tracked. |
| Cost calculations | ✅ Complete | `simulator/beer_game.py`, `simulator/node.py`, `metrics/cost_analysis.py` | Costs collected per step and total. |
| Local decentralized LLM agents | ✅ Complete | `agents/llm_agent.py`, `evaluation/repeated_runs.py` | Agents use local Ollama model per echelon. |
| Prompt engineering / orchestrator modes | ✅ Partial | `agents/llm_agent.py`, `simulator/orchestrator.py`, `simulator/config.py` | Modes exist; default is decentralized. No explicit paper prompt ablation recorded. |
| Majority voting implementation | ✅ Partial | `LLMAgent.generate_order_majority_vote`, `experiments/run_majority_vote.py`, `experiments/run_figure3.py` | Majority vote sampling implemented. |
| Fixed demand repeated runs | ✅ Complete | `evaluation/repeated_runs.py`, `simulator/demand.py` | `fixed_demand_path` and `demand_seed` used. |
| Agent bullwhip metrics | ✅ Complete | `metrics/agent_bullwhip.py`, `evaluation/repeated_runs.py` | Full σ², Ψ, Φ report generated. |
| Figure 2 boxplot generation | ✅ Complete | `experiments/run_figure2.py`, `evaluation/plotting.py` | `plots/figure2_bullwhip_boxplots.png` exists. |
| Figure 3 majority-vote plots | ✅ Partial | `experiments/run_figure3.py`, `plots/figure3_majority_vote_boxplots.png` exists | Figure 3 plot exists, but full model/experiment matching to paper not verified. |
| Reliability metrics | ✅ Complete | `metrics/reliability.py`, `evaluation/repeated_runs.py` | CV, instability, tail event, failure counts produced. |
| Law of Total Variance / decomposition | ❌ Missing | No code found for variance decomposition or explicit paper theoretical section | Only agent bullwhip and classical bullwhip metrics are implemented. |
| GRPO training / PPO | ❌ Missing | No `train/` or PPO/GRPO implementation found | `assets/` mentions future GRPO, but no training code present. |
| Figure 4 post-training reliability | ❌ Missing | No post-training figure scripts or outputs | Training paths absent. |
| Figure 5 post-training cost comparison | ❌ Missing | No dedicated figure or training comparison | No GRPO/finetuning results. |
| Human baseline comparison | ❌ Missing | No human baseline data or loader | README refers to human cost as future/needed. |

## Figures Reproduced

### Figure 1
- Status: 🟡 Partial
- Evidence: `evaluation/compare_models.py` and `experiments/llm_experiment.py` provide model comparison and cost plots, but no explicit human vs AI paper-style cost bar chart.
- Notes: Model comparison code exists, but there is no explicit `Figure 1` artifact or human baseline integration.

### Figure 2
- Status: ✅ Complete
- Evidence: `evaluation/repeated_runs.py` produces repeated-run reports and trajectories; `experiments/run_figure2.py` generates `plots/figure2_bullwhip_boxplots.png` and `.pdf`.
- Notes: Current runs use `qwen2.5:1.5b` and `results/qwen25_mit_10runs`. Report shows agent bullwhip metrics and CV.

### Figure 3
- Status: ✅ Partial/Implemented
- Evidence: `agents/llm_agent.py` majority vote, `experiments/run_majority_vote.py`, `experiments/run_figure3.py`, `plots/figure3_majority_vote_boxplots.png`.
- Notes: The functionality is present and results exist, but it is unclear whether current runs match the paper's exact N=10 and N=100 experimental conditions beyond the plot generation. The results folders `results/figure3_test_n10` and `results/figure3_test_n100` exist.

### Figure 4
- Status: ❌ Missing
- Evidence: No training or post-training reliability plot generator found.
- Notes: The repo has trajectory export and shaped reward, but no GRPO training or figure generation for post-training reliability.

### Figure 5
- Status: ❌ Missing
- Evidence: No dedicated cost-comparison plot after training exists.
- Notes: LLM experiment plots exist, but not paper-style post-trained cost comparison.

## Experiments Executed

### Discovered commands and outputs
- `python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs`
  - Output: `results/qwen25_mit_10runs/repeated_runs_report.json`, `run_costs.csv`, `trajectories/rollouts.jsonl`.
- `python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/`
  - Output: `plots/figure2_bullwhip_boxplots.png`, `plots/figure2_bullwhip_boxplots.pdf`.
- `python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 10 --output-dir results/figure3_test_n10 --progress run`
- `python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 100 --output-dir results/figure3_test_n100 --progress run`
- `python experiments/run_figure3.py --results-10 results/figure3_test_n10 --results-100 results/figure3_test_n100 --output plots/`
  - Output: `plots/figure3_majority_vote_boxplots.png`, `plots/figure3_majority_vote_boxplots.pdf`.

### Existing generated plots
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
- `plots/research/cost_boxplot.png`
- `plots/research/agent_bullwhip_heatmap.png`
- `plots/research/psi_phi_over_time.png`

## Existing Results

### Current result folders
- `results/qwen25_mit_10runs/` (current qwen25 output)
- `results/figure3_test_n10/`
- `results/figure3_test_n100/`
- `results/baseline_results.csv`
- `results/llm_experiment_results.csv`
- `results/model_comparison.csv`
- `results/model_comparison_summary.csv`
- `results/table1.csv`

### Current report evidence
- `results/qwen25_mit_10runs/repeated_runs_report.json` contains:
  - `cost` summary
  - `reliability` summary
  - `agent_bullwhip` metrics including `sigma_squared`, `psi`, `phi`, `psi_mean_by_echelon`, `phi_mean_by_echelon`.
- `results/qwen25_mit_10runs/trajectories/rollouts.jsonl` exists and is used by `experiments/run_figure2.py`.

## Missing Components

1. **Human baseline / Figure 1 human vs AI comparison**
   - No human dataset or loader.
   - No explicit human-normalized comparison in plots or metrics.

2. **GRPO / PPO training pipeline**
   - No training code or `train/` module.
   - No Gym/Gymnasium wrapper in `env/`.
   - No post-training reliability/cost figure generation.

3. **Theory decomposition / Total Variance proof**
   - No code for Law of Total Variance or paper-theory section.
   - Only agent bullwhip and classical bullwhip metrics are implemented.

4. **Figure 4 and Figure 5 artifacts**
   - No dedicated scripts or outputs matching trained agent reliability/cost comparison.

5. **Model mapping and exact paper model details**
   - Current code uses local Ollama model names `qwen2.5:1.5b` and optional `deepseek-r1:1.5b`.
   - No explicit mapping to the paper's cited model set or cloud API.

6. **Explicit demand-sharing or centralized policy experiments in finished results**
   - Orchestrator modes exist, but no documented completed results for demand-sharing/history-sharing are present in results folders.

## Recommended Next Steps

### Priority 1 — Critical tasks to reproduce core paper claims
1. Validate current Figure 2 replication with a full 30-run LLM experiment using `qwen2.5:1.5b` and publish the resulting plot.
   - Relevant files: `evaluation/repeated_runs.py`, `experiments/run_figure2.py`, `evaluation/plotting.py`.
   - Estimate: low.
2. Generate and inspect `plots/research/*` to confirm agent bullwhip metrics align with paper definitions.
   - Relevant files: `metrics/agent_bullwhip.py`, `evaluation/repeated_runs.py`, `evaluation/plotting.py`.
   - Estimate: low.
3. Create a synthetic baseline scenario or use `configs/default_experiment.yaml` to produce Table 1-style cost + CV results.
   - Relevant files: `evaluation/benchmark.py`, `configs/loader.py`, `metrics/reliability.py`.
   - Estimate: medium.

### Priority 2 — Reproduce remaining figures
1. Confirm and complete Figure 3 by re-running majority-vote experiments with N=10 and N=100, then generate plots.
   - Relevant files: `experiments/run_majority_vote.py`, `experiments/run_figure3.py`, `evaluation/plotting.py`.
   - Estimate: medium.
2. Add explicit Figure 1 script or README section that compares models and baseline policy costs.
   - Relevant files: `evaluation/compare_models.py`, `experiments/llm_experiment.py`.
   - Estimate: medium.

### Priority 3 — Nice-to-have improvements
1. Add a human baseline CSV loader and cost normalization function.
   - Relevant files: new data loader, plotting helper.
   - Estimate: medium.
2. Implement a Gymnasium wrapper and a `train/` folder for GRPO/PPO support.
   - Relevant files: new `env/` adapter, `train/` trainer.
   - Estimate: high.
3. Add unit tests for agent bullwhip and majority-vote parsing.
   - Relevant files: `experiments/test_llm_agent.py`, new tests.
   - Estimate: low.

## Suggested Commands To Run Next

```powershell
ollama serve
ollama pull qwen2.5:1.5b
python evaluation/repeated_runs.py --weeks 30 --runs 30 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

For Figure 3:

```powershell
python experiments/run_majority_vote.py --weeks 30 --runs 10 --model qwen2.5:1.5b --n-samples 10 --output-dir results/figure3_n10 --progress run
python experiments/run_majority_vote.py --weeks 30 --runs 10 --model qwen2.5:1.5b --n-samples 100 --output-dir results/figure3_n100 --progress run
python experiments/run_figure3.py --results-10 results/figure3_n10 --results-100 results/figure3_n100 --output plots/
```

## Overall Progress Estimate

- Environment replication: 90% ✅
- Figure replication: 55% 🟡
- Reliability analysis: 80% ✅
- Majority voting: 70% 🟡
- GRPO training: 0% ❌
- Full paper replication: 30% ❌

### Overall percentage estimate: 50%

Reasoning: The repo has a strong core simulation environment, repeated-run analysis, and agent bullwhip plotting. It lacks the training/post-training components, human baseline integration, and complete paper-model evidence for Figures 1, 4, and 5.

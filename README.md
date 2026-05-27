# AI Supply Chain Beer Game Replication

This repo is now set up around the simpler local model:

```text
qwen2.5:1.5b
```

That is intentional. Qwen3 reasoning mode is too slow/noisy for repeated Beer Game simulations.

## Goal

Replicate a Figure 2-style result from the paper:

```text
Run the same Beer Game many times with the same demand path,
collect each agent's order quantity,
and draw box plots by week and supply-chain role.
```

## Quick Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Ollama in another terminal:

```powershell
ollama serve
ollama pull qwen2.5:1.5b
```

## Test Without Ollama

```powershell
python main.py test-llm
python evaluation/repeated_runs.py --weeks 5 --runs 2 --offline --progress week --output-dir results/offline_debug
python experiments/run_figure2.py --results results/offline_debug --output plots/
```

## Small Qwen2.5 Test

Use this before any long run:

```powershell
python evaluation/repeated_runs.py --weeks 5 --runs 2 --model qwen2.5:1.5b --progress week --output-dir results/qwen25_debug
python experiments/run_figure2.py --results results/qwen25_debug --output plots/
```

## Practical Research Run

This is a reasonable next step:

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

## Figure 3: Majority Vote

Figure 3 is much more expensive because every agent decision samples the model many times.

Small check:

```powershell
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 10 --output-dir results/fig3_qwen25_n10_debug --progress run
python experiments/run_majority_vote.py --weeks 5 --runs 2 --model qwen2.5:1.5b --n-samples 100 --output-dir results/fig3_qwen25_n100_debug --progress run
python experiments/run_figure3.py --results-10 results/fig3_qwen25_n10_debug --results-100 results/fig3_qwen25_n100_debug --output plots/
```

Paper-style shape:

```powershell
python experiments/run_majority_vote.py --weeks 25 --runs 10 --model qwen2.5:1.5b --n-samples 10 --output-dir results/fig3_qwen25_n10 --progress run
python experiments/run_majority_vote.py --weeks 25 --runs 10 --model qwen2.5:1.5b --n-samples 100 --output-dir results/fig3_qwen25_n100 --progress run
python experiments/run_figure3.py --results-10 results/fig3_qwen25_n10 --results-100 results/fig3_qwen25_n100 --output plots/
```

Call count:

```text
runs * weeks * 4 agents * n_samples
```

For `25 weeks * 10 runs`:

```text
n=10  -> 10,000 model calls
n=100 -> 100,000 model calls
```

Full paper-style run:

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 30 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_30runs
python experiments/run_figure2.py --results results/qwen25_mit_30runs --output plots/
```

## Important Notes

- Demand defaults to the MIT Beer Game pattern: `4, 4, 4, 4, 8, ...`.
- Completed runs are checkpointed after each run.
- If you press `Ctrl+C`, completed runs remain in the output folder.
- Do not mix long LLM runs with smoke tests in the same output folder.

## Project Map

| Folder         | Purpose                     |
| -------------- | --------------------------- |
| `simulator/`   | Beer Game environment       |
| `agents/`      | Ollama LLM agent            |
| `evaluation/`  | Repeated runs and plotting  |
| `experiments/` | Small command wrappers      |
| `metrics/`     | Bullwhip, cost, reliability |
| `results/`     | Generated run data          |
| `plots/`       | Generated figures           |

# AI Supply Chain Beer Game Replication

This repo is a learning project for replicating parts of the paper:

**Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management**

The first concrete goal is simple:

1. Run the Beer Game many times with the same demand path.
2. Let each supply-chain role choose orders.
3. Collect the order quantities for every week and every role.
4. Make Figure 2-style box plots showing how order decisions vary across repeated runs.

The paper calls this **agent bullwhip**: even when demand is fixed, AI agents may make inconsistent decisions, and that inconsistency can grow as orders move upstream from Retailer to Factory.

## The Mental Model

The Beer Game has four roles:

```text
Customer demand -> Retailer -> Wholesaler -> Distributor -> Factory
```

Each week:

1. Customer demand arrives at the Retailer.
2. Each role ships what it can.
3. Each role may build inventory or backlog.
4. Each role places an order to its upstream supplier.
5. The simulator records orders, costs, inventory, backlog, and reward.

For the Figure 2 replication, the most important recorded value is:

```text
order quantity by run, week, and role
```

## What Is Already Here

| Folder         | Meaning                                      |
| -------------- | -------------------------------------------- |
| `simulator/`   | The Beer Game environment                    |
| `agents/`      | LLM agent wrapper for Ollama                 |
| `policies/`    | Simple non-LLM policies such as base-stock   |
| `evaluation/`  | Repeated runs, reliability metrics, plotting |
| `experiments/` | Friendly command wrappers                    |
| `metrics/`     | Bullwhip, agent bullwhip, cost, reliability  |
| `configs/`     | YAML experiment settings                     |
| `results/`     | Generated CSV/JSON/trajectory outputs        |
| `plots/`       | Generated figures                            |
| `docs/`        | Notes for learning and replication           |

## Fastest Way To Check The Pipeline

This does **not** require Ollama. It uses the base-stock policy, so it is useful for testing code, not for reproducing the paper's LLM behavior.

```powershell
python experiments/run_smoke_tests.py
```

Expected outputs:

```text
results/repeated_runs/repeated_runs_report.json
results/repeated_runs/run_costs.csv
results/repeated_runs/trajectories/rollouts.jsonl
plots/figure2_bullwhip_boxplots.png
```

If the box plot looks flat, that is normal for offline/base-stock runs. A deterministic policy does not create much run-to-run decision variance.

## Reproduce Figure 2-Style Box Plots

Use this once Ollama is running and the model is available:

```powershell
ollama serve
ollama pull qwen2.5:1.5b
python evaluation/repeated_runs.py --weeks 30 --runs 10 --model qwen2.5:1.5b --progress run --output-dir results/qwen25_mit_10runs
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/
```

Open:

```text
plots/figure2_bullwhip_boxplots.png
```

What you want to see:

- One panel per role: Retailer, Wholesaler, Distributor, Factory.
- One box per week.
- Wider boxes mean more run-to-run decision variability.
- If variability grows upstream, that is the agent bullwhip pattern.

## Important Difference: Smoke Test vs Paper Replication

| Run type                  | Needs Ollama?        | Purpose                     | Expected box plot        |
| ------------------------- | -------------------- | --------------------------- | ------------------------ |
| `--offline` / base-stock  | No                   | Check pipeline works        | Mostly flat              |
| LLM repeated runs         | Yes                  | Study agent unreliability   | Boxes should show spread |
| Human baseline comparison | External data needed | Match paper cost comparison | Not available yet        |

## Recommended Learning Path

1. Read `docs/REPLICATION_PLAN.md`.
2. Run the smoke test.
3. Open `results/qwen25_mit_10runs/trajectories/rollouts.jsonl` and inspect a few rows.
4. Read `docs/METRICS.md` only for the agent bullwhip section.
5. Run 3 to 5 LLM runs first.
6. Then run the full 30-run experiment.

## Useful Commands

```powershell
# Basic no-LLM checks
python main.py test-llm
python main.py test-state
python main.py demo

# Short repeated run without Ollama
python evaluation/repeated_runs.py --weeks 5 --runs 3 --offline

# Generate Figure 2-style plot from existing repeated-run output
python experiments/run_figure2.py --results results/qwen25_mit_10runs --output plots/

# Export cost summary table
python experiments/run_table1.py --results-root results/qwen25_mit_10runs --output results/table1.csv
```

## What Is Still Not Fully Replicated

This repo can generate a Figure 2-style plot, but exact paper replication still needs:

- The same model setup as the paper, or a clearly documented local substitute.
- 30 repeated LLM runs under the same demand path.
- Human baseline data for the paper's human-vs-AI cost comparison.
- Careful validation that simulator assumptions match the paper.

So the honest status is:

```text
The code can produce the artifact shape.
The research replication still needs model and data validation.
```

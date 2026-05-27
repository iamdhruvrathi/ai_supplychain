# Replication Plan

Target paper: **Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management** (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon).

This document is a **replication checklist** mapped to codebase capabilities.

---

## Replication Targets

### Tier 1 — Infrastructure (framework ready)

| #   | Paper target                                   | Status   | Command / module                             |
| --- | ---------------------------------------------- | -------- | -------------------------------------------- |
| 1.1 | Multi-echelon Beer Game (4 tiers, lead time 2) | ✓ Done   | `simulator/beer_game.py`                     |
| 1.2 | Decentralized local visibility                 | ✓ Done   | `OrchestratorMode.DECENTRALIZED`             |
| 1.3 | Repeated runs (R=30) same environment          | ✓ Done   | `evaluation/repeated_runs.py`                |
| 1.4 | Fixed demand path across runs                  | ✓ Done   | `demand_seed` in config                      |
| 1.5 | Agent bullwhip metrics                         | ✓ Done   | `metrics/agent_bullwhip.py`                  |
| 1.6 | Cost mean, std, CV                             | ✓ Done   | `metrics/reliability.py`, `cost_analysis.py` |
| 1.7 | Budget guardrail                               | ✓ Config | `constraints.budget_limit`                   |
| 1.8 | Orchestrator modes                             | ✓ Config | `orchestrator.mode` in YAML                  |
| 1.9 | YAML experiments                               | ✓ Done   | `configs/default_experiment.yaml`            |

### Tier 2 — Empirical replication (requires data + models)

| #   | Paper target                     | Status     | Gap                                         |
| --- | -------------------------------- | ---------- | ------------------------------------------- |
| 2.1 | Figure 1 cost bars (human vs AI) | ◐ Partial  | Need human baseline CSV                     |
| 2.2 | Table 1 cost + CV by scenario    | ◐ Partial  | Run `repeated_runs` per scenario            |
| 2.3 | Figure 2 agent bullwhip boxplots | ◐ Partial  | Add boxplot per echelon/week from R runs    |
| 2.4 | Figure 3 repeated sampling       | ✗ Missing  | Implement majority-vote in `LLMAgent`       |
| 2.5 | 67% cost reduction vs humans     | ✗ External | Requires human data + same cost accounting  |
| 2.6 | GPT-5 mini / Llama 4 Maverick    | ✗ API map  | Use closest Ollama models; document mapping |

### Tier 3 — Theory & training (future)

| #   | Paper target                    | Status                   |
| --- | ------------------------------- | ------------------------ |
| 3.1 | Transfer-function decomposition | ✗ Not coded              |
| 3.2 | GRPO post-training              | ✗ Trajectory export only |
| 3.3 | Tail event reduction after GRPO | ✗                        |

---

## Experiment Scenarios to Reproduce

Configure via YAML copies of `configs/default_experiment.yaml`:

### Scenario A — Decentralized baseline (Section 3.1)

```yaml
orchestrator:
  mode: decentralized
constraints:
  enabled: false
experiment:
  runs: 30
```

### Scenario B — Budget guardrail (Section 3.2)

```yaml
constraints:
  enabled: true
  budget_limit: 500 # tune to match paper units
  order_cap: 40
```

### Scenario C — Demand sharing (Section 3.3)

```yaml
orchestrator:
  mode: demand_sharing
```

### Scenario D — History sharing (Section 3.3)

```yaml
orchestrator:
  mode: history_sharing
```

### Scenario E — Agent bullwhip analysis (Section 4.2)

```yaml
experiment:
  runs: 30
  demand_seed: 42 # fixed path
evaluation:
  compute_agent_bullwhip: true
```

Run:

```powershell
python evaluation/benchmark.py --config configs/scenario_e.yaml
python -c "from evaluation.plotting import generate_research_plots; ..."
```

---

## Replication Checklist (operator)

- [ ] Install Python deps + Ollama (see [SETUP.md](../SETUP.md))
- [ ] Pull benchmark models (`qwen2.5:1.5b`, `deepseek-r1:1.5b`)
- [ ] Run `python main.py test-llm`
- [ ] Run decentralized 30-run baseline: `python evaluation/repeated_runs.py` (or benchmark)
- [ ] Verify `results/repeated_runs/repeated_runs_report.json`
- [ ] Verify Ψ_k > 1 upstream for LLM policies (agent bullwhip)
- [ ] Run with `constraints.enabled: true` — compare CV drop
- [ ] Run orchestrator modes — compare mean cost
- [ ] Export trajectories to JSONL for offline analysis
- [ ] Import human baseline (when available) for normalization

---

## Missing Experiments (prioritized)

1. **Human baseline comparison** — ingest Georgia Tech cohort CSV
2. **Majority-vote sampling** (10, 100 samples) — Section 4.3
3. **Per-week boxplots** — Figure 2 replication
4. **Cost normalization** — match paper scale (human = 100)
5. **Instruction-following failure rate** — track invalid orders per run
6. **Heterogeneous per-echelon models** — already in YAML schema

---

## Implementation Roadmap

| Sprint | Deliverable                              | Est.      |
| ------ | ---------------------------------------- | --------- |
| S1     | Agent bullwhip + repeated runs + docs    | ✓ Current |
| S2     | Figure-2 boxplots; scenario YAML suite   | 1 week    |
| S3     | Majority voting; invalid-order metrics   | 1 week    |
| S4     | Human baseline loader + normalized plots | 1 week    |
| S5     | GRPO trainer skeleton on trajectories    | 2–3 weeks |
| S6     | Gymnasium wrapper                        | 1 week    |

---

## Prioritized TODO Tree

```
Replication Framework
├── P0 Metrics correctness
│   ├── Unit tests: Ψ, Φ, σ² on synthetic orders
│   └── Validate fixed demand across runs
├── P0 Experiment harness
│   ├── benchmark.py from YAML
│   └── repeated_runs report JSON
├── P1 Paper figures
│   ├── Boxplot orders by echelon/week
│   └── Cost CV table export
├── P1 Interventions
│   ├── Budget constraint calibration
│   └── Orchestrator prompt A/B
├── P2 Sampling
│   └── Majority vote LLMAgent mode
└── P3 GRPO
    ├── Dataset from JSONL
    └── Training script in train/
```

---

## Success Criteria

Replication is **successful** when, under fixed `demand_seed` and 30 runs:

1. Mean total cost and CV are reported per model/scenario (Table 1 format).
2. Agent bullwhip shows Ψ > 1 upstream for at least one LLM configuration.
3. Budget constraint reduces CV vs. default (directional match to paper).
4. Demand-sharing reduces mean cost vs. decentralized for weaker models (directional).
5. Trajectories export cleanly for GRPO prototyping.

Exact numeric match to paper figures is **not required** until human baselines and model APIs are aligned.

# Replication Plan

This document keeps the project focused on the first research goal:

```text
Replicate the paper's Figure 2-style agent bullwhip box plots.
```

## What Figure 2 Means

In the paper, Figure 2 shows order quantities across repeated runs of the same Beer Game.

The key idea:

```text
Same demand path + same game + same prompt + repeated AI decisions
= measure decision unreliability
```

For each role and each week, we collect all orders from repeated runs and draw a box plot.

Example:

```text
Retailer, week 10: orders from 30 runs -> one box
Wholesaler, week 10: orders from 30 runs -> one box
Distributor, week 10: orders from 30 runs -> one box
Factory, week 10: orders from 30 runs -> one box
```

If the boxes get wider upstream, the model is showing **agent bullwhip**.

## Minimum Replication Target

| Target                               | Status                             |
| ------------------------------------ | ---------------------------------- |
| Four-echelon Beer Game               | Done                               |
| Repeated runs with fixed demand seed | Done                               |
| Trajectory export                    | Done                               |
| Figure 2-style box plot script       | Done                               |
| LLM run with Ollama                  | Implemented, needs local model run |
| Exact paper model/data match         | Not done                           |
| Human baseline comparison            | Not done                           |

## The Main Pipeline

```text
evaluation/repeated_runs.py
        |
        v
results/repeated_runs/trajectories/rollouts.jsonl
        |
        v
experiments/run_figure2.py
        |
        v
plots/figure2_bullwhip_boxplots.png
```

## Step 1: Prove The Code Works

No Ollama required:

```powershell
python experiments/run_smoke_tests.py
```

This creates a small offline run. The plot may look flat. That is fine.

## Step 2: Run A Tiny LLM Experiment

Use this before committing to 30 runs:

```powershell
ollama serve
ollama pull qwen2.5:1.5b
python evaluation/repeated_runs.py --weeks 5 --runs 3 --model qwen2.5:1.5b
python experiments/run_figure2.py --results results/repeated_runs --output plots/
```

Check:

```text
plots/figure2_bullwhip_boxplots.png
```

## Step 3: Run The Paper-Style Experiment

```powershell
python evaluation/repeated_runs.py --weeks 30 --runs 30 --model qwen2.5:1.5b
python experiments/run_figure2.py --results results/repeated_runs --output plots/
```

Use a different model by changing `--model`.

## How To Read The Outputs

### `rollouts.jsonl`

Each line is one agent's decision at one week:

```json
{
  "week": 1,
  "agent": "Retailer",
  "action": 7,
  "reward": -31.0
}
```

The Figure 2 script mainly needs:

```text
week
agent
action
```

### `repeated_runs_report.json`

This stores summary metrics:

```text
total_costs
cost
reliability
agent_bullwhip
invalid_orders_summary
```

### `figure2_bullwhip_boxplots.png`

This is the main visual artifact for the current goal.

## Why Offline Runs Are Not Enough

Offline/base-stock runs are useful for debugging because they are fast and deterministic.

But the paper's question is about autonomous AI reliability. For that, the ordering policy must have model-generated variation across repeated runs. That is why the real experiment needs Ollama or another LLM backend.

## What To Learn From The Experiment

Ask these questions while looking at the plot:

1. Does the Retailer stay relatively stable?
2. Do upstream roles show larger order spread?
3. Does spread grow over time within the same role?
4. Are there extreme outlier orders?
5. Do high-variance runs also create higher total cost?

These questions are more important than matching the paper's exact numbers at this stage.

## Next Improvements

Keep future work narrow:

| Priority | Improvement                                        | Why                          |
| -------- | -------------------------------------------------- | ---------------------------- |
| P0       | Validate Figure 2 plot on a real 30-run LLM output | Main replication goal        |
| P0       | Add a small test fixture for boxplot input parsing | Prevent plotting regressions |
| P1       | Save model name and run settings beside each plot  | Easier research tracking     |
| P1       | Compare `qwen2.5:1.5b` and `deepseek-r1:1.5b`      | Learn model differences      |
| P2       | Add human baseline CSV when available              | Needed for paper cost claims |
| P3       | Explore GRPO/PPO                                   | Later learning goal          |

# Replication Plan

Target paper: **Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management** (Long et al., arXiv:2605.17036).

This document is the **single replication checklist** mapped to codebase capabilities and completed experiment outputs.

**Last updated:** June 2026

---

## Replication Status Summary

| Area | Progress | Notes |
|------|----------|-------|
| Infrastructure (Tier 1) | ~100% | Simulator, metrics, repeated runs, YAML config |
| Empirical replication (Tier 2) | ~65% | Fig 2 qualitative; Fig 3 partial; Table 1 scenarios incomplete |
| Theory & training (Tier 3) | ~10% | Trajectory export only; no GRPO |

---

## Tier 1 — Infrastructure (complete)

| # | Paper target | Status | Command / module |
|---|--------------|--------|------------------|
| 1.1 | Multi-echelon Beer Game (4 tiers, lead time 2) | Done | `simulator/beer_game.py` |
| 1.2 | Decentralized local visibility | Done | `OrchestratorMode.DECENTRALIZED` |
| 1.3 | Repeated runs (R=30) same environment | Done | `evaluation/repeated_runs.py` |
| 1.4 | Fixed demand path across runs | Done | `demand_seed`, `fixed_demand_path`, MIT pattern |
| 1.5 | Agent bullwhip metrics (Ψ, Φ, σ²) | Done | `metrics/agent_bullwhip.py` |
| 1.6 | Cost mean, std, CV, median, IQR | Done | `metrics/cost_analysis.py`, `metrics/reliability.py` |
| 1.7 | Budget guardrail | Config ready | `agents/constraints.py` |
| 1.8 | Orchestrator modes (5) | Done | `simulator/orchestrator.py` |
| 1.9 | YAML experiments | Done | `configs/default_experiment.yaml`, `benchmark.py` |
| 1.10 | Trajectory export | Done | `trajectories/writer.py` (JSONL/CSV/Parquet) |
| 1.11 | Multi-backend LLM | Done | Ollama, Groq, vLLM via `agents/llm_backends.py` |

---

## Tier 2 — Empirical replication

| # | Paper target | Status | Evidence / gap |
|---|--------------|--------|----------------|
| 2.1 | Figure 1 (human vs AI cost bars) | Partial | No human baseline CSV in repo |
| 2.2 | Table 1 (cost + CV by scenario) | Partial | Orchestrator/budget scenarios not all run at R=30 |
| 2.3 | Figure 2 (agent bullwhip boxplots) | Done (qualitative) | `results/qwen25_30runs`, `experiments/run_figure2.py` |
| 2.4 | Figure 3 (majority-vote sampling) | Partial | Implemented; `figure3_n10` complete; `figure3_n100` 5/10 runs |
| 2.5 | 67% cost reduction vs humans | External | Requires human data |
| 2.6 | Frontier API models | Partial | Local proxies: `qwen2.5:1.5b`, `Qwen/Qwen3-4B`, `qwen/qwen3-32b` |

### Completed experiment outputs

| Folder | Description | Runs | Status |
|--------|-------------|------|--------|
| `results/qwen25_30runs` | Fig 2 baseline, qwen2.5:1.5b, decentralized | 30 | Complete |
| `results/qwen3_32b_fig2` | Fig 2, qwen3-32b (Groq) | 10 | Complete |
| `results/figure3_n10` | Majority vote N=10 | 10 | Complete |
| `results/figure3_n100` | Majority vote N=100 | 5/10 | Partial |
| `results/exp1_baseline` | Intervention baseline (Qwen3-4B/vLLM) | 30 | Complete |
| `results/exp2_tool` | Tool-assisted | 30 | Complete |
| `results/exp3_negotiation` | Negotiation mode | 30 | Complete |
| `results/exp4_tool_negotiation` | Tool + negotiation | 30 | Complete |

### Extension experiments (beyond paper)

| Intervention | Mean cost | CV | Consensus gap |
|--------------|-----------|-----|---------------|
| Baseline | 16,492 | 0.16 | 18.1 |
| Tool | 15,282 | 0.05 | 29.0 |
| Negotiation | 2,804 | 1.26 | 1.3 |
| Tool + Negotiation | 12,852 | 0.06 | 17.8 |

See `docs/results_analysis.md` and `analysis/negotiation_failure_report.md`.

---

## Tier 3 — Theory & training (future)

| # | Paper target | Status |
|---|--------------|--------|
| 3.1 | Transfer-function decomposition | Not coded |
| 3.2 | GRPO post-training | Trajectory export only |
| 3.3 | Tail event reduction after GRPO | Not started |

---

## Figure 2 Pipeline (primary replication goal)

```text
evaluation/repeated_runs.py
        |
        v
results/<experiment>/trajectories/rollouts.jsonl
        |
        v
experiments/run_figure2.py
        |
        v
plots/figure2_bullwhip_boxplots.png
```

### What Figure 2 measures

Same demand path + same game + same prompt + repeated AI decisions → measure **decision unreliability** (agent bullwhip). Per echelon and week, collect orders from R runs and plot boxplots.

### Quick start

**Offline smoke test (no Ollama):**

```bash
python evaluation/repeated_runs.py --weeks 5 --runs 2 --offline --output-dir results/offline_debug
python experiments/run_figure2.py --results results/offline_debug --output plots/
```

**Small LLM run:**

```bash
ollama serve
ollama pull qwen2.5:1.5b
python evaluation/repeated_runs.py --weeks 5 --runs 3 --model qwen2.5:1.5b
python experiments/run_figure2.py --results results/repeated_runs --output plots/
```

**Paper-style run:**

```bash
python evaluation/repeated_runs.py --weeks 20 --runs 30 --model qwen2.5:1.5b --output-dir results/qwen25_30runs
python experiments/run_figure2.py --results results/qwen25_30runs --output plots/
```

---

## Figure 3 Pipeline (majority vote)

Implemented in `LLMAgent.generate_order_majority_vote()` and `experiments/run_majority_vote.py`.

```bash
python experiments/run_majority_vote.py --weeks 20 --runs 10 --model qwen2.5:1.5b --n-samples 10 --output-dir results/figure3_n10
python experiments/run_majority_vote.py --weeks 20 --runs 10 --model qwen2.5:1.5b --n-samples 100 --output-dir results/figure3_n100
python experiments/run_figure3.py --results-10 results/figure3_n10 --results-100 results/figure3_n100 --output plots/
```

---

## Experiment Scenarios (YAML)

Copy `configs/default_experiment.yaml` for each scenario:

| Scenario | Orchestrator | Constraints | Purpose |
|----------|--------------|-------------|---------|
| A | `decentralized` | off | Section 3.1 baseline |
| B | any | `budget_limit` on | Section 3.2 guardrail |
| C | `demand_sharing` | off | Section 3.3 |
| D | `history_sharing` | off | Section 3.3 |
| E | `decentralized` | off, R=30 | Section 4.2 agent bullwhip |

Run:

```bash
python evaluation/benchmark.py --config configs/default_experiment.yaml
python evaluation/repeated_runs.py --config configs/scenario.yaml --runs 30 --model qwen2.5:1.5b
```

---

## Operator Checklist

- [x] Install Python deps + Ollama (see [SETUP.md](../SETUP.md))
- [x] Run `python main.py test-llm`
- [x] Decentralized 30-run baseline (`results/qwen25_30runs`)
- [x] Agent bullwhip Ψ > 1 upstream (LLM configs)
- [x] Majority-vote sampling (Fig 3)
- [x] Tool + negotiation intervention study (exp1–4)
- [ ] Budget constraint A/B at R=30
- [ ] Orchestrator modes A–D at R=30
- [ ] Complete `figure3_n100` to 30 runs
- [ ] LLM vs base-stock paired comparison at R=30
- [ ] Human baseline CSV import
- [ ] Export Table 1 CSV across scenarios

---

## Missing Experiments (prioritized)

| Priority | Experiment |
|----------|------------|
| P0 | Complete Fig 3 N=100 to 30 runs |
| P0 | LLM vs base-stock at R=30 (same demand) |
| P0 | Orchestrator modes at R=30 |
| P1 | Budget guardrail ON/OFF at R=30 |
| P1 | Tool ON/OFF on qwen2.5 (same backend as Fig 2) |
| P1 | Statistical tests (bootstrap CI, Mann-Whitney) |
| P2 | Human baseline comparison |
| P2 | Heterogeneous per-echelon models (YAML wired) |
| P3 | GRPO training on trajectories |

---

## Implementation Roadmap

| Sprint | Deliverable | Status |
|--------|-------------|--------|
| S1 | Agent bullwhip + repeated runs + docs | Done |
| S2 | Figure 2 boxplots + result folders | Done |
| S3 | Majority voting + intervention study | Done |
| S4 | Unit tests (Ψ/Φ, parse_order) | Done |
| S5 | Orchestrator/budget scenario sweep | In progress |
| S6 | Human baseline loader | Not started |
| S7 | GRPO trainer skeleton | Not started |

---

## Success Criteria

Under fixed demand and R=30 runs:

1. Mean total cost, CV, **median**, and **IQR** reported per scenario.
2. Agent bullwhip shows Ψ > 1 upstream for at least one LLM configuration.
3. Budget constraint reduces CV vs default (directional).
4. Demand-sharing reduces mean cost vs decentralized for weaker models (directional).
5. Trajectories export cleanly for offline analysis and GRPO prototyping.

Exact numeric match to paper figures is **not required** until human baselines and API models are aligned.

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `main.py` | CLI wrapper |
| `evaluation/repeated_runs.py` | Core repeated-run engine |
| `experiments/run_figure2.py` | Figure 2 boxplots |
| `experiments/run_figure3.py` | Figure 3 majority-vote plots |
| `experiments/run_majority_vote.py` | Majority-vote experiment runner |
| `experiments/smoke_test.py` | Minimal simulator smoke test (not `run_smoke_tests.py`) |
| `experiments/llm_experiment.py` | Single-run LLM experiment with plots |


Baseline anomaly:
Random policy achieves lower cost than Base Stock under
holding_cost=1 and backlog_cost=2.

Likely due to weak backlog penalty.
Requires cost-sensitivity study.
Not blocking Phase-0 experiments.

06-Jun-2026

- Completed repository cleanup phase.
- Added reliability metrics (median, IQR, failure_rate).
- Added Ψ/Φ validation tests.
- Added parser robustness tests.
- Began Figure 3 n=100 majority-vote experiment.
- Investigated baseline policies.
- Observed random policy outperforming base-stock under holding_cost=1, backlog_cost=2.
- Hypothesis: weak backlog penalty encourages under-ordering.
- Deferred cost-sensitivity study.
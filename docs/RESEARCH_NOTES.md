# Research Notes — Paper Alignment Audit

**Paper:** *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long et al., arXiv:2605.17036)

**Codebase audit date:** May 2026

---

## 1. Architecture Audit

### 1.1 Strengths (preserve)

| Module | Assessment |
|--------|------------|
| `simulator/beer_game.py` | Correct serial echelon flow; FIFO pipelines; RL `reset`/`step` API |
| `simulator/node.py` | Clean dataclass; reusable per-echelon dynamics |
| `agents/llm_agent.py` | Ollama HTTP; reasoning-model parsing; 120s timeout |
| `metrics/bullwhip.py` | Classical variance-ratio bullwhip |
| `metrics/stability.py` | Operational instability proxies |
| `evaluation/compare_models.py` | Multi-model repeated comparison |
| `trajectories` + shaped reward | GRPO-ready foundations |

### 1.2 Gaps vs. paper (addressed in this refactor)

| Paper requirement | Prior state | Current state |
|-------------------|-------------|---------------|
| Fixed demand path across R runs | Stochastic i.i.d. only | `DemandGenerator` + `demand_seed` |
| Agent bullwhip Ψ, Φ, σ² | Not implemented | `metrics/agent_bullwhip.py` |
| Reliability (CV, tails) | Partial (std in compare) | `metrics/reliability.py` |
| Repeated-run engine | `compare_models` only | `evaluation/repeated_runs.py` |
| Budget guardrails | `max_order` clamp only | `agents/constraints.py` |
| Orchestrator modes | None | `simulator/orchestrator.py` |
| YAML experiments | None | `configs/` + `benchmark.py` |
| Trajectory JSONL/parquet | In-memory list only | `trajectories/writer.py` |

### 1.3 Architectural inconsistencies (technical debt)

1. **Dual state APIs:** `get_state()` returns raw node dicts; `get_state_dict()` returns RL schema. Baseline experiment uses raw API — document and converge on `get_agent_state()`.
2. **Shared vs. per-agent reward:** Paper uses system cost; trajectories duplicate same reward per agent — acceptable for MARL but note for PPO.
3. **Classical bullwhip definition:** Code uses `Var(orders)/Var(demand)`; paper Section 4.1 also defines adjacent-echelon `Var(q_k)/Var(q_{k-1})` — now in `classical_bullwhip_adjacent()`.
4. **Human baseline data:** Paper uses Georgia Tech cohorts — not in repo; external dataset required for Figure 1 replication.
5. **Model API mismatch:** Paper uses frontier APIs (GPT-5 mini, Llama 4 Maverick); repo uses Ollama local tags — map models explicitly in configs.

### 1.4 Scalability bottlenecks

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| 4 LLM calls × weeks × runs | Slow experiments | Cache prompts; batch API; `--offline` for CI |
| Synchronous Ollama HTTP | Serial latency | Async client (future) |
| In-memory trajectories | Memory at 30×30×4×weeks | JSONL streaming writer |
| No parallel runs | Wall-clock | `multiprocessing` wrapper (future) |

---

## 2. Paper Concept Mapping

| Paper concept | Implementation |
|---------------|----------------|
| Agent bullwhip σ²(k,t) | `metrics/agent_bullwhip.sigma_squared` |
| Ψ_k(t) cross-echelon | `metrics/agent_bullwhip.psi_ratio` |
| Φ_k(t) intertemporal | `metrics/agent_bullwhip.phi_ratio` |
| Classical bullwhip | `metrics/bullwhip` |
| Budget constraint | `ConstraintConfig.budget_limit` |
| Orchestrator demand sharing | `OrchestratorMode.DEMAND_SHARING` |
| History + volatility sharing | `OrchestratorMode.HISTORY_SHARING` |
| Centralized summaries | `OrchestratorMode.CENTRALIZED` |
| Repeated sampling (Sec 4.3) | Not implemented — roadmap |
| GRPO post-training (Sec 5) | Trajectory export only — roadmap |

---

## 3. Suggested Folder Structure (target)

```
simulator/          # environment, demand, rewards, config, orchestrator
agents/             # llm, constraints (+ policies remain in policies/)
metrics/            # bullwhip, agent_bullwhip, reliability, stability, cost
evaluation/         # repeated_runs, benchmark, compare_models, plotting
trajectories/       # schema, writer
configs/            # YAML experiments
experiments/        # legacy runners (kept for CLI compat)
docs/               # research documentation
results/ plots/     # outputs
```

`beer_game.py` retained as primary import path; `environment.py` aliases for RL.

---

## 4. Replication Roadmap (summary)

See [REPLICATION_PLAN.md](REPLICATION_PLAN.md) for checklist.

**Phase A (current):** Metrics + repeated runs + config + orchestrator skeleton  
**Phase B:** Human baseline ingestion; model name mapping; budget experiments  
**Phase C:** Repeated sampling (10/100 vote); boxplot Figure 2 style  
**Phase D:** GRPO training loop on exported trajectories  

---

## 5. Technical Debt List

| Priority | Item |
|----------|------|
| P0 | Add unit tests for agent_bullwhip Ψ/Φ on synthetic tensors |
| P0 | Wire `llm_experiment` to use `get_agent_state()` + config |
| P1 | Async Ollama client for parallel echelons |
| P1 | Majority-vote sampling mode (paper Sec 4.3) |
| P2 | Human baseline CSV loader |
| P2 | Gymnasium `Env` wrapper in `env/` |
| P3 | GRPO trainer in `train/` |

---

## 6. Research Extension Opportunities

1. **Synthetic data reliability** (paper: training on synthetic data improves reliability)
2. **Demand-driven vs. decision-driven variance decomposition** (Eq. 1 in paper)
3. **Per-echelon model assignment** (heterogeneous agents in YAML)
4. **CVaR-shaped rewards** for tail-risk reduction
5. **Communication protocols** beyond orchestrator broadcast

---

## 7. PPO / GRPO Preparation Notes

| Asset | Location | GRPO use |
|-------|----------|----------|
| Standardized trajectories | `trajectories/schema.py` | Group sampling by prompt/state |
| System reward | `simulator/rewards.py` | Shared reward signal |
| `TrajectoryWriter` | JSONL/parquet export | Offline RL dataset |
| `compute_metrics()` | `BeerGame.compute_metrics()` | Evaluation after policy update |

**Recommended GRPO grouping:** Same `(week, agent_role, demand_path_id)` → sample G order completions → rank by step reward or episode return.

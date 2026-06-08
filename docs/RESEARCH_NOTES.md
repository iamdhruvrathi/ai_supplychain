# Research Notes — Paper Alignment & Current Implementation

**Paper:** *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long et al., arXiv:2605.17036)

**Last updated:** June 2026

---

## 1. Current Implementation Status

### 1.1 Core platform (complete)

| Module | Status | Role |
|--------|--------|------|
| `simulator/beer_game.py` | Complete | 4-echelon Beer Game, FIFO pipelines, RL `reset`/`step` |
| `simulator/orchestrator.py` | Complete | 5 information-sharing modes |
| `agents/llm_agent.py` | Complete | Prompt, parse, majority vote, tool injection |
| `agents/llm_backends.py` | Complete | Ollama, Groq, vLLM backends |
| `agents/constraints.py` | Complete | Order cap, budget, safety stock, smoothing |
| `evaluation/repeated_runs.py` | Complete | R-run experiment engine, negotiation protocol |
| `metrics/agent_bullwhip.py` | Complete | Ψ, Φ, σ² across repeated runs |
| `metrics/reliability.py` | Complete | CV, tails, failure_rate, event detectors |
| `metrics/cost_analysis.py` | Complete | Mean, std, median, IQR, CI |
| `trajectories/` | Complete | JSONL/CSV/Parquet export |
| `configs/` + `benchmark.py` | Complete | YAML-driven experiments |

### 1.2 Research features implemented

| Feature | Implementation | Result folders |
|---------|----------------|----------------|
| **Majority-vote sampling** (Sec 4.3) | `LLMAgent.generate_order_majority_vote()`, `--n-samples`, `run_majority_vote.py` | `figure3_n10`, `figure3_n100` (partial) |
| **Tool-assisted ordering** | `tools/inventory_tool.py`, `--use-tool-recommendation` | `exp2_tool`, `exp4_tool_negotiation` |
| **Negotiation mode** | Two-round proposal/revision in `repeated_runs.py` | `exp3_negotiation`, `exp4_tool_negotiation` |
| **Reliability metrics** | CV, tail_event_rate_p90, failure_rate, spike/collapse/backlog detectors | All `repeated_runs_report.json` |
| **Agent bullwhip** | Ψ cross-echelon, Φ intertemporal | All repeated-run reports |
| **Consensus metrics** | `consensus_gap` per week | exp1–4 reports |
| **Multi-backend** | `--backend ollama\|groq\|vllm` on `repeated_runs.py` and `llm_experiment.py` | qwen25 (Ollama), Qwen3-4B (vLLM), qwen3-32b (Groq) |

### 1.3 Not yet implemented

| Feature | Status |
|---------|--------|
| Human baseline ingestion | External CSV required |
| GRPO post-training | Trajectory export only |
| Per-echelon YAML model wiring | Schema exists; single `model_name` in runner |
| Budget guardrail experiments | Config only; no published A/B results |
| Orchestrator mode sweep at R=30 | Code ready; limited result folders |
| Dynamic policy selection | Roadmap (`updated_new_plan.md`) |
| Forecast-aware agents | EOQ tool uses last demand only |

---

## 2. Paper Concept Mapping

| Paper concept | Implementation |
|---------------|----------------|
| Agent bullwhip σ²(k,t) | `metrics/agent_bullwhip.sigma_squared` |
| Ψ_k(t) cross-echelon | `metrics/agent_bullwhip.psi_ratio` |
| Φ_k(t) intertemporal | `metrics/agent_bullwhip.phi_ratio` |
| Classical bullwhip | `metrics/bullwhip.bullwhip_ratio` |
| Reliability CV | `metrics/reliability.coefficient_of_variation` |
| Tail events | `metrics/reliability.tail_event_rate` |
| Run failure rate | `metrics/reliability.failure_rate` (tail + event detectors) |
| Budget constraint | `ConstraintConfig.budget_limit` |
| Demand sharing | `OrchestratorMode.DEMAND_SHARING` |
| History + volatility sharing | `OrchestratorMode.HISTORY_SHARING` |
| Centralized summaries | `OrchestratorMode.CENTRALIZED` |
| Negotiation / consensus | `OrchestratorMode.NEGOTIATION` + two-round protocol |
| Repeated sampling (Sec 4.3) | `generate_order_majority_vote` |
| Tool-augmented decisions | `use_tool_recommendation` + `eoq_recommendation` |
| GRPO post-training (Sec 5) | Trajectory export only |

---

## 3. Key Empirical Findings (extension study)

**Setup:** Qwen/Qwen3-4B (vLLM), 30 runs × 20 weeks, fixed MIT demand path.

| Condition | Mean cost | CV | Consensus gap |
|-----------|-----------|-----|---------------|
| Baseline | 16,492 | 0.16 | 18.1 |
| Tool | 15,282 | 0.05 | 29.0 |
| Negotiation | 2,804 | 1.26 | 1.3 |
| Tool + Negotiation | 12,852 | 0.06 | 17.8 |

**Interpretation:**
- Negotiation minimizes cost and consensus gap but produces **bimodal** outcomes (CV=1.26; costs 560–12,168).
- Tool assistance improves reliability (CV) but can worsen coordination (higher consensus gap).
- Report median/IQR and `failure_rate` alongside mean cost for negotiation results.

See `docs/results_analysis.md`, `analysis/negotiation_failure_report.md`.

---

## 4. Architecture Notes

### Strengths (preserve)

- Serial echelon flow with correct shipment pipeline
- Repeated-run isolation of LLM stochasticity (fixed demand path)
- Pluggable LLM backends with shared `LLMAgent` interface
- Standardized trajectory schema for offline RL / GRPO

### Technical debt

| Priority | Item | Status |
|----------|------|--------|
| P0 | Unit tests for Ψ/Φ | Done (`tests/test_agent_bullwhip.py`) |
| P0 | Unit tests for parse_order | Done (`tests/test_parse_order.py`) |
| P0 | `llm_experiment.py` backend wiring | Done |
| P1 | Split `beer_game.py` (758 LOC) | Open |
| P1 | Wire per-echelon YAML models | Open |
| P1 | Async Ollama for parallel echelons | Open |
| P2 | Human baseline CSV loader | Open |
| P2 | Gymnasium `Env` wrapper | Open |
| P3 | GRPO trainer in `train/` | Open |

### Scalability

| Bottleneck | Mitigation |
|------------|------------|
| 4 LLM calls × weeks × runs | `--offline` for CI; multiprocessing (future) |
| Synchronous HTTP | Async client (future) |
| In-memory trajectories | JSONL streaming writer (implemented) |

---

## 5. Orchestrator Modes

| Mode | Shared information | Use case |
|------|-------------------|----------|
| `decentralized` | Local state only | Paper baseline |
| `demand_sharing` | Current customer demand | Information sharing Sec 3.3 |
| `history_sharing` | Demand history + volatility | Richer sharing |
| `centralized` | System totals + echelon snapshot | Near-central control |
| `negotiation` | Centralized + two-round proposals | Consensus-seeking extension |

CLI: `--orchestrator-mode` on `evaluation/repeated_runs.py`.

---

## 6. Reliability & Reporting

`repeated_runs_report.json` includes:

- **cost:** mean, std, min, max, median, **iqr**, CI
- **reliability:** CV, tail_event_rate_p90, **failure_rate**, median_cost, iqr_cost
- **agent_bullwhip:** Ψ, Φ, σ² per echelon
- **consensus:** mean/max consensus gap

**failure_rate** flags runs using existing detectors: tail cost (p90), order spikes, inventory collapse, backlog explosion.

---

## 7. Testing

```bash
pytest tests/ -v
python main.py test-llm
python main.py test-state
```

---

## 8. Forward Research (see `updated_new_plan.md`)

Long-term direction: **forecast-aware LLM policy orchestrator** — per-echelon time series models feed structured summaries to an orchestrator that selects ordering policies (OUT, POUT, EOQ) rather than raw order quantities.

Near-term: complete replication gaps (orchestrator sweep, Fig 3 N=100, base-stock baseline) while building forecast layer.

---

## 9. GRPO Preparation

| Asset | Location |
|-------|----------|
| Standardized trajectories | `trajectories/schema.py` |
| System reward | `simulator/rewards.py` |
| TrajectoryWriter | JSONL/parquet export |
| Grouping key | `(week, agent_role, demand_path_id)` |


Figure 3 (n=10)
Mean Cost: 17,704.9
Runs: 10
Weeks: 20
Model: qwen2.5:1.5b
Backend: Ollama
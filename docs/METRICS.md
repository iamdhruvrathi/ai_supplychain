# Metrics Reference

Equations and implementations for paper-aligned evaluation. Code lives in `metrics/`.

---

## 1. Classical Bullwhip

**Paper (Section 4.1):** upstream amplification of order variability relative to downstream.

### Adjacent echelon (single run)

\[
B_k = \frac{\mathrm{Var}_t(q_{k,t})}{\mathrm{Var}_t(q_{k-1,t})}
\]

**Code:** `metrics/bullwhip.classical_bullwhip_adjacent(orders_upstream, orders_downstream)`

### Versus customer demand (implemented default)

\[
B_k = \frac{\mathrm{Var}_t(q_{k,t})}{\mathrm{Var}_t(D_t)}
\]

**Code:** `metrics/bullwhip.bullwhip_ratio(orders, demand)`  
**Environment:** `BeerGame.compute_bullwhip()`

### Rolling window

**Code:** `metrics/bullwhip.rolling_bullwhip(orders, demand, window=5)`

---

## 2. Agent Bullwhip (Definition 1)

**Paper:** amplification of **run-to-run** decision variance — not demand variance.

### Run-to-run order variance

\[
\sigma^2_{k,t} = \mathrm{Var}_r\big(q^{(r)}_{k,t}\big)
\]

- \(r = 1,\ldots,R\) replications  
- Same demand path, config, prompts; LLM stochasticity varies \(q\)

**Code:** `metrics/agent_bullwhip.sigma_squared(orders_by_run, echelon, week)`

### Cross-echelon amplification

\[
\Psi_k(t) = \frac{\mathrm{Var}_r(q^{(r)}_{k,t})}{\mathrm{Var}_r(q^{(r)}_{k-1,t})}
\]

\(\Psi_k(t) > 1\): run-to-run instability amplifies moving upstream.

**Code:** `metrics/agent_bullwhip.psi_ratio(...)`

### Cumulative adjacent product

\[
C_j(t) = \prod_{k=1}^{j} \Psi_k(t)
\]

**Code:** `metrics/agent_bullwhip.cumulative_psi(psi_values)`

### Intertemporal amplification

\[
\Phi_k(t) = \frac{\mathrm{Var}_r(q^{(r)}_{k,t+1})}{\mathrm{Var}_r(q^{(r)}_{k,t})}
\]

\(\Phi_k(t) > 1\): instability grows within echelon \(k\) over time.

**Code:** `metrics/agent_bullwhip.phi_ratio(...)`

### Full report

**Code:** `metrics/agent_bullwhip.agent_bullwhip_report(orders_by_run)`

**Input tensor:** `{run_id: {echelon: [q_0, q_1, ...]}}` from `orders_tensor_from_runs(histories)`

---

## 3. Reliability Metrics

Aligned with paper Table 1 and Section 4.

| Metric | Formula / definition | Code |
|--------|----------------------|------|
| Mean cost | \(\bar{C} = \frac{1}{R}\sum_r C^{(r)}\) | `cost_analysis.mean_std` |
| Std dev | \(\sigma_C\) | `cost_analysis.mean_std` |
| Coefficient of variation | \(\mathrm{CV} = \sigma_C / \bar{C}\) | `reliability.coefficient_of_variation` |
| Run-to-run instability | \(\mathrm{Var}_r(C^{(r)})\) | `reliability.run_to_run_instability` |
| 95% CI for mean | Normal approx. | `cost_analysis.confidence_interval` |
| Tail event rate | Fraction of runs above p90 cost | `reliability.tail_event_rate` |
| Order spikes | Weeks with \(q > 2.5 \times \mathrm{median}(q)\) | `reliability.detect_order_spikes` |
| Inventory collapse | Consecutive weeks with \(I \leq 1\) | `reliability.detect_inventory_collapse` |
| Backlog explosion | Weeks with \(B \geq 15\) | `reliability.detect_backlog_explosion` |

**Aggregate:** `reliability.reliability_summary(total_costs, ...)`

---

## 4. Stability Metrics (operational)

Single-run proxies for order/inventory/backlog volatility:

| Function | Description |
|----------|-------------|
| `stability.order_variance(history)` | Per-agent order variance |
| `stability.inventory_variance(history)` | Per-agent inventory variance |
| `stability.backlog_variance(history)` | Per-agent backlog variance |
| `stability.cumulative_instability(history)` | Weighted mean across agents |

---

## 5. Shaped Reward (GRPO preparation)

System-level reward per week:

\[
R_t = -\big(\alpha C_t + \beta B_t + \gamma L_t\big)
\]

| Term | Meaning |
|------|---------|
| \(C_t\) | Total holding + backlog cost |
| \(B_t\) | Overall bullwhip ratio (0 if undefined) |
| \(L_t\) | Sum of backlogs |

**Code:** `simulator/rewards.compute_shaped_reward`  
**Config:** `configs/*.yaml` → `reward.alpha/beta/gamma`

---

## 6. Simulator Timing & Assumptions

| Assumption | Value |
|------------|-------|
| Echelons | Retailer → Wholesaler → Distributor → Factory |
| Lead time | 2 weeks (FIFO deque) |
| Initial inventory | 20 units/echelon |
| Holding cost | 1.0 / unit / week |
| Backlog cost | 2.0 / unit / week |
| Customer demand | Fixed path or U[2,8] i.i.d. |
| Factory production | Order = production entering pipeline |
| Communication | None unless orchestrator enabled |
| Reward | Shared across agents |

**Weekly step order:** receive → demand → fulfill → ship → order → cost → log.

---

## 7. Experiment Flow

```
YAML config → configs/loader.py → SimulationConfig
       ↓
evaluation/repeated_runs.py (R episodes, fixed demand)
       ↓
Per run: get_agent_state → policy → constraints → step
       ↓
orders_tensor → agent_bullwhip_report
total_costs   → reliability_summary, cost_summary
trajectories  → TrajectoryWriter (JSONL/CSV/parquet)
       ↓
results/repeated_runs/repeated_runs_report.json
plots/research/*.png (optional)
```

---

## 8. API Quick Reference

```python
from simulator.beer_game import BeerGame
from simulator.config import SimulationConfig
from metrics.agent_bullwhip import agent_bullwhip_report, orders_tensor_from_runs
from evaluation.repeated_runs import run_repeated_experiment

config = SimulationConfig(max_weeks=30, demand_seed=42)
report = run_repeated_experiment(config, n_runs=30, model_name="qwen2.5:1.5b")
print(report["reliability"]["coefficient_of_variation"])
print(report["agent_bullwhip"]["psi_mean_by_echelon"])
```

---

## 9. Distinction: Classical vs. Agent Bullwhip

| | Classical bullwhip | Agent bullwhip |
|--|-------------------|----------------|
| **Variance over** | Time (within one run) | Runs (same week, fixed demand) |
| **Numerator** | Orders at echelon k | Orders at k across runs |
| **Denominator** | Demand or downstream orders | Downstream echelon across runs |
| **Measures** | Demand signal amplification | Decision unreliability amplification |

Both can coexist: a policy may have low classical bullwhip but high agent bullwhip if LLM orders are inconsistent across runs.

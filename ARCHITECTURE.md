# AI Supply Chain Architecture

## Project Overview

### Purpose

This repository implements a research-grade replication of the Beer Game supply chain using local LLM-driven agents, classical policy baselines, and repeated-run reliability experiments. It is designed to compare how language models perform under supply-chain dynamics, demand uncertainty, and different information-sharing modes.

### Research goals

- Replicate the Beer Game environment for decentralized decision-making research.
- Evaluate LLM agents against classical heuristics.
- Measure bullwhip amplification in both single-run and repeated-run settings.
- Quantify reliability across stochastic LLM outputs.
- Support tool-augmented decision support and negotiation-style coordination.

### Relationship to the Beer Game and Bullwhip Effect

This project models the canonical four-echelon Beer Game: Retailer, Wholesaler, Distributor, and Factory. It emphasizes the bullwhip effect — the amplification of order variance as information propagates upstream — and evaluates how LLM agents influence that amplification compared to baseline policies.

### Reliability and stochasticity objectives

The main objective is to isolate the impact of LLM decision stochasticity on outcome variability. The repository therefore uses repeated-run experiments with fixed demand paths and repeated seeds to measure model reliability, tail risk, and run-to-run instability.

---

## High-Level Architecture

### End-to-end system overview

The system is composed of four primary subsystems:

- Simulation Engine (simulator/): physical supply-chain dynamics and environment state.
- Agent Layer (gents/): LLM agent wrapper and backend interfaces.
- Evaluation Layer (evaluation/, experiments/): experiment execution, plotting, and report generation.
- Metrics Layer (metrics/): bullwhip, stability, cost, and reliability analysis.

Agents can be controlled by classical policies or by LLMs via Ollama, Groq, or vLLM backends. Experiment runners drive the simulation and export results into structured reports and trajectory logs.

### Main subsystems

- simulator/ — implements the Beer Game mechanics, state APIs, orchestrator information sharing, and reward shaping.
- gents/ — encapsulates the LLM prompt generation and response parsing lifecycle, plus backend abstraction for inference providers.
- 	ools/ — supplies optional decision support recommendations injected into LLM prompts.
- policies/ — provides classical order policies used as baselines.
- evaluation/ — orchestrates repeated-run experiments, model comparisons, plot generation, and benchmark execution.
- 	rajectories/ — standardizes and exports per-step rollout data.
- configs/ — holds YAML experiment definitions and loader logic.

### Responsibilities of each subsystem

- simulator/: enforces supply-chain rules, computes costs and bullwhip, retains history and trajectories.
- gents/: translates local or shared state into model prompts and returns safe order quantities.
- evaluation/: runs experiments, computes aggregate metrics, and persists output artifacts.
- metrics/: defines the quantitative measures used to evaluate bullwhip, stability, and reliability.
- 	ools/: adds optional decision support guidance for the agent prompt.
- 	rajectories/: harmonizes rollout records into JSONL/CSV/Parquet outputs.

---

## Repository Structure

### Root

- ARCHITECTURE.md — this document.
- README.md — usage instructions and experiment workflow.
- main.py — a convenience CLI wrapper for common experiments.
- 
equirements.txt — Python dependency list.

### configs/

- default_experiment.yaml — default researcher experiment settings.
- loader.py — loads YAML into SimulationConfig and related dataclasses.

### simulator/

- config.py — defines SimulationConfig, RewardConfig, ConstraintConfig, and OrchestratorMode.
- demand.py — generates demand from fixed paths, seeds, or random distributions.
- 
ode.py — implements SupplyChainNode, inventory, backlog, pipeline, and cost accounting.
- orchestrator.py — augments agent observations based on information-sharing regime.
- eer_game.py — core environment, weekly step logic, history tracking, bullwhip, and reward shaping.
- 
ewards.py — shaped reward function combining cost, backlog, and bullwhip penalties.

### gents/

- llm_agent.py — LLMAgent prompt creation, backend invocation, and order parsing.
- llm_backends.py — backend abstraction layer for Ollama, Groq, and vLLM.
- constraints.py — optional run-time guardrails for orders.

### 	ools/

- inventory_tool.py — deterministic EOQ-like recommendation used to augment prompts.

### policies/

- ase_stock.py — simple base-stock policy for classical experiments.
- moving_average.py — moving average demand-based ordering policy.
- 
andom_policy.py — random order policy for baseline variability.

### evaluation/

- 
epeated_runs.py — repeated-run experiment engine and report exporter.
- compare_models.py — multi-model evaluation runner.
- enchmark.py — YAML-driven benchmark entry point.
- plotting.py — generation of research plots and figure-specific visualizations.
- comparison_plots.py — model comparison plot utilities.

### experiments/

- llm_experiment.py — single-run LLM experiment with CSV export and plots.
- aseline_experiment.py — classical policy baseline experiments.
- 
un_majority_vote.py — majority-vote LLM experiment orchestration.
- 
un_figure2.py / 
un_figure3.py — wrappers to generate research figure plots.
- smoke_test.py / 	est_llm_agent.py / 	est_state_api.py — validation and integration checks.

### 	rajectories/

- schema.py — standard RL-style trajectory schema and conversion helpers.
- writer.py — exports trajectories to JSONL, CSV, and Parquet.

### scripts/

- groq_smoke_test.py — Groq SDK connectivity check.
- groq_smoke_test_postproc.py — tests Groq backend output normalization.
- list_model.py — lists Groq models using API key.

---

## Agent Architecture

### LLMAgent design

LLMAgent is the system-level wrapper for LLM-driven decision making.

- It receives the agent name, chosen model, backend selection, and prompt configuration.
- It constructs a prompt from local state and optional shared information.
- It delegates inference to gents.llm_backends.
- It parses the response to extract a safe integer order.
- It returns metadata such as 	ool_order, llm_order, and difference.

### Backend abstraction layer

The architecture separates agent logic from inference provider implementations.

- BaseLLMBackend defines generate(prompt, **options).
- OllamaBackend uses local Ollama HTTP generation.
- GroqBackend uses the Groq SDK and normalizes responses.
- VLLMBackend uses an OpenAI-compatible chat client for local vLLM.

### Ollama backend

The default backend. It posts to http://localhost:11434/api/generate with parameters:

- model
- prompt
- options.temperature
- options.num_predict
- keep_alive

Returns the trimmed text response.

### Groq backend

GroqBackend is compatible with the groq SDK when installed.

- It accepts model_name, pi_key, and optional pi_url.
- It calls chat.completions.create(...) with a restrictive system prompt.
- It normalizes SDK responses, handling object and dict forms.
- It cleans reasoning artifacts, fenced code, and ellipses.

### vLLM backend

VLLMBackend wraps a local OpenAI-compatible endpoint.

- Requires ase_url and uses openai.OpenAI client.
- Sends a chat completion request with a system prompt to return only a number.
- Parses the returned content directly.

### Decision generation workflow

The workflow is:

1. BeerGame computes get_agent_state(name).
2. LLMAgent.build_prompt(state) adds prompting context.
3. LLMAgent.query_model(prompt) calls the backend.
4. LLMAgent.parse_order(response_text) extracts the integer.
5. LLMAgent._clamp_order(value) enforces [0, max_order].
6. gents.constraints.apply_constraints() post-processes the order.

### Constraint enforcement

gents.constraints.apply_constraints() can impose:

- order_cap
- udget_limit
- safety_stock_min
- panic_order_threshold / panic_max_order
- order_smoothing_alpha

This happens after model output and before the order enters the simulation.

### Order extraction and validation

LLMAgent.parse_order() supports multiple patterns:

- explicit labeled quantities (order:, nswer:)
- natural-language phrases with numbers
- fenced code blocks containing a number
- the last standalone integer found
- negative numbers clamped to zero

It ensures final output is within [0, max_order].

---

## Tool-Augmented Decision Support

### Tool architecture

The tool subsystem is intentionally small and transparent.

	ools/inventory_tool.py exports a single function:

- eoq_recommendation(state) — deterministic base-stock order recommendation.

### eoq_recommendation()

The current implementation uses:

`python
forecast = int(state.get('last_customer_demand', 0) or 0)
lead_time = int(state.get('lead_time', 2) or 2)
inventory = int(state.get('inventory', 0) or 0)
backlog = int(state.get('backlog', 0) or 0)
return max(0, target - inventory + backlog)
`

It is designed as audit-friendly support rather than a black-box optimization.

### use_tool_recommendation flow

When enabled in SimulationConfig.use_tool_recommendation:

1. LLMAgent._with_tool_recommendation(state) adds 	ool_order to the state.
2. LLMAgent.build_prompt() inserts a Tool Recommendation: block.
3. generate_order or generate_order_majority_vote executes with the augmented state.
4. last_decision_metadata records the deviation from the tool.

### Prompt injection

The prompt includes the explicit recommendation text:

- Tool Recommendation:
- Order {tool_order} units.
- You may follow or ignore this recommendation.

This creates a structured decision-support signal for the LLM.

### Differences between Tool ON and Tool OFF experiments

- Tool OFF: agent prompt contains only state and orchestrator data.
- Tool ON: prompt includes an additional recommendation block and metadata.

This allows researchers to compare whether providing a heuristic anchor changes the stability or reliability of LLM orders.

---

## Simulation Engine

### Beer Game environment

BeerGame implements a four-node supply chain with the following sequence each week:

1. Receive pipeline shipments into inventory.
2. Generate customer demand at the retailer.
3. Fulfill demand downstream.
4. Advance shipments through the FIFO pipelines.
5. Add factory production to the factory pipeline.
6. Place new orders from agents or policies.
7. Compute holding/backlog costs and shaped reward.
8. Record history, trajectories, and metrics.

### Inventory state tracking

Each SupplyChainNode tracks:

- inventory
- acklog
- FIFO incoming_shipments
- last_order
- order_history
- accumulated 	otal_holding_cost, 	otal_backlog_cost, 	otal_cost

BeerGame maintains system-wide history arrays for demand, orders, inventory, backlog, cost, bullwhip, and consensus gaps.

### Order processing

Orders are placed at the end of each week after shipments and demand fulfillment.

- The retailer responds to customer demand.
- Upstream echelons respond to the downstream last order.
- The factory's outgoing shipment is equal to its own last order.

### Shipment flow

- The output of each node becomes the incoming shipment for the next downstream node.
- Player orders are not immediately delivered; they enter the upstream pipeline and arrive after lead time.
- The queue is implemented with a collections.deque sized by lead_time.

### Cost calculation

- Holding cost = inventory * holding_cost
- Backlog cost = acklog * backlog_cost
- SupplyChainNode.compute_costs() accumulates both.
- BeerGame.step() sums system cost and stores it in history.

### Demand generation

simulator.demand.DemandGenerator supports:

- fixed demand paths via ixed_demand_path
- seeded reproducible random demand via demand_seed
- pure random demand when no seed or path is provided

The repository includes the classic MIT pattern from mit_beer_game_demand_path().

### Weekly simulation loop

The environment loop is tightly ordered and deterministic given the same inputs:

1. 
eceive_shipment() for each node
2. generate_customer_demand()
3. ulfill_demand() for retailer through factory
4. dd_incoming_shipment() to downstream pipelines
5. place_order() for each node
6. compute_costs() and reward shaping
7. _record_history() and compute_bullwhip()
8. env.week += 1
9. done = week >= max_weeks
10. return 
ext_state, reward, done, info

---

## Orchestrator Modes

### Decentralized

- Local observation only
- No shared customer demand or system status
- Research hypothesis: purely decentralized agents will amplify the bullwhip effect most strongly

### Demand Sharing

- Agents receive shared_current_demand
- Only the current customer demand is shared
- Hypothesis: demand visibility may reduce but not eliminate instability

### History Sharing

- Agents receive:
  - shared_current_demand
  - shared_demand_history
  - shared_demand_volatility
- Builds on demand sharing with a short demand history window
- Hypothesis: richer shared history improves coordination and reduces variance

### Centralized

- Agents receive:
  - current demand
  - recent demand history
  - system_total_backlog
  - system_total_inventory
  - echelon_snapshot
- Hypothesis: full system visibility approximates a centralized controller

### Negotiation

- Uses centralized shared information plus proposal exchange.
- Run in two phases:
  1. agents propose orders in a first pass
  2. agents see all proposals and may revise
- Supports 
egotiation_proposals in prompt context.
- Designed to study whether explicit coordination reduces order dispersion.

---

## Experiment Pipeline

### Core scripts

- main.py — convenience CLI for tests and experiments.
- evaluation/repeated_runs.py — core repeated-run experiment engine.
- experiments/llm_experiment.py — single-run LLM-driven experiment with weekly logging.
- experiments/baseline_experiment.py — classical policy comparison.
- experiments/run_majority_vote.py — majority-vote sample-based LLM experiment.
- evaluation/compare_models.py — model comparison across repeated runs.
- evaluation/benchmark.py — YAML-driven benchmark runner.
- experiments/run_figure2.py / 
un_figure3.py — figure generation wrappers.

### Figure 2 generation

- Uses evaluation.plotting.generate_bullwhip_boxplots()
- Reads 
esults/.../*.jsonl trajectory files
- Produces per-week boxplots across echelons

### Figure 3 generation

- Uses evaluation.plotting.generate_figure3_boxplots()
- Compares two majority-vote result folders
- Produces side-by-side boxplots for sample sizes 10 and 100

### Repeated-run evaluation

- evaluation/repeated_runs.py runs 
_runs episodes with the same demand path
- For LLM experiments, it optionally builds LLMAgent objects for each echelon
- Stores episode history, trajectories, and summary reports
- Writes 
epeated_runs_report.json, 
un_costs.csv, and trajectories files

### Reliability experiments

- Designed to isolate LLM stochasticity with repeated runs
- Computes cost variability, event counts, and cross-run amplification
- Uses the same environment and demand path across runs

### Tool experiments

- Controlled via SimulationConfig.use_tool_recommendation
- Runs may compare tool-assisted prompt context against raw LLM decisions

---

## Metrics and Evaluation

### Bullwhip metrics

- metrics/bullwhip.bullwhip_ratio() computes Var(orders) / Var(demand).
- metrics/bullwhip.bullwhip_per_agent() computes this ratio for each echelon.
- BeerGame.compute_bullwhip() returns per-agent and an overall average.

### Cost metrics

- metrics.cost_analysis.cost_summary() computes mean, std, median, min/max, and confidence intervals.
- evaluation/repeated_runs.py stores 	otal_costs and per-run summaries in CSV.

### Ψ (Psi) metrics

- metrics.agent_bullwhip.psi_ratio() computes run-to-run cross-echelon amplification:
  Ψ_k(t) = Var_r(q_{k,t}) / Var_r(q_{k-1,t})
- Aggregated in gent_bullwhip_report across repeated runs.

### Φ (Phi) metrics

- metrics.agent_bullwhip.phi_ratio() computes temporal amplification:
  Φ_k(t) = Var_r(q_{k,t+1}) / Var_r(q_{k,t})
- Mean values and counts of amplification events are reported.

### Consensus metrics

- BeerGame._record_history() computes consensus_gap as max(order_vector) - min(order_vector).
- evaluation.repeated_runs._summarize_consensus() reports mean and max consensus gaps.

### Reliability measurements

- metrics.reliability.coefficient_of_variation() measures cost variability.
- 
un_to_run_instability measures total-cost variance across runs.
- 	ail_event_rate() counts extreme-cost runs above the 90th percentile.
- Inventory collapse and backlog explosion detectors identify risk events.

### Stability metrics

- metrics.stability.order_variance() and inventory_variance() track oscillation magnitude.
- cumulative_instability() aggregates order, inventory, and backlog variances.

---

## Data Flow

### Physical supply-chain data flow

`mermaid
flowchart LR
    Customer[Customer Demand] --> Retailer[Retailer]
    Retailer --> Wholesaler[Wholesaler]
    Wholesaler --> Distributor[Distributor]
    Distributor --> Factory[Factory]
    Retailer -->|incoming shipment| downstream
    Wholesaler -->|incoming shipment| downstream
    Distributor -->|incoming shipment| downstream
    Factory -->|production| Distributor
    subgraph Metrics
      Retailer -->|orders/inventory/backlog| Metrics
      Wholesaler -->|orders/inventory/backlog| Metrics
      Distributor -->|orders/inventory/backlog| Metrics
      Factory -->|orders/inventory/backlog| Metrics
    end
`

### Software execution flow

`mermaid
flowchart TD
    CLI[CLI / main.py / scripts] --> Config[Load YAML or CLI args]
    Config --> Init[Initialize SimulationConfig & Agents]
    Init --> Sim[BeerGame Simulation Loop]
    Sim --> Metrics[Metrics Calculation]
    Metrics --> Output[Write JSON/CSV/Plots/Trajectories]
`

### Detailed step flow

1. CLI selects experiment type and loads configuration.
2. SimulationConfig is built from configs/default_experiment.yaml or CLI args.
3. BeerGame and, if needed, LLMAgent objects are created.
4. The loop collects local or shared agent states.
5. Agents/policies generate orders.
6. BeerGame.step() applies orders and advances shipments.
7. History, trajectories, and metrics are recorded.
8. Experiment runners export reports and plots.

---

## Backend Architecture

### Ollama

- Default inference provider.
- Local server expected at http://localhost:11434.
- Used by LLMAgent when ackend='ollama'.
- experiments/llm_experiment.py, evaluation/repeated_runs.py, and evaluation/compare_models.py default to Ollama.

### Groq

- Optional remote backend via the groq SDK.
- gents.llm_backends.GroqBackend performs response normalization and post-processing.
- The repository includes smoke test scripts:
  - scripts/groq_smoke_test.py
  - scripts/groq_smoke_test_postproc.py
- experiments/run_majority_vote.py supports --backend groq.

### vLLM

- Optional backend for OpenAI-compatible local serving.
- Implemented in gents.llm_backends.VLLMBackend.
- Requires a compatible ase_url and the openai client.
- Supported by the same LLMAgent abstraction.

### Request lifecycle

`mermaid
sequenceDiagram
    participant Runner as Experiment Runner
    participant Env as BeerGame
    participant Agent as LLMAgent
    participant Backend as LLMBACKEND
    participant Model as Inference Server

    Runner->>Env: reset()
    Runner->>Env: get_agent_state(name)
    Env-->>Agent: observation
    Agent->>Backend: generate(prompt)
    Backend->>Model: request
    Model-->>Backend: response
    Backend-->>Agent: text
    Agent->>Runner: order
    Runner->>Env: step(actions)
    Env-->>Runner: next_state/info
`

---

## Output Artifacts

### JSON reports

- 
esults/repeated_runs/repeated_runs_report.json
  - configuration metadata
  - cost statistics
  - reliability measures
  - agent bullwhip metrics
  - consensus statistics

### CSV outputs

- 
esults/repeated_runs/run_costs.csv
  - per-run total cost time series
- 
esults/llm_experiment_results.csv
  - weekly experiment metrics and action data
- 
esults/model_comparison.csv and _summary.csv
  - model comparison results

### Plots

- plots/research/*.png — repeated-run research visualizations.
- plots/figure2_bullwhip_boxplots.png — order distribution boxplots.
- plots/figure3_majority_vote_boxplots.png — majority-vote comparison plots.
- plots/comparison/*.png — multi-model comparison figures.

### Trajectory logs

- 
esults/.../trajectories/rollouts.jsonl
- 
esults/.../trajectories/rollouts.csv
- 
esults/.../trajectories/rollouts.parquet (optional)

Trajectory records include:

- week
- gent
- state
- ction
- 
eward
- 
ext_state
- 	ool_order
- llm_order
- difference
- consensus_gap
- 
egotiation_proposals
- cost
- ullwhip

---

## Current Research Extensions

The repository adds research-specific modifications beyond a textbook Beer Game:

1. Tool-augmented prompts through 	ools/inventory_tool.py.
2. Backend abstraction for Ollama, Groq, and vLLM.
3. Majority-vote sampling for robust LLM decisions.
4. Negotiation mode with two-stage proposal exchange.
5. Repeated-run reliability experiments isolating decision variance.
6. Agent-level Ψ and Φ variance metrics.
7. Standardized trajectory export for JSONL/CSV/Parquet.
8. YAML-driven benchmark configuration and plot generation.

---

## Discrepancies Corrected

- The system supports Ollama, Groq, and vLLM backends, not Ollama only.
- The orchestrator mode enum values are actual values found in simulator.config.OrchestratorMode.
- evaluation/repeated_runs.py includes negotiation and tool support workflows which were not fully documented previously.
- experiments/run_majority_vote.py and figure generation wrappers are part of the current implementation.
- There is no 
un_table1.py in the current repository.

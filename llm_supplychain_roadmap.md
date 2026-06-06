# From Time Series to Smart Ordering: Your LLM-Orchestrated Supply Chain Roadmap

> A plain-English, in-depth guide for researchers building an LLM agent that reads forecasts from multiple supply chain echelons and decides which ordering policy to use.

---

## Part 0 — Your Idea, Validated (and Where the Gaps Are)

### What You Said

You want to:
1. Run **different time series models at different echelons** (retailer, warehouse, distributor, factory)
2. Feed their outputs into an **LLM acting as an orchestrator**
3. The LLM **communicates with agents** at each echelon and **picks an ordering policy** (OUT, POUT, EOQ, etc.) based on what it sees

### Is This Sound?

**Yes — and it's a genuine research gap.** The existing literature (InvAgent 2024, AIM-Bench 2025, Jannelli et al. 2025) puts LLM agents in the *ordering* role but gives them raw demand numbers, not pre-processed forecasts. Nobody has systematically studied what happens when you give the LLM **echelon-specific forecast signals** and let it reason about *which policy structure* to deploy. That is your novelty.

### What You Got Right

| Your Assumption | Verdict |
|---|---|
| Different echelons need different forecasting approaches | ✅ Correct — upstream echelons see noisier, more autocorrelated signals |
| The LLM can act as an orchestrator between agents | ✅ Correct — this is exactly the LangGraph / multi-agent pattern |
| The LLM can choose among ordering policies dynamically | ✅ Plausible — but it needs structured prompts + policy definitions to do this well |
| Policy choice affects bullwhip behavior | ✅ Core finding of Dejonckheere et al. (2003) |

### What You Are Missing (Honest Gaps)

| Gap | Why It Matters |
|---|---|
| **State management between echelons** | The LLM needs to know *current* inventory, pipeline orders, and lead times — not just forecasts. You need a shared state object. |
| **Prompt engineering for policy selection** | LLMs anchor on mean demand (AIM-Bench 2025). Without structured prompts, they'll just mimic a human's panic ordering. |
| **Lead time signals as a separate input** | Stochastic lead times are a separate cause of bullwhip. Your time series layer needs to forecast lead times too, not just demand. |
| **Feedback loops** | After the LLM picks a policy and orders are placed, the next period's state changes. Your system needs a loop, not a one-shot call. |
| **Evaluation metric** | How do you know if the LLM made a *good* choice? You need to track bullwhip ratio (Var(Orders)/Var(Demand)) and inventory cost per period. |
| **LangGraph / CrewAI as the wiring layer** | Without a framework to manage agent memory, routing, and state, your system is just disconnected API calls. |
| **RAG for policy knowledge** | The LLM doesn't inherently know what POUT is. You need to give it policy definitions via RAG or system prompts. |
| **Fallback / guardrails** | LLMs hallucinate. You need a rule-based safety net (e.g., "never order more than 3× average demand") to cap bad decisions. |

---

## Part 1 — The Supply Chain: What Are Echelons?

Think of a supply chain as a relay race. The customer demand starts at the finish line (the retailer) and the signal passes backward through:

```
Customer → Retailer → Warehouse → Distributor → Factory → Supplier
    (echelon 1)  (echelon 2)   (echelon 3)   (echelon 4)
```

Each player only sees what's happening in front of them. The retailer sees real customer demand. The warehouse only sees the retailer's *orders* (which are already distorted). By the time you reach the factory, the signal looks nothing like actual customer demand — it's amplified, noisy, and delayed.

This amplification is the **Bullwhip Effect**.

**Why does this matter for your time series layer?**
- Echelon 1 (retailer) demand is relatively clean — good for ARIMA, Prophet
- Echelon 2 (warehouse) sees lumpy, batched orders — LSTM handles this better
- Echelon 3+ (factory/supplier) see very noisy, autocorrelated signals — transformer models or ARMA with higher lag terms

---

## Part 2 — Time Series Models (One Per Echelon)

### 2.1 What Is a Time Series?

A time series is just a sequence of numbers recorded over time. "How many units did we sell each week for the past 2 years?" When you *forecast* it, you predict the next few weeks.

### 2.2 The Models You Should Know

#### ARIMA — for clean, structured demand (Echelon 1)
**AR**imo**I**ntegrated **M**oving **A**verage. Three knobs:
- **AR (p)**: How much does today's demand depend on *yesterday's* demand? (autocorrelation)
- **I (d)**: Is the series drifting upward or downward? (differencing to make it stationary)
- **MA (q)**: How much does today's demand depend on *yesterday's forecast error*?

**Bullwhip link**: Lee et al. (1997) and Chen et al. (2000) use AR(1) demand as the baseline for all their bullwhip math. ARIMA is the foundation of classical supply chain theory.

**Best for**: Retailer echelon. Clean, stationary or mildly trending demand. Short history.

**Limitation**: Assumes linear relationships. Can't handle sudden demand shocks.

---

#### Exponential Smoothing (ETS) — the practitioner's favorite
Instead of averaging all past data equally, exponential smoothing gives more weight to recent observations. The weight of old data decays *exponentially*.

- **Simple ES**: Good for demand with no trend
- **Holt's method**: Adds a trend component
- **Holt-Winters**: Adds seasonality

**Bullwhip link**: Disney & Lambrecht (2008) show that exponential smoothing *inside* an OUT policy is a primary cause of bullwhip — the smoothing parameter α directly controls how much order variance you generate. This is why your LLM orchestrator's choice of forecasting method and ordering policy are **coupled**.

**Best for**: Echelon 1-2. Stationary or mildly seasonal demand.

---

#### LSTM (Long Short-Term Memory) — for complex, nonlinear demand (Echelon 2-3)
LSTMs are a type of neural network that has a *memory*. They can learn that "high demand 3 weeks ago usually predicts a spike now" without you telling them explicitly.

Think of it like this: ARIMA is a rule you write down. LSTM is a pattern it figures out on its own.

**Architecture in one sentence**: The network has gates (input, forget, output) that decide what to remember and what to forget from the sequence so far.

**Bullwhip link**: At upstream echelons, demand is autocorrelated in complex ways. LSTM captures long-range dependencies that ARIMA misses.

**Best for**: Echelon 2-3. Lumpy, nonlinear, or seasonally complex demand. Longer history needed (weeks/months of data).

**Limitation**: Black box — you can't easily explain *why* it made a prediction.

---

#### Temporal Fusion Transformer (TFT) — the modern choice for multi-variate signals
Transformers use *attention* — they look at all past time steps simultaneously and learn which ones matter most, instead of processing step-by-step like LSTMs.

TFT (Lim et al., 2021) is specifically designed for forecasting. It can take:
- Historical demand
- Known future information (holidays, planned promotions)
- Static features (product category, echelon ID)

**Best for**: Echelon 2-4 with rich feature sets. Works well when you have multiple products or multiple echelons sharing a model.

---

#### Stochastic Lead Time Forecasting — the missing piece most people forget
Lead times are themselves a time series: "How long did our supplier take each week?" You can model this with:
- **ARMA models** for correlated lead times (Michna, Disney & Nielsen, 2020)
- **Empirical distributions** (parametric or kernel density)
- **LSTM** if lead times have complex patterns

**Why it matters**: Michna et al. identified lead time forecasting as a *separate cause of bullwhip* beyond demand forecasting. If your LLM orchestrator doesn't get a lead time signal, it's missing a key input.

---

### 2.3 What to Feed to the LLM

Don't give the LLM raw numbers. Give it a **structured summary** of what the time series model found:

```json
{
  "echelon": "warehouse",
  "forecast_model": "LSTM",
  "demand_next_period": 340,
  "demand_uncertainty": "high",
  "trend": "slightly_upward",
  "seasonality_detected": false,
  "lead_time_forecast": 4.2,
  "lead_time_variance": "high",
  "autocorrelation": 0.72,
  "recommended_safety_stock_buffer": "increase"
}
```

Now the LLM has *interpreted* information it can reason about, not just a number.

---

## Part 3 — Ordering Policies (What the LLM Picks From)

Think of these as the "rules of the game" for deciding how much to order each period.

### 3.1 Order-Up-To (OUT) / Base-Stock Policy
**The idea**: Every period, check your inventory position (on-hand + orders in transit). Order enough to bring it up to a fixed target level S.

**Mathematically**: `Order_t = S - (Inventory_t + Pipeline_t)`

**When it generates bullwhip**: When the target S is updated using an exponential smoothing forecast. Every time demand jumps, S jumps, and so do orders — amplifying the signal.

**When it doesn't**: Under i.i.d. demand with MMSE (minimum mean-squared-error) forecasting, OUT just passes on orders without amplification. It becomes lot-for-lot.

**LLM should choose this when**: Demand is relatively stable, lead times are short and deterministic, and there's little autocorrelation.

---

### 3.2 Proportional Order-Up-To (POUT) — the bullwhip dampener
**The idea**: Like OUT, but instead of filling the *entire* gap between target and current inventory, you only fill a *fraction* of it.

`Order_t = Demand_forecast + α × (S - Inventory_t) + β × (Pipeline_target - Pipeline_t)`

Where α and β are fractions (0 to 1). Setting α=1, β=1 gives you the standard OUT policy.

**Why it helps**: Smooths order swings. A demand spike doesn't cause a proportionally giant order — just a nudge.

**Tradeoff**: Slower to recover from genuine demand level changes. Higher risk of stockouts during real demand surges.

**LLM should choose this when**: High bullwhip risk is detected (high autocorrelation, upstream echelon, high lead time variance), and the system can tolerate slightly higher stockout risk to avoid massive amplification.

---

### 3.3 Lot-for-Lot (Pass-On-Orders)
**The idea**: Just order exactly what you sold. No smoothing, no safety stock adjustments.

**When it's optimal**: Under i.i.d. demand (no autocorrelation) with MMSE forecasting and deterministic lead times — mathematically proven to be bullwhip-neutral.

**When it fails badly**: Any autocorrelated or trending demand turns this into a disaster, as you never build enough buffer.

**LLM should choose this when**: Demand is verified i.i.d., lead times are deterministic and short, and the echelon is the retailer (close to real demand).

---

### 3.4 Economic Order Quantity (EOQ) / (s, Q) Policy
**The idea**: Fixed reorder point s, fixed order size Q derived from minimizing holding + ordering costs.

`Q = √(2 × Demand × Ordering_Cost / Holding_Cost)`

**Bullwhip problem**: Ordering in fixed batches creates artificial demand "peaks" and "troughs" that have nothing to do with real customer demand. This is the *order batching* cause of bullwhip (Lee et al. 1997).

**When it makes sense**: When ordering costs are high and demand is relatively stable. Less relevant for continuous-review settings with digital ordering.

**LLM should choose this when**: Lead times are long, ordering costs are significant, and demand at the echelon is stable enough to justify batch consolidation.

---

### 3.5 Min-Max (s, S) Policy
**The idea**: Do nothing until inventory drops below s, then order up to S in one shot.

**Bullwhip behavior**: Creates lumpy orders — long quiet periods then a big burst. Amplifies variance at upstream echelons.

**LLM should choose this when**: Products have long shelf life, demand is very low and sporadic, and ordering costs are very high relative to holding costs.

---

### 3.6 Full-State Feedback OUT
**The idea**: Instead of just looking at current inventory gap, incorporate the full statistical structure of the demand process into the order computation.

If demand is ARMA(p,q), the order formula uses all p past demands and q past forecast errors in an optimal linear combination (Gaalman & Disney, 2009).

**Why it's powerful**: Theoretically optimal — can eliminate bullwhip for *any* linear demand process.

**Why it's hard**: You need to know the demand process structure. In practice, you estimate it — estimation error reintroduces bullwhip.

**LLM should trigger this when**: Time series analysis has confidently identified an ARMA(p,q) structure at a specific echelon, and the LLM can route the appropriate parameters to a solver.

---

## Part 4 — The LLM as Orchestrator

### 4.1 What "Orchestrator" Means

An orchestrator is a conductor in an orchestra. It doesn't play an instrument — it decides who plays what, when, and how loud.

In your system:
- **Instruments** = echelon agents (each with their time series + inventory state)
- **Conductor** = the LLM
- **Score** = the ordering policy the LLM picks

The LLM reads the forecast summaries from all echelons, detects patterns (high bullwhip risk? demand surge? lead time disruption?), and tells each echelon which policy to run.

### 4.2 What the LLM Actually Does (Step by Step)

```
Period t begins:
│
├─ [Echelon 1 Agent] runs ARIMA on retailer demand
│   └─ outputs: {forecast: 120, trend: stable, autocorr: 0.1}
│
├─ [Echelon 2 Agent] runs LSTM on warehouse orders
│   └─ outputs: {forecast: 180, trend: rising, autocorr: 0.6}
│
├─ [Echelon 3 Agent] runs TFT on distributor orders
│   └─ outputs: {forecast: 260, trend: rising, autocorr: 0.8, lead_time_var: high}
│
└─ [LLM Orchestrator] receives all three summaries
    ├─ Detects: upstream autocorrelation increasing → bullwhip risk HIGH
    ├─ Detects: lead time variance HIGH at echelon 3 → extra safety stock needed
    ├─ Decision: Echelon 1 → LOT-FOR-LOT (stable, no risk)
    │            Echelon 2 → POUT with α=0.4 (moderate smoothing)
    │            Echelon 3 → POUT with α=0.2 + increased safety stock
    └─ Sends policy parameters back to each agent
```

### 4.3 Chain-of-Thought Prompting

The LLM works better when you make it *reason step by step* before deciding. Example system prompt:

```
You are a supply chain orchestrator. For each echelon, you will receive:
- Demand forecast and uncertainty
- Detected autocorrelation level
- Lead time forecast and variance
- Current inventory position

Step 1: Assess bullwhip risk at each echelon (low/medium/high)
Step 2: Identify the primary cause (demand autocorrelation, lead time variance, order batching)
Step 3: Select the ordering policy that best addresses the identified cause
Step 4: Set the smoothing parameters (α, β for POUT) based on risk level
Step 5: Output your decisions as structured JSON

Policy options: [LOT_FOR_LOT, OUT, POUT, EOQ, MIN_MAX]
```

Forcing structured reasoning dramatically reduces the anchoring bias identified in AIM-Bench (2025).

---

## Part 5 — The Tech Stack You Need

### 5.1 LangGraph — Your Missing Piece

LangGraph is a Python framework for building multi-agent systems as **graphs**, where:
- **Nodes** = individual agents (or the LLM orchestrator)
- **Edges** = the flow of information between them
- **State** = a shared object that persists across all nodes

Think of it like a flowchart that can loop and has memory.

**Why you need it**:
- Without LangGraph, your agents are isolated. They can't share inventory state across periods.
- LangGraph gives you **conditional routing** (if bullwhip risk = high, go to POUT node)
- It gives you **persistence** — the system remembers last period's orders and inventory

**Basic structure for your system**:

```python
from langgraph.graph import StateGraph

# Define shared state
class SupplyChainState(TypedDict):
    echelon_forecasts: dict      # from time series agents
    inventory_positions: dict    # current stock at each echelon
    pipeline_orders: dict        # orders in transit
    current_policies: dict       # what policy each echelon is using
    orchestrator_reasoning: str  # LLM's chain-of-thought

# Build the graph
graph = StateGraph(SupplyChainState)
graph.add_node("echelon_1_forecast", echelon1_agent)
graph.add_node("echelon_2_forecast", echelon2_agent)
graph.add_node("echelon_3_forecast", echelon3_agent)
graph.add_node("orchestrator", llm_orchestrator)
graph.add_node("execute_orders", order_execution_agent)
graph.add_node("update_inventory", inventory_update_agent)

# Wire them up
graph.add_edge("echelon_1_forecast", "orchestrator")
graph.add_edge("echelon_2_forecast", "orchestrator")
graph.add_edge("echelon_3_forecast", "orchestrator")
graph.add_edge("orchestrator", "execute_orders")
graph.add_edge("execute_orders", "update_inventory")
graph.add_edge("update_inventory", "echelon_1_forecast")  # next period loop
```

---

### 5.2 RAG — Give the LLM Policy Knowledge

The LLM doesn't inherently know what POUT is, or when stochastic lead times suggest using it over OUT. You need to **inject that knowledge at runtime**.

**RAG (Retrieval-Augmented Generation)** works like this:
1. You build a knowledge base of supply chain policy descriptions (from papers, your own notes)
2. When the LLM gets a scenario, it queries: "what policies reduce bullwhip under high autocorrelation?"
3. The relevant policy descriptions are retrieved and included in the LLM's prompt
4. The LLM reasons with this grounded knowledge

**What to put in your RAG knowledge base**:
- Descriptions of each policy (OUT, POUT, EOQ, etc.) with their conditions
- Excerpts from Dejonckheere et al. (2003) on when POUT eliminates bullwhip
- Rules linking time series characteristics to policy recommendations
- Historical decisions and their outcomes (for iterative improvement)

**Tools**: LangChain's vector store integrations (ChromaDB, Pinecone), with `text-embedding-3-small` for embeddings.

---

### 5.3 Tool Use — Let the LLM Call Solvers

Instead of the LLM computing optimal POUT parameters in its head (it'll be wrong), give it **tools** it can call:

```python
@tool
def compute_pout_parameters(autocorrelation: float, lead_time: float, lead_time_variance: float) -> dict:
    """Computes optimal α and β for POUT policy given demand and lead time characteristics."""
    # Mathematical solver based on Disney & Lambrecht (2008)
    alpha = 1 / (1 + lead_time * autocorrelation)
    beta = alpha * 0.5  # simplified
    return {"alpha": alpha, "beta": beta}

@tool
def compute_bullwhip_ratio(order_history: list, demand_history: list) -> float:
    """Computes Var(Orders)/Var(Demand) — the bullwhip ratio."""
    return np.var(order_history) / np.var(demand_history)
```

The LLM calls these tools when reasoning, gets back precise numbers, and uses them in its decision. This is *hybrid* intelligence — LLM for qualitative reasoning, math for quantitative precision.

---

### 5.4 Memory — What the LLM Needs to Remember

LLMs have no memory between calls unless you give it to them. You need:

**Short-term memory (within a simulation run)**:
- Last 5-10 periods of orders, demand, and inventory at each echelon
- Last period's policy choice and its outcome

**Long-term memory (across simulation runs)**:
- Which policy worked best under which demand regime
- This is where your system starts to *learn* over time

**Implementation**: LangGraph's built-in checkpointing stores state per-period. For long-term, a simple SQLite or Redis store works fine for research.

---

### 5.5 Full Tech Stack Summary

| Layer | What It Does | Tool |
|---|---|---|
| Time series | Forecast demand + lead time at each echelon | `statsmodels` (ARIMA), `PyTorch`/`Keras` (LSTM), `pytorch-forecasting` (TFT) |
| Orchestration | Route agents, manage state, control loops | **LangGraph** |
| LLM reasoning | Policy selection, chain-of-thought | Anthropic Claude / GPT-4 via API |
| Policy knowledge | Ground LLM with supply chain theory | **RAG** via LangChain + ChromaDB |
| Math solvers | Compute policy parameters precisely | Custom Python tools / SciPy |
| Guardrails | Cap bad decisions | Rule-based filters in LangGraph conditional edges |
| Evaluation | Track bullwhip ratio + cost per period | Custom metrics module |
| Simulation environment | Multi-echelon inventory simulator | `OR-Gym`, custom Python, or adapted Beer Game env |

---

## Part 6 — Reading Roadmap

Work through these in order. Each section builds on the previous.

### Stage 1 — Supply Chain Foundations (2-3 weeks)
Read these to understand *what you're actually trying to solve*:

1. **Lee, Padmanabhan & Whang (1997)** — "Information Distortion in a Supply Chain: The Bullwhip Effect" (*Management Science*) — The foundational paper. Must read.
2. **Lee, So & Tang (2000)** — "The Value of Information Sharing in a Two-Level Supply Chain" — Why sharing demand info helps.
3. **Dejonckheere, Disney, Lambrecht & Towill (2003)** — "Measuring and avoiding the bullwhip effect: A control theoretic approach" (*EJOR*) — Introduces POUT and shows mathematically how to eliminate bullwhip.
4. **Disney & Lambrecht (2008)** — "On Replenishment Rules, Forecasting, and the Bullwhip Effect in Supply Chains" (*Foundations and Trends in TIO Management*) — The comprehensive policy reference. Dense but worth it.

---

### Stage 2 — Stochastic Lead Times (1-2 weeks)
5. **Wang & Disney (2017)** — "Mitigating variance amplification under stochastic lead-time: The proportional control approach" (*EJOR*) — POUT under stochastic lead times.
6. **Michna, Disney & Nielsen (2020)** — "The impact of stochastic lead times on the bullwhip effect under correlated demand" — Lead time forecasting as a cause of bullwhip. Your key gap.
7. **Disney et al. (2016)** — Global supply chains with stochastic lead times and order crossover effects.

---

### Stage 3 — Time Series for Supply Chain (2 weeks)
8. **Box, Jenkins, Reinsel — "Time Series Analysis" (textbook)** — The ARIMA bible. Read Chapters 1-5.
9. **Hyndman & Athanasopoulos — "Forecasting: Principles and Practice" (free online)** — Covers ETS, ARIMA, practical decomposition. Excellent.
10. **Lim et al. (2021)** — "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" — The TFT paper. Read after you're comfortable with LSTM basics.
11. The comparative paper on **ARIMA vs LSTM vs Prophet** (Preprints.org, 2026) — Synthesizes when each model wins.

---

### Stage 4 — LLM Agents in Supply Chain (2 weeks)
12. **InvAgent (Quan & Liu, 2024)** — arXiv:2407.11384 — First major LLM multi-agent inventory system. Start here.
13. **AIM-Bench (Zhao et al., 2025)** — arXiv:2508.11416 — Benchmarks LLM biases in inventory decisions. Shows what goes wrong.
14. **Jannelli et al. (2025)** — "Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking" (*IJPR*) — Multi-agent consensus, bullwhip experiments with LLM agents.
15. **LLMs for Supply Chain Management (arXiv:2505.18597, 2025)** — Good survey of where the field is.

---

### Stage 5 — Orchestration Tech (2 weeks)
16. **LangGraph documentation** — https://langchain-ai.github.io/langgraph/ — Work through the multi-agent tutorial.
17. **LangChain RAG documentation** — Build a simple RAG pipeline before integrating it with LangGraph.
18. **ReAct paper (Yao et al., 2022)** — "ReAct: Synergizing Reasoning and Acting in Language Models" — The foundational paper for tool-using LLM agents. Short, very important.
19. **Agentic RAG Survey (arXiv:2501.09136, 2025)** — When and how to use RAG inside agentic workflows.

---

### Stage 6 — Putting It Together (ongoing)
20. **OR-Gym (Hubbs et al., 2020)** — Open-source reinforcement learning environments for supply chain. Use this as your simulation base, then replace the RL agent with your LLM orchestrator.
21. **MARLIM (Multi-Agent RL for Inventory Management, 2023)** — arXiv:2308.01649 — Shows the RL baseline your system needs to beat (or complement).

---

## Part 7 — System Architecture (Full Picture)

```
┌─────────────────────────────────────────────────────────┐
│                   SIMULATION ENVIRONMENT                 │
│         (multi-echelon inventory, stochastic demand,     │
│                 stochastic lead times)                   │
└──────────────────────────┬──────────────────────────────┘
                           │ Period t state
          ┌────────────────▼──────────────────┐
          │         LANGGRAPH STATE            │
          │  {inventory, pipeline, forecasts,  │
          │   policies, period, history}       │
          └──┬──────────┬──────────┬──────────┘
             │          │          │
    ┌────────▼──┐ ┌─────▼────┐ ┌──▼──────────┐
    │ Echelon 1 │ │Echelon 2 │ │  Echelon 3  │
    │  Agent    │ │  Agent   │ │   Agent     │
    │ (ARIMA)   │ │ (LSTM)   │ │   (TFT)     │
    └────────┬──┘ └─────┬────┘ └──┬──────────┘
             │          │         │
             └──────────▼─────────┘
                        │ Structured forecast summaries
              ┌─────────▼──────────┐
              │   LLM ORCHESTRATOR  │◄── RAG: policy knowledge
              │                     │◄── Tools: math solvers
              │  Chain-of-thought:  │
              │  1. Assess risk     │
              │  2. Identify cause  │
              │  3. Select policy   │
              │  4. Set parameters  │
              └─────────┬──────────┘
                        │ Policy decisions (JSON)
              ┌─────────▼──────────┐
              │  ORDER EXECUTION    │
              │  (apply policies,   │
              │   place orders)     │
              └─────────┬──────────┘
                        │ Orders placed, time advances
              ┌─────────▼──────────┐
              │  INVENTORY UPDATE   │
              │  (receive shipments,│
              │   update state)     │
              └─────────┬──────────┘
                        │ New state → next period
              ┌─────────▼──────────┐
              │   EVALUATION        │
              │  Bullwhip ratio     │
              │  Cost per period    │
              │  Service level      │
              └────────────────────┘
```

---

## Part 8 — The Things That Will Break (and How to Handle Them)

### Problem 1: LLM Anchors on Mean Demand
The LLM sees "average demand = 100" and orders 100 every time, ignoring variance entirely.

**Fix**: Explicitly include variance, autocorrelation, and lead time variance in the prompt. Force the LLM to address uncertainty before making a decision (chain-of-thought step 1).

### Problem 2: LLM Makes Inconsistent Decisions
In period 5 it chooses POUT; in period 6 with almost identical conditions, it chooses OUT.

**Fix**: Include the last period's decision and its outcome in the context. Add a "policy consistency" guideline to the system prompt.

### Problem 3: State Explosion in Long Simulations
After 100 periods, the history is too long for the LLM's context window.

**Fix**: Summarize history into rolling statistics (last-5-period average, variance trend) rather than passing raw history. This is called *context compression*.

### Problem 4: LLM Doesn't Know Math
You ask it to compute the optimal α for POUT and it makes up a number.

**Fix**: Tool use. Give the LLM a `compute_pout_alpha(autocorrelation, lead_time)` function it can call. It decides *when* to use it; the function does the math.

### Problem 5: No Feedback Loop
The LLM picks a policy but never finds out if it worked.

**Fix**: At the start of each period, include the previous period's realized bullwhip ratio and inventory cost in the LLM's input. Let it self-evaluate.

---

## Quick Reference: Which Time Series Model for Which Echelon?

| Echelon | Demand Characteristics | Recommended Model | Rationale |
|---|---|---|---|
| Retailer (E1) | Closest to real demand, moderate noise | ARIMA or ETS | Stationary or mildly trended; classical theory uses AR(1) |
| Warehouse (E2) | Lumpy, batch orders, rising autocorrelation | LSTM | Captures nonlinear autocorrelation better than ARIMA |
| Distributor (E3) | High autocorrelation, lead time variance | TFT | Multi-variate, incorporates lead time as covariate |
| Factory/Supplier (E4) | Extremely noisy, very high autocorrelation | TFT or ARMA(p,q) with large p | Heavy autocorrelation structure, needs long memory |

---

## Quick Reference: Which Policy for Which Scenario?

| Scenario | LLM Should Choose |
|---|---|
| Low autocorrelation, deterministic lead times | Lot-for-Lot or OUT |
| Moderate autocorrelation, short lead times | OUT with MMSE forecast |
| High autocorrelation, any lead time | POUT (α ≈ 0.3-0.5) |
| Very high autocorrelation, stochastic lead times | POUT (α ≈ 0.1-0.3) + safety stock buffer |
| Known ARMA(p,q) structure | Full-state feedback OUT |
| High ordering costs, stable demand | EOQ |
| Sporadic, low-volume demand | Min-Max (s,S) |

---

*Key papers cited throughout: Lee et al. (1997), Dejonckheere et al. (2003), Disney & Lambrecht (2008), Wang & Disney (2017), Michna et al. (2020), InvAgent / Quan & Liu (2024), AIM-Bench / Zhao et al. (2025), Jannelli et al. (2025).*

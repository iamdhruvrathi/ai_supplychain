# AI Supply Chain — Beer Game Simulator

This repository is a research-focused implementation of the classic Beer Game
multi-echelon supply chain, built for RL, LLM experiments, and empirical study
of the bullwhip effect. It provides a modular simulator, classical policies,
metrics, plotting, and an experiment harness.

Highlights

- Multi-echelon supply chain (Retailer → Wholesaler → Distributor → Factory)
- FIFO shipment pipelines with configurable lead times
- Inventory/backlog handling and per-step cost accounting
- History recording for orders, demand, inventory, backlog, and costs
- Modular metrics including bullwhip computations
- Classical policies and an experiment harness
- **LLM-driven agents** via Ollama for autonomous supply chain decisions
- RL-ready state API for future PPO/GRPO training

Installation

- Create a virtual environment and install requirements:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Quick usage

```python
from simulator.beer_game import BeerGame

env = BeerGame(max_weeks=30, verbose=False)
env.reset()

actions = {name: 5 for name in ["Retailer","Wholesaler","Distributor","Factory"]}
while True:
    _, _, done, info = env.step(actions)
    if done:
        break

print('Bullwhip:', info.get('bullwhip'))
```

Running baseline experiments

```powershell
python experiments\baseline_experiment.py
```

## LLM-Driven Experiments with Ollama

Run autonomous supply chains controlled by local LLMs.

### Setup

1. **Install Ollama** from https://ollama.ai
2. **Start Ollama**:
   ```powershell
   ollama serve
   ```
3. **Pull a model** (in another terminal):
   ```powershell
   ollama pull qwen:1.5b
   ```
   (Or use another lightweight model like `ollama pull phi` or `ollama pull neural-chat`)

### Test LLM Agent

Verify prompt building and parsing (no Ollama connection required):

```powershell
python experiments\test_llm_agent.py
```

### Run LLM Experiment

Run a full Beer Game simulation with each node controlled by an LLM:

```powershell
python experiments\llm_experiment.py --weeks 30 --model qwen:1.5b
```

Options:

- `--weeks N` — number of weeks (default 30)
- `--model NAME` — Ollama model (default qwen:1.5b)
- `--url URL` — Ollama API endpoint (default http://localhost:11434)
- `--max-order N` — maximum order quantity (default 100)
- `--output PATH` — CSV output file (default results/llm_experiment_results.csv)

Example:

```powershell
python experiments\llm_experiment.py --weeks 50 --model qwen:1.5b --output results/my_experiment.csv
```

Sample prompt sent to Ollama:

```text
You are an inventory management agent in a multi-echelon supply chain Beer Game.

Your goal is to:
* minimize stockouts
* avoid excessive inventory
* reduce supply chain instability

Current state:
Inventory: 12
Backlog: 3
Incoming shipments: 5
Pipeline inventory: 4
Last customer demand: 8
Last order placed: 6
Current week: 7

Decide how many units to order this week.

Rules:
* Return ONLY a single integer.
* No explanation.
* Order must be between 0 and 100.

Order:
```

Results will be saved as CSV with per-week metrics: demand, cost, bullwhip, orders, inventory, backlog.
Plots are also generated in `plots/` including demand vs orders, inventory/backlog trajectories, bullwhip ratios, and cumulative cost.

## Architecture (LLM decision loop)

Each echelon runs an independent `LLMAgent` with no shared messages:

```
env.get_all_states()  →  local state per node
       ↓
LLMAgent.build_prompt(state)  →  Ollama /api/generate
       ↓
LLMAgent.parse_order(response)  →  clamped integer order
       ↓
env.step(actions)  →  FIFO shipments, costs, bullwhip history
       ↓
repeat until max_weeks
```

RL-ready observation (same schema for future PPO/GRPO/MARL):

```python
state = env.get_state("Retailer")
# inventory, backlog, incoming_shipments, pipeline_inventory,
# last_customer_demand, last_order, current_week
all_states = env.get_all_states()
```

## Future RL roadmap

1. **Behavioral baseline** — LLM policies as a reference for bullwhip and cost.
2. **Shared state API** — `get_state` / `get_all_states` feed Gymnasium wrappers.
3. **PPO / GRPO** — train per-echelon or centralized critics on the same observations.
4. **MARL** — optional communication channels layered on top of local state.
5. **Comparison studies** — LLM vs `base_stock`, `moving_average`, and learned policies.

Project layout

- `simulator/` — core simulator and node abstraction
- `policies/` — classical policy implementations (base-stock, moving average, random)
- `agents/` — LLM / RL agent scaffolds
- `metrics/` — bullwhip and variance metrics
- `experiments/` — reproducible experiment runners
- `plots/` — saved figures (created by plotting routines)

Research notes

- The environment uses FIFO pipelines: each node's `incoming_shipments` deque has
  length equal to its lead time. On each step the left-most element arrives and is
  added to inventory; new outgoing shipments are appended to the right to arrive
  after the configured lead time.
- Costs: holding cost is `inventory * holding_cost`, backlog cost is `backlog * backlog_cost`.
- Bullwhip is computed as the variance of agent orders divided by variance of customer demand.
- LLM agents receive structured state (inventory, backlog, pipeline, demand, orders, week) and output order quantities via Ollama.
- Each node acts independently—no communication between agents. This allows study of decentralized decision-making.

Roadmap

- ✓ Core simulator with FIFO pipelines and cost tracking
- ✓ Classical policies (base-stock, moving average, random)
- ✓ Bullwhip metrics and variance analysis
- ✓ LLM-driven agents via Ollama
- [ ] RL agent scaffolding (PPO, GRPO)
- [ ] Dataset-backed demand generation
- [ ] Multi-agent communication patterns
- [ ] LLM fine-tuning for supply chain
- [ ] Visualization dashboard

Citation

If you use this project in research, please cite:

```bibtex
@software{beer_game_llm_2026,
  title={AI Supply Chain Beer Game Simulator},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ai_supplychain}
}
```

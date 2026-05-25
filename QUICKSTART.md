"""
BEER GAME LLM EXPERIMENT - QUICK START GUIDE

This guide walks you through setting up and running LLM-driven Beer Game experiments.

==================================================
WHAT WAS BUILT
==================================================

1. SupplyChainNode Abstraction (simulator/node.py)
   - Dataclass-based node representation
   - Manages inventory, backlog, FIFO pipeline, orders, costs
   - Reusable for any supply chain node

2. Enhanced BeerGame Simulator (simulator/beer_game.py)
   - RL-ready state API: get_state_dict(agent_name), get_all_states_dict()
   - History tracking with bullwhip metrics
   - Optional verbose logging
   - Plotting utilities

3. LLMAgent with Ollama (agents/llm_agent.py)
   - Builds structured prompts from state
   - Queries local Ollama API via HTTP
   - Safe output parsing with clamping
   - Graceful error handling and logging

4. Modular Policies (policies/)
   - base_stock.py - classic base-stock policy
   - moving_average.py - demand-driven policy
   - random_policy.py - baseline random policy

5. Metrics (metrics/bullwhip.py)
   - Classical bullwhip computation
   - Per-agent and overall ratios
   - Variance-based analysis

6. Experiment Framework
   - experiments/baseline_experiment.py - compare classical policies
   - experiments/llm_experiment.py - run LLM-driven simulations
   - experiments/test_llm_agent.py - unit tests
   - experiments/test_state_api.py - state API validation

==================================================
SETUP INSTRUCTIONS
==================================================

1. Clone/download the repository

2. Create a Python virtual environment:
   python -m venv venv
   venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt

4. Install and run Ollama:
   a) Download from https://ollama.ai
   b) Start the server:
   ollama serve
   c) In a new terminal, pull a model (lightweight):
   ollama pull qwen:1.5b
   (Or: ollama pull phi or ollama pull neural-chat)

5. Verify Ollama is accessible:
   curl http://localhost:11434/api/tags

==================================================
RUNNING EXPERIMENTS
==================================================

A. Test LLM Agent (no Ollama needed):
python experiments\test_llm_agent.py

This tests:

- Prompt building
- Output parsing
- Connection fallback

Expected: ✓ All tests passed!

B. Test State API:
python experiments\test_state_api.py

Expected: ✓ State API test passed!

C. Run Classical Baselines:
python experiments\baseline_experiment.py

Results saved to: results/baseline_results.csv
Compares: random, base-stock, moving-average policies

D. Run LLM Experiment (requires Ollama running):
python experiments\llm_experiment.py --weeks 30 --model qwen:1.5b

Options:
--weeks N Number of simulation weeks (default 30)
--model NAME Ollama model name (default qwen:1.5b)
--url URL Ollama API URL (default http://localhost:11434)
--max-order N Max order quantity (default 100)
--output PATH CSV output file (default results/llm_experiment_results.csv)

Results saved to: results/llm_experiment_results.csv
Contains: per-week metrics, orders, inventory, backlog, bullwhip

==================================================
EXPECTED OUTPUT
==================================================

LLM Experiment Console Output:
2026-05-25 10:30:45 - **main** - INFO - Starting LLM experiment with model: qwen:1.5b
2026-05-25 10:30:45 - **main** - INFO - Max weeks: 30, Max order: 100
2026-05-25 10:30:46 - **main** - INFO - Starting simulation loop...
2026-05-25 10:30:50 - **main** - INFO - Week 5: Demand=5, Cost=42.0, Bullwhip=0.0
2026-05-25 10:30:55 - **main** - INFO - Week 10: Demand=4, Cost=38.0, Bullwhip=0.1
...
2026-05-25 10:31:30 - **main** - INFO - Results saved to results/llm_experiment_results.csv
2026-05-25 10:31:30 - **main** - INFO - === Experiment Summary ===
2026-05-25 10:31:30 - **main** - INFO - Total weeks: 30
2026-05-25 10:31:30 - **main** - INFO - Average system cost: 45.32
2026-05-25 10:31:30 - **main** - INFO - Total system cost: 1359.60
2026-05-25 10:31:30 - **main** - INFO - Average bullwhip: 1.25
2026-05-25 10:31:30 - **main** - INFO - ✓ Experiment completed successfully!

CSV Output Columns:
week | customer_demand | total_system_cost | bullwhip_overall
order_Retailer | order_Wholesaler | order_Distributor | order_Factory
inventory_Retailer | inventory_Wholesaler | ...
backlog_Retailer | backlog_Wholesaler | ...

==================================================
EXAMPLE: USE LLM AGENT DIRECTLY
==================================================

from simulator.beer_game import BeerGame
from agents.llm_agent import LLMAgent

# Initialize environment and agent

env = BeerGame(max_weeks=10, verbose=False)
env.reset()

agent = LLMAgent(model_name="qwen:1.5b", max_order=100)

# Run one week

state = env.get_state_dict("Retailer")
order = agent.generate_order("Retailer", state, fallback=5)
print(f"Retailer ordered: {order}")

# Collect orders from all agents

actions = {
node.name: agent.generate_order(node.name, env.get_state_dict(node.name))
for node in env.nodes
}

# Step environment

next_state, reward, done, info = env.step(actions)
print(f"Week {info['week']}: Cost={info['total_system_cost']}, Bullwhip={info['bullwhip']}")

==================================================
ARCHITECTURE OVERVIEW
==================================================

Beer Game Simulator
├── SupplyChainNode (inventory, backlog, pipeline)
├── BeerGame Environment (RL-style reset/step)
├── State API (get_state_dict, get_all_states_dict)
└── History & Metrics (bullwhip, costs, orders)

LLM-Driven Agents
├── LLMAgent (Ollama integration)
├── Prompt Building (state → natural language)
├── Output Parsing (response → order quantity)
└── Error Handling (fallback, timeouts)

Experiment Framework
├── Baseline Policies (random, base-stock, moving-avg)
├── LLM Experiment Runner (orchestrates simulation)
└── Tests & Validation (unit tests, integration tests)

Metrics & Analysis
├── Bullwhip Ratio (variance of orders / variance of demand)
├── Per-Agent Analysis (individual agent metrics)
└── CSV Export (for plotting and further analysis)

==================================================
RESEARCH NEXT STEPS
==================================================

1. Compare LLM vs Classical Policies:
   python experiments/baseline_experiment.py
   python experiments/llm_experiment.py
   → Compare costs, bullwhip, stability

2. Analyze Prompt Engineering:
   Modify agents/llm_agent.py build_prompt() to experiment with:
   - Different context lengths
   - Additional hints (moving average, trend)
   - Different instruction styles

3. Test Multiple Models:
   ollama pull phi
   python experiments/llm_experiment.py --model phi

   → Compare model quality, speed, decision consistency

4. Study Bullwhip with LLMs:
   - Plot variance growth by echelon
   - Compare LLM vs base-stock bullwhip
   - Analyze decision patterns over time

5. Future: RL Training
   Once LLM baseline is established:
   - Implement PPO/GRPO agents
   - Use LLM as a behavioral baseline
   - Study multi-agent coordination

==================================================
TROUBLESHOOTING
==================================================

Q: "Failed to connect to Ollama at http://localhost:11434"
A: Make sure Ollama is running:
ollama serve
(should show "Listening on ...")

Q: "Model 'qwen:1.5b' not found"
A: Pull the model first:
ollama pull qwen:1.5b

List available models:
ollama list

Q: LLM responses are slow
A: Try a smaller model:
ollama pull phi
python experiments/llm_experiment.py --model phi

Or increase timeout:
--timeout 60

Q: No integer found in LLM response
A: The model is being too verbose. The prompt is designed to force
single-integer output. If this happens, try:

- Different model (phi, neural-chat)
- Lower temperature (already set to 0.2)
- Modify the prompt in agents/llm_agent.py build_prompt()

==================================================
PROJECT STRUCTURE
==================================================

project/
├── simulator/
│ ├── beer_game.py (main environment with state API)
│ ├── node.py (SupplyChainNode dataclass)
│ └── **init**.py
├── agents/
│ ├── llm_agent.py (Ollama-integrated LLM agent)
│ └── **init**.py
├── policies/
│ ├── base_stock.py (base-stock policy)
│ ├── moving_average.py (moving-average policy)
│ ├── random_policy.py (random policy)
│ └── **init**.py
├── metrics/
│ ├── bullwhip.py (bullwhip calculations)
│ └── **init**.py
├── experiments/
│ ├── baseline_experiment.py (compare classical policies)
│ ├── llm_experiment.py (LLM-driven simulation)
│ ├── test_llm_agent.py (unit tests)
│ ├── test_state_api.py (state API test)
│ └── smoke_test.py (basic integration test)
├── plots/ (saved figures)
├── results/ (CSV outputs)
├── requirements.txt
├── README.md
└── main.py (optional entry point)

==================================================
CITATIONS & REFERENCES
==================================================

Beer Game (original):
Sterman, J. D. (1989). "Modeling managerial behavior:
Misperceptions of feedback in a dynamic decision making experiment."
Management Science, 35(3), 321-339.

Bullwhip Effect:
Lee, H. L., Padmanabhan, V., & Whang, S. (1997).
"The bullwhip effect in supply chains."
Sloan Management Review, 38(3), 93-102.

Ollama:
https://ollama.ai

For updates and community:
https://github.com/ollama/ollama

==================================================
END OF GUIDE
==================================================
"""

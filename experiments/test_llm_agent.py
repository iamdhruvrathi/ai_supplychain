"""Tests for LLM agent, state API, and experiment runner.

Usage:
    python experiments/test_llm_agent.py
    python experiments/test_llm_agent.py --ollama   # optional live Ollama check
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.llm_agent import LLMAgent
from experiments.llm_experiment import run_llm_experiment
from simulator.beer_game import BeerGame


def test_prompt_building() -> None:
    agent = LLMAgent(agent_name="Retailer", model_name="qwen:1.5b")

    state = {
        "inventory": 15,
        "backlog": 2,
        "incoming_shipments": 5,
        "pipeline_inventory": 5,
        "last_customer_demand": 8,
        "last_order": 10,
        "current_week": 7,
    }

    prompt = agent.build_prompt(state)
    assert "inventory management agent" in prompt.lower()
    assert "Inventory: 15" in prompt
    assert "Backlog: 2" in prompt
    assert "Return ONLY a single integer" in prompt
    assert "between 0 and 100" in prompt
    print("[OK] Prompt building test passed")
    print(f"Sample prompt:\n{prompt}\n")


def test_tool_prompt_and_metadata() -> None:
    agent = LLMAgent(
        agent_name="Retailer",
        model_name="qwen:1.5b",
        use_tool_recommendation=True,
    )

    state = {
        "inventory": 12,
        "backlog": 5,
        "incoming_shipments": 0,
        "pipeline_inventory": 0,
        "last_customer_demand": 8,
        "last_order": 10,
        "current_week": 1,
        "lead_time": 2,
    }

    prompt = agent.build_prompt(state)
    assert "Tool Recommendation:" in prompt
    assert "Order 9 units." in prompt
    assert "You may follow or ignore this recommendation." in prompt

    original = agent.query_model
    agent.query_model = lambda prompt: "11"
    try:
        order = agent.generate_order(state, fallback=0)
    finally:
        agent.query_model = original

    assert order == 11
    assert agent.last_decision_metadata["tool_order"] == 9
    assert agent.last_decision_metadata["llm_order"] == 11
    assert agent.last_decision_metadata["difference"] == 2
    print("[OK] Tool prompt and metadata test passed\n")


def test_negotiation_prompt_context() -> None:
    agent = LLMAgent(agent_name="Retailer", model_name="qwen:1.5b")
    state = {
        "inventory": 15,
        "backlog": 2,
        "incoming_shipments": 5,
        "pipeline_inventory": 5,
        "last_customer_demand": 8,
        "last_order": 10,
        "current_week": 7,
        "negotiation_proposals": {
            "Retailer": 25,
            "Wholesaler": 18,
            "Distributor": 16,
            "Factory": 15,
        },
    }

    prompt = agent.build_prompt(state)
    assert "Other agents propose:" in prompt
    assert "Retailer: 25" in prompt
    assert "Would you revise your order?" in prompt
    print("[OK] Negotiation prompt context test passed\n")


def test_parsing() -> None:
    agent = LLMAgent(agent_name="Retailer", max_order=100)

    cases = [
        ("15", 15),
        ("Order: 25", 25),
        ("The order is 30 units", 30),
        ("-5", 0),
        ("150", 100),
        ("No number here", 0),
        (None, 0),
        (
            "Let me think step by step. Inventory is low. "
            "Final order quantity: 42",
            42,
        ),
        (
            "Reasoning: backlog is high.\n\nAnswer: 18",
            18,
        ),
    ]

    for response, expected in cases:
        result = agent.parse_order(response, default=0)
        assert result == expected, f"{response!r} -> {result}, expected {expected}"
        print(f"  [OK] parse_order({response!r}) -> {result}")

    print("[OK] Parsing test passed\n")


def test_connection_fallback() -> None:
    agent = LLMAgent(
        agent_name="Retailer",
        ollama_url="http://localhost:59999",
    )
    assert agent.query_model("test prompt") is None
    assert agent.generate_order({"inventory": 10, "current_week": 1}) == 0
    print("[OK] Connection fallback test passed\n")


def test_trajectory_logging() -> None:
    env = BeerGame(max_weeks=2, verbose=False)
    env.reset()
    actions = {n: 5 for n in ("Retailer", "Wholesaler", "Distributor", "Factory")}
    env.step(actions)

    trajectories = env.get_trajectories()
    assert len(trajectories) == 4
    entry = trajectories[0]
    required = {
        "week", "agent", "state", "action", "reward",
        "next_state", "cost", "bullwhip",
    }
    assert required <= entry.keys()
    assert entry["action"] == 5
    assert "tool_order" in entry
    assert "llm_order" in entry
    assert "difference" in entry
    assert "consensus_gap" in entry
    print("[OK] Trajectory logging test passed\n")


def test_reward_shaping() -> None:
    env = BeerGame(max_weeks=1, alpha=1.0, beta=0.1, gamma=0.5)
    env.reset()
    actions = {n: 5 for n in ("Retailer", "Wholesaler", "Distributor", "Factory")}
    _, reward, _, info = env.step(actions)

    components = info["reward_components"]
    expected = -(
        components["cost"] + components["bullwhip"] + components["backlog"]
    )
    assert abs(reward - expected) < 1e-6
    print("[OK] Reward shaping test passed\n")


def test_state_api() -> None:
    env = BeerGame(max_weeks=2, verbose=False)
    env.reset()
    actions = {n: 5 for n in ("Retailer", "Wholesaler", "Distributor", "Factory")}
    env.step(actions)

    retailer = env.get_state("Retailer")
    required = {
        "inventory",
        "backlog",
        "incoming_shipments",
        "pipeline_inventory",
        "last_customer_demand",
        "last_order",
        "current_week",
    }
    assert required <= retailer.keys()

    all_states = env.get_all_states()
    assert set(all_states.keys()) == {
        "Retailer", "Wholesaler", "Distributor", "Factory"
    }
    assert len(all_states) == 4
    assert all(required <= s.keys() for s in all_states.values())

    print("[OK] State API test passed\n")


def test_offline_experiment_run() -> None:
    original = LLMAgent.generate_order

    def stub_order(self, state, fallback=0):
        return 5

    LLMAgent.generate_order = stub_order
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "llm_experiment_results.csv")
            df = run_llm_experiment(
                max_weeks=3,
                results_file=output,
            )
            assert len(df) == 3
            assert os.path.exists(output)
            assert "total_system_cost" in df.columns
            assert "bullwhip_overall" in df.columns
        print("[OK] Offline experiment run test passed\n")
    finally:
        LLMAgent.generate_order = original


def test_ollama_connection() -> None:
    agent = LLMAgent(agent_name="Retailer", model_name="qwen:1.5b")
    state = {
        "inventory": 12,
        "backlog": 0,
        "incoming_shipments": 0,
        "pipeline_inventory": 0,
        "last_customer_demand": 5,
        "last_order": 5,
        "current_week": 1,
    }
    order = agent.generate_order(state, fallback=-1)
    assert 0 <= order <= agent.max_order
    print(f"[OK] Ollama live test returned order: {order}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Run optional live Ollama integration test",
    )
    args = parser.parse_args()

    print("Running LLM agent tests...\n")
    test_prompt_building()
    test_tool_prompt_and_metadata()
    test_negotiation_prompt_context()
    test_parsing()
    test_connection_fallback()
    test_trajectory_logging()
    test_reward_shaping()
    test_state_api()
    test_offline_experiment_run()

    if args.ollama:
        try:
            test_ollama_connection()
        except Exception as exc:
            print(f"[FAIL] Ollama live test failed: {exc}")
            sys.exit(1)

    print("[OK] All tests passed!")


if __name__ == "__main__":
    main()

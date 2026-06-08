"""Unit tests for LLMAgent order parsing and clamping."""

from __future__ import annotations

import pytest

from agents.llm_agent import LLMAgent


@pytest.fixture
def agent() -> LLMAgent:
    return LLMAgent(agent_name="Retailer", model_name="qwen:1.5b", max_order=100)


class TestParseOrder:
    def test_plain_integer(self, agent: LLMAgent) -> None:
        assert agent.parse_order("42") == 42

    def test_explicit_order_label(self, agent: LLMAgent) -> None:
        assert agent.parse_order("After analysis, order: 17 units") == 17

    def test_reasoning_heavy_output_uses_last_integer(self, agent: LLMAgent) -> None:
        text = (
            "Let me think step by step. Inventory is low. "
            "Backlog is rising. Therefore I will order 23 units."
        )
        assert agent.parse_order(text) == 23

    def test_fenced_code_block(self, agent: LLMAgent) -> None:
        assert agent.parse_order("```\n55\n```") == 55

    def test_json_structured_output(self, agent: LLMAgent) -> None:
        assert agent.parse_order('{"order": 33}') == 33

    def test_none_returns_default(self, agent: LLMAgent) -> None:
        assert agent.parse_order(None, default=7) == 7

    def test_empty_string_returns_default(self, agent: LLMAgent) -> None:
        assert agent.parse_order("", default=5) == 5

    def test_invalid_text_returns_default(self, agent: LLMAgent) -> None:
        assert agent.parse_order("no numbers here", default=3) == 3

    def test_clamp_above_max_order(self, agent: LLMAgent) -> None:
        assert agent.parse_order("500") == 100

    def test_clamp_negative_to_zero(self, agent: LLMAgent) -> None:
        assert agent.parse_order("order: -5") == 0

    def test_parse_order_optional_returns_none_on_invalid(self, agent: LLMAgent) -> None:
        # parse_order_optional uses sentinel; empty -> default sentinel -> None
        assert agent.parse_order_optional("") is None

    def test_parse_order_optional_returns_value(self, agent: LLMAgent) -> None:
        assert agent.parse_order_optional("12") == 12


class TestGenerateOrderFallback:
    def test_fallback_when_model_returns_none(self, agent: LLMAgent) -> None:
        agent.query_model = lambda prompt: None  # type: ignore[method-assign]
        state = {"inventory": 10, "backlog": 0, "incoming_shipments": 0,
                 "pipeline_inventory": 0, "last_customer_demand": 4,
                 "last_order": 0, "current_week": 1}
        assert agent.generate_order(state, fallback=15) == 15

    def test_fallback_on_unparseable_response(self, agent: LLMAgent) -> None:
        agent.query_model = lambda prompt: "I cannot decide"  # type: ignore[method-assign]
        state = {"inventory": 10, "backlog": 0, "incoming_shipments": 0,
                 "pipeline_inventory": 0, "last_customer_demand": 4,
                 "last_order": 0, "current_week": 1}
        assert agent.generate_order(state, fallback=9) == 9

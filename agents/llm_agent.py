"""LLM agent for Beer Game with Ollama integration.

Each echelon uses a dedicated LLMAgent instance that receives local state only,
queries a local Ollama model, and returns a clamped integer order quantity.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Optional

from tools.inventory_tool import eoq_recommendation

from agents.llm_backends import (
    OllamaBackend,
    GroqBackend,
    VLLMBackend,
)

logger = logging.getLogger(__name__)


class LLMAgent:
    """Agent that uses a local Ollama model to make supply chain decisions."""

    def __init__(
        self,
        agent_name: str,
        model_name: str = "qwen:1.5b",
        ollama_url: str = "http://localhost:11434",
        max_order: int = 10000,
        temperature: float = 0.2,
        timeout: float = 120.0,
        num_predict: int = 32,
        reasoning_mode: bool = False,
        keep_alive: str = "30m",
        backend: str = "ollama",
        backend_kwargs: Optional[Dict[str, Any]] = None,
        use_tool_recommendation: bool = False,
    ) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.max_order = max_order
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = int(num_predict)
        # Whether the agent is allowed to use chain-of-thought / internal
        # deliberation. When True, we allocate a larger token budget to
        # permit multi-step reasoning traces. Default False keeps backwards
        # compatible behaviour (disable reasoning traces).
        self.reasoning_mode = bool(reasoning_mode)
        self.keep_alive = keep_alive
        self.backend_name = backend or "ollama"
        self.backend_kwargs = backend_kwargs or {}
        self.use_tool_recommendation = bool(use_tool_recommendation)
        self.last_decision_metadata: Dict[str, Any] = {}

        # Initialize the chosen backend. Keep Ollama behaviour unchanged by
        # default; Groq backend implemented in `agents.llm_backends`.
        if self.backend_name.lower() == "ollama":
            self.backend = OllamaBackend(
                model_name=self.model_name,
                ollama_url=self.ollama_url,
                timeout=self.timeout,
                num_predict=self.num_predict,
                reasoning_mode=self.reasoning_mode,
                keep_alive=self.keep_alive,
            )
        elif self.backend_name.lower() == "groq":
            self.backend = GroqBackend(
                model_name=self.model_name,
                api_key=self.backend_kwargs.get("api_key"),
                api_url=self.backend_kwargs.get("api_url"),
                timeout=self.timeout,
                reasoning_mode=self.reasoning_mode,
            )
        elif self.backend_name.lower() == "vllm":
            self.backend = VLLMBackend(
                model_name=self.model_name,
                base_url=self.backend_kwargs.get(
                    "base_url",
                    "http://localhost:8002/v1",
                ),
                timeout=self.timeout,
                reasoning_mode=self.reasoning_mode,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def _orchestrator_context(self, state: Dict[str, Any]) -> str:
        """Append curated shared information when orchestrator mode is active."""
        lines = []
        if "shared_current_demand" in state:
            lines.append(f"Shared customer demand: {state['shared_current_demand']}")
        if "shared_demand_history" in state:
            lines.append(f"Demand history: {state['shared_demand_history']}")
        if "shared_demand_volatility" in state:
            lines.append(f"Demand volatility: {state['shared_demand_volatility']:.2f}")
        if "system_total_backlog" in state:
            lines.append(f"System backlog: {state['system_total_backlog']}")
        if not lines:
            return ""
        return "\nOrchestrator shared data:\n" + "\n".join(f"* {ln}" for ln in lines) + "\n"

    def _with_tool_recommendation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attach decision-support output when enabled.
        
        Tool recommendations are clamped to [0, max_order] to ensure they
        remain within the feasible action space. This prevents the LLM from
        seeing out-of-bounds recommendations that it cannot follow.
        """
        if not self.use_tool_recommendation:
            return dict(state)
        augmented = dict(state)
        if "tool_order" not in augmented:
            raw_recommendation = eoq_recommendation(augmented)
            # Clamp recommendation to valid action space [0, max_order]
            augmented["tool_order"] = max(0, min(self.max_order, raw_recommendation))
        return augmented

    def _negotiation_context(self, state: Dict[str, Any]) -> str:
        proposals = state.get("negotiation_proposals")
        if not proposals:
            return ""

        lines = ["\nOther agents propose:"]
        for name, order in proposals.items():
            lines.append(f"{name}: {order}")
        lines.append("\nWould you revise your order?")
        return "\n".join(lines) + "\n"

    def _tool_context(self, state: Dict[str, Any]) -> str:
        if "tool_order" not in state:
            return ""
        return (
            "\nTool Recommendation:\n"
            f"Order {state['tool_order']} units.\n\n"
            "You may follow or ignore this recommendation.\n"
        )

    def build_prompt(self, state: Dict[str, Any]) -> str:
        """Build a structured prompt from the agent's local state."""
        state = self._with_tool_recommendation(state)
        prompt = (
            "You are an inventory management agent in a multi-echelon "
            "supply chain Beer Game.\n\n"
            "Your goal is to:\n"
            "* minimize stockouts\n"
            "* avoid excessive inventory\n"
            "* reduce supply chain instability\n\n"
            "Current state:\n"
            f"Inventory: {state.get('inventory', 0)}\n"
            f"Backlog: {state.get('backlog', 0)}\n"
            f"Incoming shipments: {state.get('incoming_shipments', 0)}\n"
            f"Pipeline inventory: {state.get('pipeline_inventory', 0)}\n"
            f"Last customer demand: {state.get('last_customer_demand', 0)}\n"
            f"Last order placed: {state.get('last_order', 0)}\n"
            f"Current week: {state.get('current_week', 0)}\n"
        )
        extra = self._orchestrator_context(state)
        prompt = prompt + extra + self._tool_context(state)
        prompt = prompt + self._negotiation_context(state) + (
            "\nDecide how many units to order this week.\n\n"
            "Rules:\n"
            "* Return ONLY a single integer.\n"
            "* No explanation.\n"
            "* No units, markdown, JSON, or extra text.\n"
            f"* Order must be between 0 and {self.max_order}.\n"
        )
        return prompt

    def query_model(self, prompt: str) -> Optional[str]:
        """Delegate generation to the configured backend implementation."""
        try:
            return self.backend.generate(
                prompt,
                temperature=self.temperature,
                num_predict=self.num_predict,
                keep_alive=self.keep_alive,
                reasoning_mode=self.reasoning_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("LLM backend '%s' generation failed: %s", self.backend_name, exc)
            return None

    def _clamp_order(self, value: int) -> int:
        """Clamp order to valid range [0, max_order]."""
        clamped = max(0, min(self.max_order, int(value)))
        if clamped != value:
            logger.info(
                "Clamped order from %s to %s (valid range: 0-%s)",
                value,
                clamped,
                self.max_order,
            )
        return clamped

    def parse_order(
        self,
        response_text: Optional[str],
        default: int = 0,
    ) -> int:
        """Extract an integer order from LLM output (including reasoning models).

        Reasoning models often emit explanations before the final quantity.
        This method prefers explicit order patterns, then the last standalone
        integer in the response.
        """
        if response_text is None:
            return default

        text = str(response_text).strip()
        if not text:
            return default

        # Record raw response (truncated) for metadata and analysis.
        raw = (str(response_text)[:500]) if response_text is not None else ""
        # Ensure counters exist for this agent instance. These counters allow
        # experiment harnesses to compute percentages of parse methods used.
        if not hasattr(self, "parse_method_counters"):
            self.parse_method_counters = Counter()

        try:
            # First, attempt to parse JSON-style structured outputs commonly
            # requested from models when structured output is enabled.
            import json

            try:
                j = json.loads(response_text)
                if isinstance(j, dict) and "order" in j:
                    val = int(j["order"])
                    self.parse_method_counters.update(["json"])
                    self._last_parse_info = {"parse_method_used": "json", "raw_response": raw}
                    return self._clamp_order(val)
            except Exception:
                # Not JSON or doesn't contain order; continue with existing rules
                pass
            explicit_patterns = [
                r"(?:final\s+)?order\s*(?:quantity|amount)?\s*[:=]\s*(\d+)",
                r"(?:will\s+)?order\s+(\d+)\s*(?:units?)?",
                r"(?:answer|quantity|units?)\s*[:=]\s*(\d+)",
                r"```\s*(\d+)\s*```",
            ]
            for pattern in explicit_patterns:
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                if matches:
                    self.parse_method_counters.update(["explicit_pattern"])
                    self._last_parse_info = {"parse_method_used": "explicit_pattern", "raw_response": raw}
                    return self._clamp_order(int(matches[-1]))

            standalone = re.findall(r"\b(\d+)\b", text)
            if standalone:
                # This is a fallback commonly triggered when the model outputs
                # reasoning text followed by a final number. We log this case
                # because it indicates the model produced extra tokens that
                # were not pure final-answer-only outputs.
                last = standalone[-1]
                idx = text.rfind(last)
                prefix = text[max(0, idx - 3):idx]
                if "-" in prefix:
                    self.parse_method_counters.update(["standalone_integer"])
                    self._last_parse_info = {"parse_method_used": "standalone_integer", "raw_response": raw}
                    logger.warning(
                        "Parser falling back to standalone integer extraction for agent %s; raw_response begins: %s",
                        self.agent_name,
                        raw[:200],
                    )
                    return self._clamp_order(0)
                self.parse_method_counters.update(["standalone_integer"])
                self._last_parse_info = {"parse_method_used": "standalone_integer", "raw_response": raw}
                return self._clamp_order(int(last))

            signed = re.search(r"-?\d+", text)
            if signed:
                self.parse_method_counters.update(["signed_integer"])
                self._last_parse_info = {"parse_method_used": "signed_integer", "raw_response": raw}
                return self._clamp_order(int(signed.group()))

            self.parse_method_counters.update(["fallback_default"])
            self._last_parse_info = {"parse_method_used": "fallback_default", "raw_response": raw}
            logger.warning("No integer found in LLM response: %s", text[:200])
            return default
        except (TypeError, ValueError) as exc:
            logger.error(
                "Error parsing order from '%s': %s", text[:200], exc
            )
            return default

    def parse_order_optional(self, response_text: Optional[str]) -> Optional[int]:
        """Parse an order, returning None when no integer is found."""
        sentinel = -10**9
        parsed = self.parse_order(response_text, default=sentinel)
        if parsed == sentinel:
            return None
        return parsed

    def generate_order(
        self,
        state: Dict[str, Any],
        fallback: int = 0,
    ) -> int:
        """Build prompt, query Ollama, and return a safe order quantity."""
        prompt_state = self._with_tool_recommendation(state)
        prompt = self.build_prompt(prompt_state)
        response = self.query_model(prompt)
        parsed = self.parse_order(response, default=fallback)
        order = self._clamp_order(parsed)
        tool_order = prompt_state.get("tool_order")
        # Attach richer metadata for analysis and reproducibility.
        parse_info = getattr(self, "_last_parse_info", {}) or {}
        raw_response = parse_info.get("raw_response") if parse_info else (str(response)[:500] if response else None)
        parse_method = parse_info.get("parse_method_used") if parse_info else None
        diff = (
            abs(int(order) - int(tool_order)) if tool_order is not None else None
        )
        self.last_decision_metadata = {
            "tool_order": int(tool_order) if tool_order is not None else None,
            "llm_order": int(order),
            "difference": diff,
            "tool_followed": diff,
            "parse_method_used": parse_method,
            "raw_response": raw_response,
        }
        return order

    def generate_order_majority_vote(
        self,
        state: Dict[str, Any],
        n_samples: int,
        fallback: int = 0,
    ) -> int:
        """Sample n orders and return the most common parsed value."""
        prompt_state = self._with_tool_recommendation(state)
        prompt = self.build_prompt(prompt_state)
        parsed_values = []

        for _ in range(max(1, int(n_samples))):
            response = self.query_model(prompt)
            parsed = self.parse_order_optional(response)
            if parsed is not None:
                parsed_values.append(parsed)

        if not parsed_values:
            order = self._clamp_order(fallback)
        else:
            counts = Counter(parsed_values)
            max_count = max(counts.values())
            winners = [value for value, count in counts.items() if count == max_count]
            order = self._clamp_order(min(winners))

        tool_order = prompt_state.get("tool_order")
        # Record parse method & raw response if available from last parse.
        parse_info = getattr(self, "_last_parse_info", {}) or {}
        raw_response = parse_info.get("raw_response") if parse_info else None
        parse_method = parse_info.get("parse_method_used") if parse_info else None

        self.last_decision_metadata = {
            "tool_order": int(tool_order) if tool_order is not None else None,
            "llm_order": int(order),
            "difference": (
                abs(int(order) - int(tool_order)) if tool_order is not None else None
            ),
            "tool_followed": (
                abs(int(order) - int(tool_order)) if tool_order is not None else None
            ),
            "parse_method_used": parse_method,
            "raw_response": raw_response,
            "sample_values": parsed_values,
            "sample_count": len(parsed_values),
        }
        return order

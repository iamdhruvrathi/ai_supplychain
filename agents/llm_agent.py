"""LLM agent for Beer Game with Ollama integration.

Each echelon uses a dedicated LLMAgent instance that receives local state only,
queries a local Ollama model, and returns a clamped integer order quantity.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class LLMAgent:
    """Agent that uses a local Ollama model to make supply chain decisions."""

    def __init__(
        self,
        agent_name: str,
        model_name: str = "qwen:1.5b",
        ollama_url: str = "http://localhost:11434",
        max_order: int = 100,
        temperature: float = 0.2,
        timeout: float = 120.0,
        num_predict: int = 8,
        keep_alive: str = "30m",
    ) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.max_order = max_order
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive

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

    def build_prompt(self, state: Dict[str, Any]) -> str:
        """Build a structured prompt from the agent's local state."""
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
        prompt = prompt + extra + (
            "\nDecide how many units to order this week.\n\n"
            "Rules:\n"
            "* Return ONLY a single integer.\n"
            "* No explanation.\n"
            "* No units, markdown, JSON, or extra text.\n"
            f"* Order must be between 0 and {self.max_order}.\n"
        )
        return prompt

    def query_model(self, prompt: str) -> Optional[str]:
        """Query the local Ollama /api/generate endpoint."""
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            }
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", data.get("text", "")).strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                "Failed to connect to Ollama at %s. "
                "Ensure Ollama is running: ollama serve",
                self.ollama_url,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
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

        try:
            explicit_patterns = [
                r"(?:final\s+)?order\s*(?:quantity|amount)?\s*[:=]\s*(\d+)",
                r"(?:will\s+)?order\s+(\d+)\s*(?:units?)?",
                r"(?:answer|quantity|units?)\s*[:=]\s*(\d+)",
                r"```\s*(\d+)\s*```",
            ]
            for pattern in explicit_patterns:
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                if matches:
                    return self._clamp_order(int(matches[-1]))

            standalone = re.findall(r"\b(\d+)\b", text)
            if standalone:
                last = standalone[-1]
                idx = text.rfind(last)
                prefix = text[max(0, idx - 3):idx]
                if "-" in prefix:
                    return self._clamp_order(0)
                return self._clamp_order(int(last))

            signed = re.search(r"-?\d+", text)
            if signed:
                return self._clamp_order(int(signed.group()))

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
        prompt = self.build_prompt(state)
        response = self.query_model(prompt)
        return self.parse_order(response, default=fallback)

    def generate_order_majority_vote(
        self,
        state: Dict[str, Any],
        n_samples: int,
        fallback: int = 0,
    ) -> int:
        """Sample n orders and return the most common parsed value."""
        prompt = self.build_prompt(state)
        parsed_values = []

        for _ in range(max(1, int(n_samples))):
            response = self.query_model(prompt)
            parsed = self.parse_order_optional(response)
            if parsed is not None:
                parsed_values.append(parsed)

        if not parsed_values:
            return self._clamp_order(fallback)

        counts = Counter(parsed_values)
        max_count = max(counts.values())
        winners = [value for value, count in counts.items() if count == max_count]
        return self._clamp_order(min(winners))

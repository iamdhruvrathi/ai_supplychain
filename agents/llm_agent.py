"""LLM agent for Beer Game with Ollama integration.

Each echelon uses a dedicated LLMAgent instance that receives local state only,
queries a local Ollama model, and returns a clamped integer order quantity.
"""

from __future__ import annotations

import logging
import re
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
    ) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.max_order = max_order
        self.temperature = temperature
        self.timeout = timeout

    def build_prompt(self, state: Dict[str, Any]) -> str:
        """Build a structured prompt from the agent's local state."""
        return (
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
            f"Current week: {state.get('current_week', 0)}\n\n"
            "Decide how many units to order this week.\n\n"
            "Rules:\n"
            "* Return ONLY a single integer.\n"
            "* No explanation.\n"
            f"* Order must be between 0 and {self.max_order}.\n"
        )

    def query_model(self, prompt: str) -> Optional[str]:
        """Query the local Ollama /api/generate endpoint."""
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
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

    def parse_order(
        self,
        response_text: Optional[str],
        default: int = 0,
    ) -> int:
        """Extract an integer order and clamp to [0, max_order]."""
        if response_text is None:
            return default

        try:
            match = re.search(r"-?\d+", response_text)
            if not match:
                logger.warning(
                    "No integer found in LLM response: %s", response_text
                )
                return default

            value = int(match.group())
            clamped = max(0, min(self.max_order, value))
            if clamped != value:
                logger.info(
                    "Clamped order from %s to %s (valid range: 0-%s)",
                    value,
                    clamped,
                    self.max_order,
                )
            return clamped
        except (TypeError, ValueError) as exc:
            logger.error(
                "Error parsing order from '%s': %s", response_text, exc
            )
            return default

    def generate_order(
        self,
        state: Dict[str, Any],
        fallback: int = 0,
    ) -> int:
        """Build prompt, query Ollama, and return a safe order quantity."""
        prompt = self.build_prompt(state)
        response = self.query_model(prompt)
        return self.parse_order(response, default=fallback)

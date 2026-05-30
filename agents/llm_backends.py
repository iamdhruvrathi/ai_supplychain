"""LLM backend abstractions: Ollama and Groq implementations.

New backends should implement `generate(prompt, **options) -> Optional[str]`.
This module is used by `agents.llm_agent` to delegate model inference.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None

logger = logging.getLogger(__name__)


class BaseLLMBackend:
    def generate(self, prompt: str, **options: Any) -> Optional[str]:
        raise NotImplementedError()


class OllamaBackend(BaseLLMBackend):
    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_predict: int = 8,
        keep_alive: str = "30m",
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = float(timeout)
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive

    def generate(self, prompt: str, **options: Any) -> Optional[str]:
        try:
            url = f"{self.ollama_url}/api/generate"
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": float(options.get("temperature", 0.2)),
                    "num_predict": int(options.get("num_predict", self.num_predict)),
                },
            }
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", data.get("text", "")).strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                "Failed to connect to Ollama at %s. Ensure Ollama is running: ollama serve",
                self.ollama_url,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return None


class GroqBackend(BaseLLMBackend):
    """Basic Groq API backend for Qwen-family models.

    This implementation uses the HTTP API and provides retry and timeout handling.
    It attempts to be permissive about returned JSON shapes so it can interoperate
    with multiple Groq response formats.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        api_url: Optional[str] = None,  # ignored
    **kwargs,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.timeout = float(timeout)

        if Groq is None:
            logger.warning(
                "groq SDK not installed; GroqBackend will be disabled until 'groq' is available"
            )
            self._client = None
        else:
            try:
                # Prefer explicit API key if provided, otherwise SDK picks up env var
                if self.api_key:
                    self._client = Groq(api_key=self.api_key)
                else:
                    self._client = Groq()
            except Exception as exc:  # pragma: no cover - SDK init failures
                logger.error("Failed to initialize Groq client: %s", exc)
                self._client = None

    def generate(self, prompt: str, **options: Any) -> Optional[str]:
        if self._client is None:
            logger.error("Groq SDK client not available; cannot call Groq API")
            return None

        # Build parameters
        temperature = float(options.get("temperature", 0.2))
        max_tokens = int(options.get("max_tokens", 256))
        # Map Ollama's `num_predict` to Groq `max_tokens` when provided.
        if "num_predict" in options:
            try:
                max_tokens = int(options.get("num_predict", max_tokens))
            except Exception:
                pass

        try:
            # Use chat completions API as requested.
            # Add a system instruction to discourage chain-of-thought/reasoning traces.
            system_msg = {
                "role": "system",
                "content": (
                    "Do NOT provide chain-of-thought, internal reasoning, or step-by-step"
                    " traces. Return ONLY the final concise answer with no explanations."
                ),
            }
            user_msg = {"role": "user", "content": prompt}

            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_msg, user_msg],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
            except TypeError:
                # Some SDK versions may not accept timeout or n; fall back gracefully
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_msg, user_msg],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Try common response shapes
            # SDK may return an object with 'choices' list
            if hasattr(resp, "choices") and resp.choices:
                first = resp.choices[0]
                # choice may have 'message' with 'content'
                msg = getattr(first, "message", None)
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("text")
                    if content:
                        text = str(content).strip()
                        return self._post_process(text)
                if hasattr(msg, "content"):
                    return self._post_process(str(msg.content).strip())

                # fallback: choice.text or choice.delta
                if hasattr(first, "text"):
                    return self._post_process(str(first.text).strip())

            # dict-like fallback
            if isinstance(resp, dict):
                # OpenAI-like
                choices = resp.get("choices") or []
                if choices:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        # message.content
                        msg = c0.get("message") or c0.get("delta")
                        if isinstance(msg, dict):
                            content = msg.get("content") or msg.get("text")
                            if content:
                                return self._post_process(str(content).strip())
                        if "text" in c0:
                            return self._post_process(str(c0.get("text")).strip())

            # As a last resort, stringify the response and post-process
            return self._post_process(str(resp).strip())
        except Exception as exc:
            logger.error("Groq SDK call failed: %s", exc)
            return None

    def _post_process(self, text: str) -> str:
        """Remove common chain-of-thought / reasoning traces from model output.

        - Remove fenced code blocks (```...```).
        - Remove ellipsis-delimited blocks: text between '...' and '...'.
        - Remove any remaining '...' sequences.
        - Collapse whitespace and return stripped text.
        """
        if not text:
            return text

        # Remove fenced code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Remove blocks delimited by three or more dots: '... ...'
        text = re.sub(r"\.{3,}[\s\S]*?\.{3,}", "", text)

        # Remove XML-style reasoning blocks like <think>...</think> (DOTALL, case-insensitive)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining standalone ellipses
        text = text.replace("...", "")

        # Collapse multiple whitespace into single spaces and strip
        text = re.sub(r"\s+", " ", text).strip()

        return text

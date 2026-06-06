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
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

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
        reasoning_mode: bool = False,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = float(timeout)
        self.num_predict = int(num_predict)
        # Whether chain-of-thought (thinking) is enabled. When enabled,
        # we allocate a larger token budget to avoid truncating intermediate
        # reasoning traces which would invalidate experimental comparisons.
        self.reasoning_mode = bool(reasoning_mode)
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
                    # Respect provided num_predict but enforce a sensible
                    # minimum token budget depending on reasoning_mode.
                    # If reasoning is enabled, require at least 512 tokens
                    # to avoid truncating chain-of-thought outputs. If not,
                    # require at least 64 tokens as a safe default.
                    "num_predict": int(options.get("num_predict", self.num_predict)),
                },
            }
            # Determine effective token budget and adjust payload.
            specified = int(options.get("num_predict", self.num_predict))
            min_budget = 512 if self.reasoning_mode else 64
            final_num_predict = max(specified, min_budget)
            payload["options"]["num_predict"] = final_num_predict
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", data.get("text", "") ) or ""

            # Log generation metadata for reproducibility and debugging.
            resp_len_chars = len(text)
            resp_len_words = len(text.split())
            truncation_flag = None
            # Ollama may include fields signalling truncation; log if present.
            if isinstance(data, dict):
                truncation_flag = data.get("truncated") or data.get("finish_reason") or data.get("stop_reason")

            logger.info(
                "Ollama generate model=%s num_predict=%d response_chars=%d response_words=%d truncation=%s",
                self.model_name,
                final_num_predict,
                resp_len_chars,
                resp_len_words,
                repr(truncation_flag),
            )

            return text.strip()
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
    reasoning_mode: bool = False,
    **kwargs,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        self.timeout = float(timeout)
        self.reasoning_mode = bool(reasoning_mode)

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

        # Enforce a minimum token budget depending on whether reasoning is
        # permitted; this prevents truncation of multi-step reasoning traces
        # which would invalidate experiments comparing thinking vs non-thinking.
        min_budget = 512 if self.reasoning_mode else 64
        final_max_tokens = max(max_tokens, min_budget)

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

            # Only include the explicit 'no chain-of-thought' system
            # instruction when reasoning is disabled. When reasoning is
            # enabled we avoid adding this instruction so models are free
            # to provide intermediate reasoning traces.
            messages = [user_msg]
            if not self.reasoning_mode:
                messages = [system_msg, user_msg]

            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=final_max_tokens,
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
                    out = self._post_process(str(msg.content).strip())
                    # Logging metadata
                    try:
                        finish = getattr(first, "finish_reason", None)
                    except Exception:
                        finish = None
                    logger.info(
                        "Groq generate model=%s max_tokens=%d response_chars=%d finish_reason=%s",
                        self.model_name,
                        final_max_tokens,
                        len(out),
                        repr(finish),
                    )
                    return out

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
                                    out = self._post_process(str(content).strip())
                                    logger.info(
                                        "Groq generate model=%s max_tokens=%d response_chars=%d finish_reason=%s",
                                        self.model_name,
                                        final_max_tokens,
                                        len(out),
                                        repr(c0.get("finish_reason")),
                                    )
                                    return out
                        if "text" in c0:
                                out = self._post_process(str(c0.get("text")).strip())
                                logger.info(
                                    "Groq generate model=%s max_tokens=%d response_chars=%d",
                                    self.model_name,
                                    final_max_tokens,
                                    len(out),
                                )
                                return out

            # As a last resort, stringify the response and post-process
            out = self._post_process(str(resp).strip())
            logger.info(
                "Groq generate model=%s max_tokens=%d response_chars=%d",
                self.model_name,
                final_max_tokens,
                len(out),
            )
            return out
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

class VLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8002/v1",
        timeout: float = 120.0,
        **kwargs,
    ):
        self.model_name = model_name
        self.timeout = timeout

        # vLLM (OpenAI-compatible shim) client. We do not require a real
        # API key for local vLLM instances; a placeholder is provided.
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        # Whether to permit chain-of-thought reasoning traces.
        self.reasoning_mode = bool(kwargs.get("reasoning_mode", False))

    def generate(self, prompt: str, **options):
        temperature = float(options.get("temperature", 0.2))
        # vLLM uses `max_tokens` but older code passed `num_predict`.
        specified = int(options.get("num_predict", 8))
        min_budget = 512 if self.reasoning_mode else 64
        max_tokens = max(specified, min_budget)

        try:
            # When reasoning is disabled, some backends accept a special
            # prefix to discourage chain-of-thought. Preserve that behavior
            # when `reasoning_mode` is False, otherwise send the prompt as-is.
            user_content = ("/no_think\n" + prompt) if not self.reasoning_mode else prompt

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the final order quantity.",
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Attempt to extract finish reason if present and log metadata.
            try:
                choice = resp.choices[0]
                finish = getattr(choice, "finish_reason", None)
                content = choice.message.content.strip() if hasattr(choice, "message") else str(choice).strip()
            except Exception:
                finish = None
                content = str(resp)

            logger.info(
                "vLLM generate model=%s max_tokens=%d response_chars=%d finish_reason=%s",
                self.model_name,
                max_tokens,
                len(content),
                repr(finish),
            )

            return content

        except Exception as exc:
            logger.error("vLLM call failed: %s", exc)
            return None
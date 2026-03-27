"""Ollama (local LLM) provider for NeuroPack."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from neuropack.llm.models import LLMConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3:1.7b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Local LLM provider via Ollama.

    Works with any model installed in Ollama (llama3, qwen3, mistral, etc.).
    No API key required -- everything runs locally.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion from a prompt string."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        body = self._post("/api/generate", payload)
        return body.get("response", "").strip()

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat-completions-style request to Ollama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        body = self._post("/api/chat", payload)
        msg = body.get("message", {})
        return msg.get("content", "").strip()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list[str]:
        """Return names of locally-installed Ollama models."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                models = body.get("models", [])
                return [m["name"] for m in models if "name" in m]
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

    def pull_model(self, model: str) -> None:
        """Download a model into Ollama (blocking)."""
        payload = {"name": model, "stream": False}
        self._post("/api/pull", payload, timeout=600)
        logger.info("Pulled model %s", model)

    # ------------------------------------------------------------------
    # LLMProvider-compatible call()
    # ------------------------------------------------------------------

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str | None:
        """Match the LLMProvider.call() interface so the registry can test it."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: int = 180) -> dict:
        """POST JSON to Ollama and return the parsed response body."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? (ollama serve)  Error: {reason}"
            ) from exc

    # ------------------------------------------------------------------
    # Factory from LLMConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: LLMConfig) -> OllamaProvider:
        """Create an OllamaProvider from an LLMConfig."""
        return cls(
            model=config.model or _DEFAULT_MODEL,
            base_url=config.base_url or _DEFAULT_BASE_URL,
        )

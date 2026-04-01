"""OpenAI-compatible local LLM provider.

Supports any local server that implements the OpenAI chat completions API:
  - LM Studio (default port 1234)
  - LocalAI (default port 8080)
  - Jan (default port 1337)
  - llama.cpp server (default port 8080)
  - vLLM (default port 8000)
  - KoboldCPP (default port 5001)
  - Ollama in OpenAI-compat mode (port 11434/v1)
  - text-generation-webui with --api (port 5000)

Usage:
    from neuropack.llm.local_openai import LocalOpenAIProvider

    # LM Studio
    llm = LocalOpenAIProvider(base_url="http://localhost:1234/v1")

    # Jan
    llm = LocalOpenAIProvider(base_url="http://localhost:1337/v1")

    # Auto-detect: tries common ports
    llm = LocalOpenAIProvider.auto_detect()
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Known local servers and their default endpoints
_KNOWN_SERVERS = {
    "lm-studio": {"url": "http://localhost:1234/v1", "name": "LM Studio"},
    "jan": {"url": "http://localhost:1337/v1", "name": "Jan"},
    "localai": {"url": "http://localhost:8080/v1", "name": "LocalAI"},
    "llamacpp": {"url": "http://localhost:8080/v1", "name": "llama.cpp"},
    "vllm": {"url": "http://localhost:8000/v1", "name": "vLLM"},
    "koboldcpp": {"url": "http://localhost:5001/v1", "name": "KoboldCPP"},
    "textgen": {"url": "http://localhost:5000/v1", "name": "text-generation-webui"},
    "ollama": {"url": "http://localhost:11434/v1", "name": "Ollama (OpenAI mode)"},
}


class LocalOpenAIProvider:
    """Local LLM via any OpenAI-compatible API endpoint.

    No API key needed. Everything runs on your machine.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "",
        api_key: str = "not-needed",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat completion request."""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model

        body = self._post("/chat/completions", payload)
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Generate from a prompt (wraps as chat)."""
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str | None:
        """LLMProvider-compatible call() interface."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            logger.warning("Local LLM call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the server is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models", method="GET",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list[str]:
        """List available models on the server."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models", method="GET",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                models = body.get("data", [])
                return [m.get("id", "unknown") for m in models]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    @classmethod
    def auto_detect(cls) -> LocalOpenAIProvider | None:
        """Try all known local servers and return the first one that responds."""
        for server_id, info in _KNOWN_SERVERS.items():
            provider = cls(base_url=info["url"])
            if provider.is_available():
                models = provider.list_models()
                if models:
                    provider.model = models[0]
                logger.info(
                    "Auto-detected %s at %s (model: %s)",
                    info["name"], info["url"], provider.model or "default",
                )
                return provider
        return None

    @staticmethod
    def list_known_servers() -> list[dict]:
        """List all known OpenAI-compatible local servers."""
        return [
            {"id": k, "name": v["name"], "url": v["url"]}
            for k, v in _KNOWN_SERVERS.items()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

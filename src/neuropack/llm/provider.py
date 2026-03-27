"""Universal LLM caller that wraps any registered provider."""
from __future__ import annotations

import logging
from typing import Any

from neuropack.llm.models import LLMConfig

logger = logging.getLogger(__name__)

# Default models per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
    "openai-compatible": "llama3.2",
    "ollama": "qwen3:1.7b",
}


class LLMProvider:
    """Universal LLM provider that dispatches to the right SDK."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None
        self._model = config.model or _DEFAULT_MODELS.get(config.provider, "")

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str | None:
        """Call the LLM and return the response text, or None on error."""
        try:
            if self.config.provider in ("openai", "openai-compatible"):
                return self._call_openai(system, user, max_tokens, temperature)
            elif self.config.provider == "anthropic":
                return self._call_anthropic(system, user, max_tokens, temperature)
            elif self.config.provider == "gemini":
                return self._call_gemini(system, user, max_tokens, temperature)
            elif self.config.provider == "ollama":
                return self._call_ollama(system, user, max_tokens, temperature)
            else:
                logger.warning("Unknown provider: %s", self.config.provider)
                return None
        except Exception as e:
            logger.warning("LLM call failed (%s): %s", self.config.provider, e)
            return None

    def _get_openai_client(self) -> Any:
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {"timeout": self.config.timeout}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.headers:
                kwargs["default_headers"] = self.config.headers
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _call_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str | None:
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str | None:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key, timeout=self.config.timeout
            )
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return resp.content[0].text.strip()

    def _call_gemini(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str | None:
        if self._client is None:
            import google.generativeai as genai

            if self.config.api_key:
                genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(self._model)
        resp = self._client.generate_content(f"{system}\n\n{user}")
        return resp.text.strip()

    def _call_ollama(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str | None:
        if self._client is None:
            from neuropack.llm.ollama import OllamaProvider

            self._client = OllamaProvider.from_config(self.config)
        return self._client.call(system, user, max_tokens, temperature)

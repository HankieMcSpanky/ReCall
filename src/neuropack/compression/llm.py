"""Optional LLM-powered compression for high-quality L3/L2 generation."""
from __future__ import annotations

import json
import logging
from typing import Any

from neuropack.compression.extractive import ExtractiveCompressor

logger = logging.getLogger(__name__)

L3_SYSTEM = (
    "You are a compression engine. Given text, produce a single-sentence abstract "
    "(max 200 chars). Return ONLY the abstract, no quotes."
)
L2_SYSTEM = (
    "You are a fact extractor. Given text, extract 3-5 key facts as a JSON array "
    "of strings. Return ONLY the JSON array."
)


class LLMCompressor:
    """LLM-powered compressor with fallback to extractive."""

    def __init__(
        self,
        provider: str,
        api_key: str = "",
        model: str = "",
        timeout: int = 30,
    ):
        self._provider = provider
        self._api_key = api_key
        self._model = model or self._default_model(provider)
        self._timeout = timeout
        self._fallback = ExtractiveCompressor()
        self._client: Any = None
        self._llm_provider: Any = None  # LLMProvider instance if from_provider

    @classmethod
    def from_provider(cls, provider: object) -> LLMCompressor:
        """Create compressor using a universal LLMProvider instance."""
        instance = cls.__new__(cls)
        instance._provider = "universal"
        instance._api_key = ""
        instance._model = ""
        instance._timeout = 30
        instance._fallback = ExtractiveCompressor()
        instance._client = None
        instance._llm_provider = provider
        return instance

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "gemini": "gemini-2.0-flash",
        }.get(provider, "")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=self._api_key, timeout=self._timeout)
        elif self._provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif self._provider == "gemini":
            import google.generativeai as genai
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model)
        return self._client

    def _call_llm(self, system: str, user: str) -> str | None:
        if self._llm_provider is not None:
            return self._llm_provider.call(system=system, user=user)
        try:
            client = self._get_client()
            if self._provider == "openai":
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=500,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            elif self._provider == "anthropic":
                resp = client.messages.create(
                    model=self._model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=500,
                )
                return resp.content[0].text.strip()
            elif self._provider == "gemini":
                resp = client.generate_content(f"{system}\n\n{user}")
                return resp.text.strip()
        except Exception as e:
            logger.warning("LLM compression failed (%s): %s", self._provider, e)
            return None
        return None

    def compress_l3(self, text: str) -> str:
        result = self._call_llm(L3_SYSTEM, text[:4000])
        if result:
            return result[:200]
        return self._fallback.compress_l3(text)

    def compress_l2(self, text: str) -> list[str]:
        result = self._call_llm(L2_SYSTEM, text[:4000])
        if result:
            try:
                facts = json.loads(result)
                if isinstance(facts, list) and all(isinstance(f, str) for f in facts):
                    return facts[:5]
            except json.JSONDecodeError:
                logger.warning("LLM returned invalid JSON for L2, falling back")
        return self._fallback.compress_l2(text)

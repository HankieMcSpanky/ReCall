"""LLM configuration model."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Named LLM provider configuration."""

    name: str
    provider: str  # "openai" | "anthropic" | "gemini" | "openai-compatible" | "ollama"
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout: int = 30
    headers: dict[str, str] = field(default_factory=dict)
    is_default: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "headers": self.headers,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMConfig:
        return cls(
            name=data["name"],
            provider=data["provider"],
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            base_url=data.get("base_url", ""),
            timeout=data.get("timeout", 30),
            headers=data.get("headers", {}),
            is_default=data.get("is_default", False),
        )

    def masked_key(self) -> str:
        """Return masked API key for display. Never expose full keys."""
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "***"
        return self.api_key[:3] + "..." + self.api_key[-3:]

"""Multi-provider LLM registry backed by SQLite metadata."""
from __future__ import annotations

import json
import logging
import time

from neuropack.llm.models import LLMConfig
from neuropack.storage.database import Database

logger = logging.getLogger(__name__)

REGISTRY_KEY = "llm_registry"


class LLMRegistry:
    """Manages named LLM configurations stored in the metadata table."""

    def __init__(self, db: Database):
        self._db = db

    def _load(self) -> list[LLMConfig]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (REGISTRY_KEY,)
        ).fetchone()
        if row is None:
            return []
        try:
            data = json.loads(dict(row)["value"])
            return [LLMConfig.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError):
            return []

    def _save(self, configs: list[LLMConfig]) -> None:
        value = json.dumps([c.to_dict() for c in configs])
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (REGISTRY_KEY, value),
            )

    def add(self, config: LLMConfig) -> None:
        """Add or update a named LLM config."""
        configs = self._load()
        # Replace existing with same name
        configs = [c for c in configs if c.name != config.name]
        # If this is marked default, clear other defaults
        if config.is_default:
            for c in configs:
                c.is_default = False
        configs.append(config)
        self._save(configs)

    def remove(self, name: str) -> bool:
        """Remove a config by name. Returns True if found."""
        configs = self._load()
        new_configs = [c for c in configs if c.name != name]
        if len(new_configs) == len(configs):
            return False
        self._save(new_configs)
        return True

    def get(self, name: str) -> LLMConfig | None:
        """Get a config by name."""
        for c in self._load():
            if c.name == name:
                return c
        return None

    def list_all(self) -> list[LLMConfig]:
        """List all configs."""
        return self._load()

    def get_default(self) -> LLMConfig | None:
        """Get the default LLM config."""
        configs = self._load()
        for c in configs:
            if c.is_default:
                return c
        # If only one config, treat it as default
        if len(configs) == 1:
            return configs[0]
        return None

    def set_default(self, name: str) -> None:
        """Mark a config as the default."""
        configs = self._load()
        found = False
        for c in configs:
            if c.name == name:
                c.is_default = True
                found = True
            else:
                c.is_default = False
        if not found:
            raise ValueError(f"LLM config '{name}' not found")
        self._save(configs)

    def test_connection(self, name: str) -> dict:
        """Test an LLM connection with a simple prompt."""
        config = self.get(name)
        if config is None:
            return {"ok": False, "error": f"Config '{name}' not found"}

        from neuropack.llm.provider import LLMProvider

        provider = LLMProvider(config)
        start = time.time()
        try:
            result = provider.call(
                system="You are a helpful assistant.",
                user="Say hello in one word.",
                max_tokens=10,
                temperature=0.0,
            )
            elapsed = round(time.time() - start, 2)
            if result:
                return {"ok": True, "response": result, "time_s": elapsed}
            return {"ok": False, "error": "No response received"}
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            return {"ok": False, "error": str(e), "time_s": elapsed}

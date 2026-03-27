"""Tests for LLM registry, models, and provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.llm.models import LLMConfig
from neuropack.llm.registry import LLMRegistry


@pytest.fixture
def registry(store: MemoryStore) -> LLMRegistry:
    return store._llm_registry


class TestLLMConfig:
    def test_to_dict_and_from_dict(self):
        config = LLMConfig(
            name="test-llm",
            provider="openai",
            api_key="sk-abc123def456",
            model="gpt-4o-mini",
        )
        d = config.to_dict()
        restored = LLMConfig.from_dict(d)
        assert restored.name == "test-llm"
        assert restored.provider == "openai"
        assert restored.api_key == "sk-abc123def456"
        assert restored.model == "gpt-4o-mini"

    def test_masked_key(self):
        config = LLMConfig(name="x", provider="openai", api_key="sk-abc123def456789")
        assert config.masked_key() == "sk-...789"

    def test_masked_key_empty(self):
        config = LLMConfig(name="x", provider="openai", api_key="")
        assert config.masked_key() == ""

    def test_masked_key_short(self):
        config = LLMConfig(name="x", provider="openai", api_key="short")
        assert config.masked_key() == "***"


class TestLLMRegistry:
    def test_add_and_get(self, registry: LLMRegistry):
        config = LLMConfig(name="gpt4", provider="openai", api_key="key123")
        registry.add(config)
        result = registry.get("gpt4")
        assert result is not None
        assert result.name == "gpt4"
        assert result.provider == "openai"

    def test_add_replaces_existing(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="test", provider="openai", model="old"))
        registry.add(LLMConfig(name="test", provider="openai", model="new"))
        result = registry.get("test")
        assert result.model == "new"
        assert len(registry.list_all()) == 1

    def test_remove(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="temp", provider="openai"))
        assert registry.remove("temp") is True
        assert registry.get("temp") is None

    def test_remove_nonexistent(self, registry: LLMRegistry):
        assert registry.remove("nope") is False

    def test_list_all(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="a", provider="openai"))
        registry.add(LLMConfig(name="b", provider="anthropic"))
        all_configs = registry.list_all()
        assert len(all_configs) == 2

    def test_get_default(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="a", provider="openai", is_default=False))
        registry.add(LLMConfig(name="b", provider="openai", is_default=True))
        default = registry.get_default()
        assert default is not None
        assert default.name == "b"

    def test_get_default_single_config(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="only", provider="openai"))
        default = registry.get_default()
        assert default is not None
        assert default.name == "only"

    def test_set_default(self, registry: LLMRegistry):
        registry.add(LLMConfig(name="a", provider="openai", is_default=True))
        registry.add(LLMConfig(name="b", provider="openai"))
        registry.set_default("b")
        assert registry.get_default().name == "b"
        # Old default should be cleared
        a = registry.get("a")
        assert a.is_default is False

    def test_set_default_not_found(self, registry: LLMRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.set_default("nope")

    def test_test_connection_not_found(self, registry: LLMRegistry):
        result = registry.test_connection("nope")
        assert result["ok"] is False
        assert "not found" in result["error"]

    @patch("neuropack.llm.provider.LLMProvider.call", return_value="Hello!")
    def test_test_connection_ok(self, mock_call, registry: LLMRegistry):
        registry.add(LLMConfig(name="mock-llm", provider="openai", api_key="key"))
        result = registry.test_connection("mock-llm")
        assert result["ok"] is True
        assert "Hello" in result["response"]

    @patch("neuropack.llm.provider.LLMProvider.call", side_effect=Exception("timeout"))
    def test_test_connection_fail(self, mock_call, registry: LLMRegistry):
        registry.add(LLMConfig(name="fail-llm", provider="openai", api_key="key"))
        result = registry.test_connection("fail-llm")
        assert result["ok"] is False


class TestLegacyMigration:
    def test_legacy_config_migrated_to_registry(self, tmp_path):
        """When legacy config has llm_provider set and registry is empty, auto-migrate."""
        config = NeuropackConfig(
            db_path=str(tmp_path / "test.db"),
            llm_provider="openai",
            llm_api_key="sk-test123",
            llm_model="gpt-4o-mini",
        )
        store = MemoryStore(config)
        store.initialize()
        try:
            configs = store._llm_registry.list_all()
            assert len(configs) == 1
            assert configs[0].name == "default"
            assert configs[0].provider == "openai"
            assert configs[0].is_default is True
        finally:
            store.close()


class TestLLMCompressorFromProvider:
    def test_from_provider_uses_universal(self):
        from neuropack.compression.llm import LLMCompressor

        mock_provider = MagicMock()
        mock_provider.call.return_value = "Test abstract"

        compressor = LLMCompressor.from_provider(mock_provider)
        result = compressor.compress_l3("some text about things")
        assert result == "Test abstract"
        mock_provider.call.assert_called_once()

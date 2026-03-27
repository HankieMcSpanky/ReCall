"""Tests for API key management."""
from __future__ import annotations

from pathlib import Path

import pytest

from neuropack.auth.keys import APIKeyManager
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.storage.database import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(str(tmp_path / "keys.db"))
    d.initialize_schema()
    return d


@pytest.fixture
def key_manager(db: Database) -> APIKeyManager:
    return APIKeyManager(db)


class TestAPIKeyManager:
    def test_create_key(self, key_manager):
        raw_key = key_manager.create_key("test-agent", ["read", "write"])
        assert raw_key.startswith("np_")
        assert len(raw_key) > 10

    def test_validate_key(self, key_manager):
        raw_key = key_manager.create_key("agent1", ["read"])
        result = key_manager.validate_key(raw_key)
        assert result is not None
        assert result["name"] == "agent1"
        assert result["scopes"] == ["read"]

    def test_validate_invalid_key(self, key_manager):
        result = key_manager.validate_key("np_invalid_key_12345")
        assert result is None

    def test_validate_non_np_key(self, key_manager):
        result = key_manager.validate_key("not-an-np-key")
        assert result is None

    def test_list_keys(self, key_manager):
        key_manager.create_key("agent1", ["read"])
        key_manager.create_key("agent2", ["read", "write"])
        keys = key_manager.list_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"agent1", "agent2"}

    def test_revoke_key(self, key_manager):
        raw_key = key_manager.create_key("to-revoke", ["read"])
        assert key_manager.revoke_key("to-revoke") is True
        # Key should no longer validate
        assert key_manager.validate_key(raw_key) is None
        # Should not appear in list
        assert len(key_manager.list_keys()) == 0

    def test_revoke_nonexistent_key(self, key_manager):
        assert key_manager.revoke_key("nonexistent") is False

    def test_duplicate_name_raises(self, key_manager):
        key_manager.create_key("unique-name", ["read"])
        with pytest.raises(ValueError, match="already exists"):
            key_manager.create_key("unique-name", ["write"])

    def test_invalid_scope_raises(self, key_manager):
        with pytest.raises(ValueError, match="Invalid scopes"):
            key_manager.create_key("bad-scope", ["read", "superadmin"])

    def test_scopes_persisted(self, key_manager):
        raw_key = key_manager.create_key("scoped", ["read", "write", "admin"])
        result = key_manager.validate_key(raw_key)
        assert set(result["scopes"]) == {"read", "write", "admin"}

    def test_key_prefix_stored(self, key_manager):
        raw_key = key_manager.create_key("prefixed", ["read"])
        keys = key_manager.list_keys()
        assert keys[0]["key_prefix"] == raw_key[:8]


class TestAPIKeyIntegration:
    def test_store_has_key_manager(self, store):
        assert hasattr(store, "_api_key_manager")
        assert isinstance(store._api_key_manager, APIKeyManager)

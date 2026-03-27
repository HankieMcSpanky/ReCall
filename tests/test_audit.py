"""Tests for audit logging."""
from __future__ import annotations

from pathlib import Path

import pytest

from neuropack.audit import AuditLogger
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.storage.database import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(str(tmp_path / "audit.db"))
    d.initialize_schema()
    return d


@pytest.fixture
def audit(db: Database) -> AuditLogger:
    return AuditLogger(db)


class TestAuditLogger:
    def test_log_and_query(self, audit):
        audit.log("store", actor="user1", memory_id="abc123", namespace="default")
        entries = audit.query()
        assert len(entries) == 1
        assert entries[0]["action"] == "store"
        assert entries[0]["actor"] == "user1"
        assert entries[0]["memory_id"] == "abc123"

    def test_log_with_details(self, audit):
        audit.log("delete", details={"reason": "cleanup"})
        entries = audit.query()
        assert entries[0]["details"] == {"reason": "cleanup"}

    def test_query_filter_by_action(self, audit):
        audit.log("store", memory_id="1")
        audit.log("delete", memory_id="2")
        audit.log("store", memory_id="3")
        entries = audit.query(action="store")
        assert len(entries) == 2
        assert all(e["action"] == "store" for e in entries)

    def test_query_filter_by_actor(self, audit):
        audit.log("store", actor="agent-a")
        audit.log("store", actor="agent-b")
        entries = audit.query(actor="agent-a")
        assert len(entries) == 1
        assert entries[0]["actor"] == "agent-a"

    def test_count(self, audit):
        assert audit.count() == 0
        audit.log("store")
        audit.log("delete")
        assert audit.count() == 2

    def test_ordering(self, audit):
        """Entries are returned newest-first."""
        audit.log("store", memory_id="first")
        audit.log("store", memory_id="second")
        entries = audit.query()
        assert entries[0]["memory_id"] == "second"
        assert entries[1]["memory_id"] == "first"


class TestAuditIntegration:
    def test_store_creates_audit_entry(self, store):
        store.store(content="audit test", tags=["test"])
        entries = store._audit.query(action="store")
        assert len(entries) == 1

    def test_delete_creates_audit_entry(self, store):
        record = store.store(content="to delete", tags=["test"])
        store.forget(record.id)
        entries = store._audit.query(action="delete")
        assert len(entries) == 1
        assert entries[0]["memory_id"] == record.id

    def test_update_creates_audit_entry(self, store):
        record = store.store(content="to update", tags=["test"])
        store.update(record.id, content="updated content")
        entries = store._audit.query(action="update")
        assert len(entries) == 1

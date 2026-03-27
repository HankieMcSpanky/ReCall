"""Tests for agent-managed memory lifecycle (Phase 5)."""
from __future__ import annotations

import json

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def _create_agent_memory(store, agent="tester"):
    record = store.store(
        "Important finding about API design.",
        tags=["observation"],
        source=f"agent:{agent}",
        namespace=agent,
    )
    return record


def test_promote_boosts_priority(store):
    record = _create_agent_memory(store)
    mgr = store.agent_memory("tester")

    ok = mgr.promote(record.id, priority=0.9)
    assert ok is True

    # Verify priority increased
    conn = store._db.connect()
    row = conn.execute("SELECT priority, staleness, tags FROM memories WHERE id = ?", (record.id,)).fetchone()
    d = dict(row)
    assert d["priority"] >= 0.9
    assert d["staleness"] == "stable"
    assert "promoted" in json.loads(d["tags"])


def test_promote_wrong_namespace(store):
    record = _create_agent_memory(store, agent="other")
    mgr = store.agent_memory("tester")
    ok = mgr.promote(record.id)
    assert ok is False  # Can't promote another agent's memory


def test_demote_lowers_priority(store):
    record = _create_agent_memory(store)
    mgr = store.agent_memory("tester")

    ok = mgr.demote(record.id, priority=0.1)
    assert ok is True

    conn = store._db.connect()
    row = conn.execute("SELECT priority, staleness FROM memories WHERE id = ?", (record.id,)).fetchone()
    d = dict(row)
    assert d["priority"] <= 0.2
    assert d["staleness"] == "volatile"


def test_archive_saves_version(store):
    record = _create_agent_memory(store)
    mgr = store.agent_memory("tester")

    ok = mgr.archive(record.id, reason="outdated")
    assert ok is True

    # Verify version saved
    conn = store._db.connect()
    versions = conn.execute(
        "SELECT * FROM memory_versions WHERE memory_id = ?", (record.id,)
    ).fetchall()
    assert len(versions) == 1
    v = dict(versions[0])
    assert v["reason"] == "outdated"

    # Verify tagged
    row = conn.execute("SELECT tags, priority FROM memories WHERE id = ?", (record.id,)).fetchone()
    d = dict(row)
    assert "archived" in json.loads(d["tags"])
    assert d["priority"] == 0.1


def test_pin_sets_max_priority(store):
    record = _create_agent_memory(store)
    mgr = store.agent_memory("tester")

    ok = mgr.pin(record.id)
    assert ok is True

    conn = store._db.connect()
    row = conn.execute("SELECT priority, staleness, tags FROM memories WHERE id = ?", (record.id,)).fetchone()
    d = dict(row)
    assert d["priority"] == 1.0
    assert d["staleness"] == "stable"
    assert "pinned" in json.loads(d["tags"])


def test_get_pinned(store):
    r1 = _create_agent_memory(store)
    r2 = store.store("Another memory", tags=["test"], namespace="tester")

    mgr = store.agent_memory("tester")
    mgr.pin(r1.id)

    pinned = mgr.get_pinned()
    assert len(pinned) == 1
    assert pinned[0]["id"] == r1.id


def test_create_working_memory(store):
    mgr = store.agent_memory("tester")

    mid = mgr.create_working_memory("Temporary scratchpad data")
    assert mid is not None

    conn = store._db.connect()
    row = conn.execute("SELECT staleness, memory_type, tags FROM memories WHERE id = ?", (mid,)).fetchone()
    d = dict(row)
    assert d["staleness"] == "volatile"
    assert d["memory_type"] == "observation"
    assert "working_memory" in json.loads(d["tags"])


def test_agent_memory_factory(store):
    """store.agent_memory() returns an AgentMemoryManager."""
    from neuropack.core.agent_memory import AgentMemoryManager
    mgr = store.agent_memory("myagent")
    assert isinstance(mgr, AgentMemoryManager)

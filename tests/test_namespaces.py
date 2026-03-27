from __future__ import annotations

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@pytest.fixture
def ns_store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), namespace="default")
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def test_store_default_namespace(ns_store):
    record = ns_store.store("Hello world", tags=["test"])
    assert record.namespace == "default"


def test_store_custom_namespace(ns_store):
    record = ns_store.store("Agent 1 data", tags=["test"], namespace="agent1")
    assert record.namespace == "agent1"


def test_namespace_isolation(ns_store):
    ns_store.store("The quick brown fox jumps over the lazy dog in the forest", tags=["a1"], namespace="agent1")
    ns_store.store("SQLite uses B-trees for indexing and full-text search via FTS5", tags=["a2"], namespace="agent2")

    a1_records = ns_store.list(namespace="agent1")
    a2_records = ns_store.list(namespace="agent2")

    assert len(a1_records) == 1
    assert a1_records[0].namespace == "agent1"
    assert len(a2_records) == 1
    assert a2_records[0].namespace == "agent2"


def test_recall_with_namespace_filter(ns_store):
    ns_store.store("Python programming guide", tags=["tech"], namespace="agent1")
    ns_store.store("Python snake facts", tags=["animals"], namespace="agent2")

    results = ns_store.recall("Python", namespaces=["agent1"])
    assert len(results) >= 1
    assert all(r.record.namespace == "agent1" for r in results)


def test_recall_without_namespace_finds_all(ns_store):
    ns_store.store("Alpha data", tags=["test"], namespace="ns1")
    ns_store.store("Beta data", tags=["test"], namespace="ns2")

    results = ns_store.recall("data")
    assert len(results) == 2


def test_share_memory(ns_store):
    record = ns_store.store("Shared knowledge", tags=["shared"], namespace="agent1")
    shared = ns_store.share_memory(record.id, "shared")

    assert shared.namespace == "shared"
    assert shared.id != record.id
    assert shared.content == record.content


def test_list_namespaces(ns_store):
    ns_store.store("A", tags=["test"], namespace="ns1")
    ns_store.store("B", tags=["test"], namespace="ns1")
    ns_store.store("C", tags=["test"], namespace="ns2")

    ns_list = ns_store.list_namespaces()
    ns_map = {n["namespace"]: n["count"] for n in ns_list}

    assert ns_map["ns1"] == 2
    assert ns_map["ns2"] == 1


def test_stats_per_namespace(ns_store):
    ns_store.store("Python is a programming language created by Guido van Rossum", tags=["test"], namespace="ns1")
    ns_store.store("SQLite uses B-trees for indexing and supports full-text search via FTS5", tags=["test"], namespace="ns1")
    ns_store.store("The weather in Tokyo is warm and humid in summer with cherry blossoms in spring", tags=["test"], namespace="ns2")

    stats_ns1 = ns_store.stats(namespace="ns1")
    stats_ns2 = ns_store.stats(namespace="ns2")
    stats_all = ns_store.stats()

    assert stats_ns1.total_memories == 2
    assert stats_ns2.total_memories == 1
    assert stats_all.total_memories == 3


def test_config_namespace(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), namespace="myagent")
    s = MemoryStore(config)
    s.initialize()

    record = s.store("Test data", tags=["test"])
    assert record.namespace == "myagent"
    s.close()


def test_context_summary_namespace(ns_store):
    ns_store.store("Context A", tags=["test"], namespace="ns1")
    ns_store.store("Context B", tags=["test"], namespace="ns2")

    summary = ns_store.context_summary(namespace="ns1")
    assert len(summary) == 1

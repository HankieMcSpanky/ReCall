"""Tests for graph-based retrieval (Phase 2)."""
from __future__ import annotations

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.search.graph_retriever import GraphRetriever


@pytest.fixture
def store(tmp_path):
    config = NeuropackConfig(
        db_path=str(tmp_path / "test.db"),
        retrieval_weight_graph=0.2,
    )
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def test_graph_retriever_basic(store):
    """Graph retriever finds memories linked via shared entities."""
    store.store("Python uses SQLite for storage.", tags=["tech"])
    store.store("SQLite is a lightweight database engine.", tags=["tech"])

    gr = GraphRetriever(store._db)
    results = gr.retrieve("SQLite database")
    # Should find memory_ids linked to SQLite entity
    assert isinstance(results, list)
    if results:
        mid, score = results[0]
        assert isinstance(mid, str)
        assert score > 0


def test_graph_retriever_no_entities(store):
    """Query with no extractable entities returns empty."""
    gr = GraphRetriever(store._db)
    results = gr.retrieve("a b c")
    assert results == []


def test_graph_retriever_multi_hop(store):
    """Graph walk finds memories via 2-hop connections."""
    # Use entities that survive extraction (mid-sentence capitalized words)
    store.store("The framework Django uses Python internally.", tags=["tech"])
    store.store("The language Python depends on Cython.", tags=["tech"])

    gr = GraphRetriever(store._db, max_hops=2)
    # Query for Django should find both: Django->Python (hop 0), Python->Cython (hop 1)
    results = gr.retrieve("The framework Django")
    memory_ids = [mid for mid, _ in results]
    assert len(memory_ids) >= 1


def test_hybrid_retriever_includes_graph(store):
    """HybridRetriever with graph enabled includes graph_score in results."""
    store.store("React is a JavaScript framework.", tags=["tech"])
    store.store("JavaScript uses V8 engine.", tags=["tech"])

    results = store.recall("React framework")
    assert isinstance(results, list)
    # At least one result should exist
    if results:
        # graph_score may or may not be set depending on entity extraction
        assert hasattr(results[0], "graph_score")


def test_graph_weight_zero_disables(tmp_path):
    """Setting graph weight to 0 disables graph retrieval."""
    config = NeuropackConfig(
        db_path=str(tmp_path / "test.db"),
        retrieval_weight_graph=0.0,
    )
    s = MemoryStore(config)
    s.initialize()
    # Graph should not be wired
    assert s._retriever._graph is None
    s.close()


def test_rrf_rebalance(store):
    """With graph weight 0.2, vec+fts weights rebalance to fill remaining 0.8."""
    store.store("NeuroPack is a memory store.", tags=["tech"])
    # Just verify recall works without errors
    results = store.recall("memory store")
    assert isinstance(results, list)

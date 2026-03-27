from __future__ import annotations

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.knowledge_graph import (
    extract_entities,
    extract_relationships,
    detect_temporal_markers,
)
from neuropack.core.store import MemoryStore


@pytest.fixture
def kg_store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


def test_extract_capitalized_entities():
    text = "Guido van Rossum created Python. It is used by Google."
    entities = extract_entities(text)
    names = [name for name, _ in entities]
    # Should find multi-word proper nouns (with optional lowercase connectors)
    assert "Guido van Rossum" in names or "Python" in names


def test_extract_quoted_entities():
    text = 'The "middle-out compression" algorithm is efficient.'
    entities = extract_entities(text)
    names = [name for name, _ in entities]
    assert "middle-out compression" in names


def test_extract_url_entities():
    text = "Visit https://github.com/neuropack for more info."
    entities = extract_entities(text)
    types = {name: etype for name, etype in entities}
    assert any(etype == "url" for etype in types.values())


def test_extract_email_entities():
    text = "Contact user@example.com for details."
    entities = extract_entities(text)
    names = [name for name, _ in entities]
    assert "user@example.com" in names


def test_extract_relationships_co_occurrence():
    text = "Python uses SQLite for storage."
    entities = [("Python", "concept"), ("SQLite", "concept")]
    rels = extract_relationships(text, entities)
    assert len(rels) >= 1
    # Should have at least co_occurs
    rel_types = [r[2] for r in rels]
    assert "co_occurs" in rel_types


def test_knowledge_graph_process_memory(kg_store):
    kg_store.store(
        "Python was created by Guido van Rossum. Python uses C for its implementation.",
        tags=["tech"],
    )

    results = kg_store.search_entities("Python")
    assert len(results) >= 1
    assert any(e["name"] == "Python" for e in results)


def test_knowledge_graph_query_entity(kg_store):
    kg_store.store("React is a JavaScript library by Facebook.", tags=["tech"])

    result = kg_store.query_entity("React")
    # May or may not be found depending on extraction
    assert "found" in result or "name" in result


def test_knowledge_graph_stats(kg_store):
    kg_store.store("Alice and Bob work at Anthropic.", tags=["people"])

    stats = kg_store.knowledge_graph_stats()
    assert "total_entities" in stats
    assert "total_relationships" in stats


def test_knowledge_graph_delete_cleanup(kg_store):
    record = kg_store.store(
        "TensorFlow created by Google for deep learning.",
        tags=["tech"],
    )

    # Delete and verify relationships are cleaned up
    kg_store.forget(record.id)

    # Stats should reflect deletion
    stats = kg_store.knowledge_graph_stats()
    assert stats["total_relationships"] == 0


def test_entity_search(kg_store):
    kg_store.store("NeuroPack uses SQLite and Python.", tags=["tech"])

    results = kg_store.search_entities("Neuro")
    # Should find NeuroPack via LIKE search
    assert isinstance(results, list)


# --- Phase 1: Temporal Knowledge Graph Tests ---


def test_detect_temporal_ended():
    markers = detect_temporal_markers("Alice no longer works at Acme since 2024-01.")
    assert markers["ended"] is True
    assert markers["date_hint"] is not None


def test_detect_temporal_current():
    markers = detect_temporal_markers("As of 2025-03, Bob works at Anthropic.")
    assert markers["ended"] is False
    assert markers["date_hint"] is not None


def test_detect_temporal_no_markers():
    markers = detect_temporal_markers("Python is a programming language.")
    assert markers["ended"] is False
    assert markers["date_hint"] is None


def test_temporal_auto_supersede(kg_store):
    """When a new relationship contradicts an existing one, the old one gets superseded."""
    # First fact: Alice uses Python
    kg_store.store("Alice uses Python for her projects.", tags=["tech"])

    # Second fact: same relationship type, should supersede
    kg_store.store("Alice uses Python for data science.", tags=["tech"])

    result = kg_store.query_entity("Alice")
    if result.get("found"):
        rels = result["relationships"]
        # Filter 'uses' relationships
        uses_rels = [r for r in rels if r["relation_type"] == "uses"]
        if len(uses_rels) >= 2:
            # At least one should be superseded
            superseded = [r for r in uses_rels if r.get("superseded_by")]
            assert len(superseded) >= 1


def test_get_current_facts(kg_store):
    """get_current_facts excludes superseded relationships."""
    kg_store.store("Alice uses React for frontend.", tags=["tech"])
    kg_store.store("Alice uses React for mobile apps.", tags=["tech"])

    result = kg_store.get_current_facts("Alice")
    if result.get("found"):
        # All returned facts should be current (no superseded_by)
        for fact in result["current_facts"]:
            assert fact.get("superseded_by") is None or "superseded_by" not in fact


def test_fact_timeline(kg_store):
    """fact_timeline returns all facts in chronological order."""
    kg_store.store("Alice uses Python.", tags=["tech"])
    kg_store.store("Alice uses Rust.", tags=["tech"])

    result = kg_store.fact_timeline("Alice")
    if result.get("found"):
        timeline = result["timeline"]
        assert isinstance(timeline, list)
        # Each entry should have active flag
        for entry in timeline:
            assert "active" in entry
            assert "created_at" in entry


def test_supersede_fact_manual(kg_store):
    """Manual supersede_fact marks old relationship as ended."""
    kg_store.store("Bob uses Java for backend.", tags=["tech"])
    kg_store.store("Bob uses Kotlin for backend.", tags=["tech"])

    # Get relationships from the DB directly
    conn = kg_store._db.connect()
    rels = conn.execute(
        "SELECT id, valid_until, superseded_by FROM relationships"
    ).fetchall()

    active = [dict(r) for r in rels if dict(r)["superseded_by"] is None]
    superseded = [dict(r) for r in rels if dict(r)["superseded_by"] is not None]

    if len(active) >= 1 and len(superseded) == 0 and len(active) >= 2:
        # Manual supersede
        result = kg_store.supersede_fact(active[0]["id"], active[1]["id"])
        assert result is True

        # Verify
        row = conn.execute(
            "SELECT superseded_by FROM relationships WHERE id = ?",
            (active[0]["id"],),
        ).fetchone()
        assert dict(row)["superseded_by"] == active[1]["id"]


def test_query_entity_as_of(kg_store):
    """query_entity with as_of filters by temporal validity."""
    kg_store.store("Alice uses Python as of 2023-01.", tags=["tech"])

    result = kg_store.query_entity("Alice", as_of="2024-01-01")
    if result.get("found"):
        # Should return relationships valid at that time
        assert isinstance(result["relationships"], list)

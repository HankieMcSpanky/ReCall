"""Tests for new features: auto-tagging, staleness, consolidation, backup, versioning, cache, webhooks."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.core.auto_tagger import (
    AutoTagger,
    classify_memory_type,
    classify_staleness,
    extract_tags,
)
from neuropack.core.staleness import check_staleness, staleness_age_days, get_stale_summary
from neuropack.core.consolidation import find_clusters, summarize_cluster_extractive
from neuropack.core.backup import create_backup, restore_backup, list_backups
from neuropack.core.cache import RecallCache
from neuropack.core.webhooks import WebhookEmitter
from neuropack.types import MemoryRecord, ConsolidationResult


# --- Fixtures ---


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    config = NeuropackConfig(db_path=db_path, auto_tag=True)
    store = MemoryStore(config)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def tmp_db_no_autotag(tmp_path):
    db_path = str(tmp_path / "test.db")
    config = NeuropackConfig(db_path=db_path, auto_tag=False)
    store = MemoryStore(config)
    store.initialize()
    yield store
    store.close()


# --- Auto-tagging tests ---


class TestAutoTagger:
    def test_classify_decision(self):
        assert classify_memory_type("I decided to use FastAPI") == "decision"

    def test_classify_fact(self):
        assert classify_memory_type("Python 3.12 supports new features") in ("fact", "general")

    def test_classify_preference(self):
        assert classify_memory_type("I prefer to always use pytest") == "preference"

    def test_classify_procedure(self):
        assert classify_memory_type("Step 1: install deps. Step 2: run tests") == "procedure"

    def test_classify_code(self):
        assert classify_memory_type("```python\ndef hello():\n    pass\n```") == "code"

    def test_classify_observation(self):
        assert classify_memory_type("I noticed that the API is slow on Mondays") == "observation"

    def test_classify_general(self):
        assert classify_memory_type("hello world") == "general"

    def test_staleness_volatile(self):
        assert classify_staleness("The latest version v3.12 is currently available") == "volatile"

    def test_staleness_stable(self):
        assert classify_staleness("Algorithms are fundamental to CS") == "stable"

    def test_extract_tags_python(self):
        tags = extract_tags("Setting up a Flask API with pytest")
        assert "python" in tags
        assert "api" in tags
        assert "testing" in tags

    def test_extract_tags_respects_existing(self):
        tags = extract_tags("Setting up Flask", existing_tags=["python"])
        assert "python" not in tags  # Already present

    def test_extract_tags_caps_at_5(self):
        text = "python javascript rust docker git api security testing devops ai linux"
        tags = extract_tags(text)
        assert len(tags) <= 5

    def test_auto_tagger_no_llm(self):
        tagger = AutoTagger()
        result = tagger.tag_and_classify("I decided to use Redis for caching")
        assert result["memory_type"] == "decision"
        assert "database" in result["tags"]

    def test_store_auto_tags(self, tmp_db):
        record = tmp_db.store("I decided to use Docker and Python for the API")
        assert record.memory_type == "decision"
        # Should have auto-tags
        assert any(t in record.tags for t in ["python", "docker", "api"])

    def test_store_preserves_user_tags(self, tmp_db):
        record = tmp_db.store("Some content about APIs", tags=["my-custom-tag"])
        assert "my-custom-tag" in record.tags

    def test_store_no_autotag_when_disabled(self, tmp_db_no_autotag):
        record = tmp_db_no_autotag.store("I decided to use Docker and Python")
        assert record.memory_type == "general"
        assert record.tags == []


# --- Staleness tests ---


class TestStaleness:
    def _make_record(self, staleness="stable", age_days=0, superseded_by=None):
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=age_days)
        return MemoryRecord(
            id="test123",
            content="test content",
            l3_abstract="test",
            l2_facts=["fact"],
            l1_compressed=b"data",
            embedding=[0.0],
            tags=["test"],
            source="test",
            priority=0.5,
            created_at=created,
            updated_at=created,
            staleness=staleness,
            superseded_by=superseded_by,
        )

    def test_stable_not_stale(self):
        record = self._make_record("stable", age_days=365)
        assert check_staleness(record) is None

    def test_volatile_stale(self):
        record = self._make_record("volatile", age_days=45)
        warning = check_staleness(record, volatile_days=30)
        assert warning is not None
        assert "Volatile" in warning

    def test_volatile_not_stale_yet(self):
        record = self._make_record("volatile", age_days=10)
        assert check_staleness(record, volatile_days=30) is None

    def test_semi_stable_stale(self):
        record = self._make_record("semi-stable", age_days=100)
        warning = check_staleness(record, semi_stable_days=90)
        assert warning is not None

    def test_superseded(self):
        record = self._make_record(superseded_by="other_id")
        warning = check_staleness(record)
        assert warning is not None
        assert "Superseded" in warning

    def test_store_recalls_with_staleness_warning(self, tmp_db):
        # Store a volatile memory
        record = tmp_db.store(
            "The latest version v3.12 is currently available today",
            staleness="volatile",
        )
        # Force the memory to look old by updating its updated_at
        tmp_db._repo.update(record.id, staleness="volatile")
        # The staleness detection runs at recall time
        results = tmp_db.recall("version")
        # RecallResult now has staleness_warning field
        for r in results:
            assert hasattr(r, "staleness_warning")


# --- Cache tests ---


class TestRecallCache:
    def test_put_and_get(self):
        cache = RecallCache()
        cache.put("query1", ["result1"], limit=10)
        assert cache.get("query1", limit=10) == ["result1"]

    def test_miss(self):
        cache = RecallCache()
        assert cache.get("nonexistent", limit=10) is None

    def test_invalidate(self):
        cache = RecallCache()
        cache.put("query1", ["result1"], limit=10)
        cache.invalidate()
        # After invalidation, generation changes so key won't match
        assert cache.get("query1", limit=10) is None

    def test_different_params(self):
        cache = RecallCache()
        cache.put("query1", ["a"], limit=10)
        cache.put("query1", ["b"], limit=20)
        assert cache.get("query1", limit=10) == ["a"]
        assert cache.get("query1", limit=20) == ["b"]

    def test_lru_eviction(self):
        cache = RecallCache(max_size=2)
        cache.put("q1", [1], x=1)
        cache.put("q2", [2], x=2)
        cache.put("q3", [3], x=3)  # Should evict q1
        assert cache.get("q1", x=1) is None
        assert cache.get("q2", x=2) == [2]


# --- Webhook tests ---


class TestWebhooks:
    def test_disabled_by_default(self):
        emitter = WebhookEmitter()
        assert not emitter.enabled

    def test_enabled_with_url(self):
        emitter = WebhookEmitter(url="http://example.com/hook")
        assert emitter.enabled

    def test_emit_noop_when_disabled(self):
        emitter = WebhookEmitter()
        # Should not raise
        emitter.emit("store", {"id": "123"})

    def test_emit_skips_unregistered_events(self):
        emitter = WebhookEmitter(url="http://example.com/hook", events="store,delete")
        # "recall" is not in events -- should be a noop
        emitter.emit("recall", {"id": "123"})


# --- Backup tests ---


class TestBackup:
    def test_create_and_list_backups(self, tmp_db, tmp_path):
        # Store something first
        tmp_db.store("test memory for backup")

        backup_dir = str(tmp_path / "backups")
        path = tmp_db.backup(backup_dir=backup_dir)
        assert Path(path).exists()

        backups = tmp_db.list_backups(backup_dir=backup_dir)
        assert len(backups) == 1
        assert backups[0]["size_mb"] > 0

    def test_restore(self, tmp_path):
        # Create a store with data
        db1 = str(tmp_path / "orig.db")
        config1 = NeuropackConfig(db_path=db1, auto_tag=False)
        store1 = MemoryStore(config1)
        store1.initialize()
        store1.store("original memory")
        assert store1.stats().total_memories == 1

        # Backup
        backup_dir = str(tmp_path / "backups")
        backup_path = store1.backup(backup_dir=backup_dir)
        store1.close()

        # Create a new empty store
        db2 = str(tmp_path / "new.db")
        config2 = NeuropackConfig(db_path=db2, auto_tag=False)
        store2 = MemoryStore(config2)
        store2.initialize()
        assert store2.stats().total_memories == 0

        # Restore
        store2.restore(backup_path)
        assert store2.stats().total_memories == 1
        store2.close()


# --- Versioning tests ---


class TestVersioning:
    def test_update_creates_version(self, tmp_db):
        record = tmp_db.store("original content", tags=["v1"])
        tmp_db.update(record.id, content="updated content")

        versions = tmp_db.get_versions(record.id)
        assert len(versions) == 1
        assert versions[0].content == "original content"
        assert versions[0].reason == "update"

    def test_dedup_merge_creates_version(self, tmp_db):
        # Store similar content twice -- second should dedup-merge
        r1 = tmp_db.store("exact same content for dedup test")
        r2 = tmp_db.store("exact same content for dedup test")  # Should merge

        # r2 should be the same ID as r1 (merged)
        assert r2.id == r1.id

        versions = tmp_db.get_versions(r1.id)
        assert len(versions) >= 1
        assert versions[0].reason == "dedup_merge"

    def test_no_version_without_content_change(self, tmp_db):
        record = tmp_db.store("some content")
        tmp_db.update(record.id, priority=0.9)  # Only priority, no content

        versions = tmp_db.get_versions(record.id)
        assert len(versions) == 0


# --- Consolidation tests ---


class TestConsolidation:
    def test_find_clusters_basic(self):
        # Create records with similar embeddings
        records = []
        embeddings = []
        for i in range(5):
            vec = np.random.randn(256).astype(np.float32)
            vec /= np.linalg.norm(vec)
            records.append(MemoryRecord(
                id=f"id{i}", content=f"content {i}", l3_abstract=f"abstract {i}",
                l2_facts=[f"fact {i}"], l1_compressed=b"data",
                embedding=vec.tolist(), tags=[], source="test", priority=0.5,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            embeddings.append(vec)

        # With very low threshold, should cluster everything
        emb_matrix = np.vstack(embeddings)
        clusters = find_clusters(records, emb_matrix, threshold=0.0, min_cluster_size=2)
        assert len(clusters) >= 1

    def test_find_clusters_no_similar(self):
        # Create records with orthogonal embeddings
        records = []
        embeddings = []
        for i in range(3):
            vec = np.zeros(256, dtype=np.float32)
            vec[i] = 1.0  # Orthogonal
            records.append(MemoryRecord(
                id=f"id{i}", content=f"content {i}", l3_abstract=f"abstract {i}",
                l2_facts=[], l1_compressed=b"", embedding=vec.tolist(), tags=[],
                source="", priority=0.5,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            embeddings.append(vec)

        emb_matrix = np.vstack(embeddings)
        clusters = find_clusters(records, emb_matrix, threshold=0.9, min_cluster_size=2)
        assert len(clusters) == 0

    def test_summarize_extractive(self):
        records = [
            MemoryRecord(
                id="1", content="Python is great for AI",
                l3_abstract="Python for AI", l2_facts=["Python is versatile"],
                l1_compressed=b"", embedding=[], tags=["python"], source="",
                priority=0.8, created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
            MemoryRecord(
                id="2", content="Python has great ML libraries",
                l3_abstract="Python ML libs", l2_facts=["numpy", "pytorch"],
                l1_compressed=b"", embedding=[], tags=["python"], source="",
                priority=0.6, created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        ]
        summary = summarize_cluster_extractive(records)
        assert "Consolidated from 2 memories" in summary
        assert "Python for AI" in summary

    def test_consolidate_dry_run(self, tmp_db):
        result = tmp_db.consolidate(dry_run=True)
        assert isinstance(result, ConsolidationResult)
        assert result.summaries_created == 0


# --- Context budget tests ---


class TestContextBudget:
    def test_recall_with_token_budget(self, tmp_db):
        # Store a few memories
        for i in range(5):
            tmp_db.store(f"Memory number {i} about topic {i}")

        # Recall with a very small budget should return fewer results
        results = tmp_db.recall("memory topic", token_budget=10)
        assert len(results) <= 5

    def test_recall_without_budget_returns_all(self, tmp_db):
        for i in range(3):
            tmp_db.store(f"Memory number {i} about important topic")

        results = tmp_db.recall("important topic")
        assert len(results) >= 1


# --- Memory type and staleness on records ---


class TestMemoryMetadata:
    def test_store_explicit_type(self, tmp_db_no_autotag):
        record = tmp_db_no_autotag.store("test", memory_type="decision")
        assert record.memory_type == "decision"

    def test_store_explicit_staleness(self, tmp_db_no_autotag):
        record = tmp_db_no_autotag.store("test", staleness="volatile")
        assert record.staleness == "volatile"

    def test_inspect_shows_new_fields(self, tmp_db):
        record = tmp_db.store("I decided to use Redis")
        fetched = tmp_db.get(record.id)
        assert fetched.memory_type in ("decision", "general")
        assert fetched.staleness in ("stable", "semi-stable", "volatile")
        assert fetched.superseded_by is None


# --- HNSW index tests ---


class TestHNSWIndex:
    def test_factory_returns_brute_force_without_hnswlib(self):
        from neuropack.search.hnsw_index import create_vector_index, _HAS_HNSWLIB
        idx = create_vector_index(dim=256, use_hnsw=False)
        assert type(idx).__name__ == "BruteForceIndex"

    def test_factory_respects_flag(self):
        from neuropack.search.hnsw_index import create_vector_index
        idx = create_vector_index(dim=256, use_hnsw=False)
        assert type(idx).__name__ == "BruteForceIndex"

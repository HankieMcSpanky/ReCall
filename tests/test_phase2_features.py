"""Tests for Phase 2 features: trust, contradictions, PII, retention, adaptive re-ranking."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.core.trust import TrustScorer, AnomalyDetector, check_memory_trust
from neuropack.core.contradictions import (
    detect_contradictions,
    _extract_key_terms,
    _has_negation,
    _check_opposite_pairs,
    Contradiction,
)
from neuropack.core.pii import detect_pii, redact_content, pii_summary, PIIAction
from neuropack.core.retention import (
    RetentionPolicy,
    find_expired_memories,
    parse_retention_config,
)
from neuropack.core.priority import PriorityScorer
from neuropack.types import MemoryRecord, RecallResult
from neuropack.exceptions import PIIDetectedError


# --- Fixtures ---


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    config = NeuropackConfig(
        db_path=db_path,
        auto_tag=True,
        pii_mode="warn",
        contradiction_check=False,  # Disable by default to avoid test interference
    )
    store = MemoryStore(config)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def tmp_db_pii_block(tmp_path):
    db_path = str(tmp_path / "test_pii.db")
    config = NeuropackConfig(db_path=db_path, auto_tag=False, pii_mode="block")
    store = MemoryStore(config)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def tmp_db_pii_redact(tmp_path):
    db_path = str(tmp_path / "test_redact.db")
    config = NeuropackConfig(db_path=db_path, auto_tag=False, pii_mode="redact")
    store = MemoryStore(config)
    store.initialize()
    yield store
    store.close()


# --- Trust Scoring Tests ---


class TestTrustScorer:
    def test_initial_trust(self):
        scorer = TrustScorer(prior=0.5)
        assert scorer.trust_score("new_source") == 0.5

    def test_trust_increases_with_success(self):
        scorer = TrustScorer(prior=0.5)
        initial = scorer.trust_score("source1")
        scorer.record_success("source1")
        scorer.record_success("source1")
        assert scorer.trust_score("source1") > initial

    def test_trust_decreases_with_failure(self):
        scorer = TrustScorer(prior=0.5)
        initial = scorer.trust_score("source1")
        scorer.record_failure("source1")
        scorer.record_failure("source1")
        assert scorer.trust_score("source1") < initial

    def test_save_and_load_state(self):
        scorer = TrustScorer()
        scorer.record_success("a")
        scorer.record_failure("b")
        state = scorer.save_state()

        scorer2 = TrustScorer()
        scorer2.load_state(state)
        assert scorer2.trust_score("a") == scorer.trust_score("a")
        assert scorer2.trust_score("b") == scorer.trust_score("b")


class TestAnomalyDetector:
    def test_no_anomaly_with_few_samples(self):
        detector = AnomalyDetector()
        vec = np.random.randn(256).astype(np.float32)
        is_anom, z = detector.is_anomaly(vec)
        assert not is_anom
        assert z == 0.0

    def test_detects_outlier(self):
        detector = AnomalyDetector(threshold_sigma=2.0)
        # Fit with similar vectors
        vecs = [(f"id{i}", np.random.randn(256).astype(np.float32) * 0.1)
                for i in range(50)]
        detector.fit(vecs)

        # Create an obvious outlier
        outlier = np.ones(256, dtype=np.float32) * 100.0
        is_anom, z = detector.is_anomaly(outlier)
        assert is_anom
        assert z > 2.0

    def test_normal_vector_not_anomaly(self):
        detector = AnomalyDetector(threshold_sigma=3.0)
        vecs = [(f"id{i}", np.random.randn(256).astype(np.float32) * 0.1)
                for i in range(50)]
        detector.fit(vecs)

        # A vector from the same distribution
        normal = np.random.randn(256).astype(np.float32) * 0.1
        is_anom, _ = detector.is_anomaly(normal)
        assert not is_anom

    def test_incremental_update(self):
        detector = AnomalyDetector()
        vec = np.ones(256, dtype=np.float32)
        detector.update(vec)
        assert detector._n == 1
        assert detector._centroid is not None


class TestCheckMemoryTrust:
    def test_trusted_memory(self):
        scorer = TrustScorer()
        detector = AnomalyDetector()
        record = MemoryRecord(
            id="test", content="test", l3_abstract="test", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="good_source",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        report = check_memory_trust(record, scorer, detector)
        assert report["is_trusted"]
        assert len(report["warnings"]) == 0


# --- Contradiction Detection Tests ---


class TestContradictions:
    def test_extract_key_terms(self):
        terms = _extract_key_terms("Python is great for AI applications")
        assert "python" in terms
        assert "great" in terms
        assert "is" not in terms  # stopword

    def test_has_negation(self):
        assert _has_negation("This is not working")
        assert _has_negation("Don't use this library")
        assert not _has_negation("This is working well")

    def test_check_opposite_pairs(self):
        result = _check_opposite_pairs("Always use pytest", "Never use pytest")
        assert result is not None
        assert "always" in result.lower() or "never" in result.lower()

    def test_no_opposites(self):
        result = _check_opposite_pairs("Python is great", "Python is wonderful")
        assert result is None

    def test_detect_contradiction_negation(self):
        # Create a mock recall result
        existing = MemoryRecord(
            id="existing1", content="Python is great for machine learning and AI applications",
            l3_abstract="test", l2_facts=[], l1_compressed=b"",
            embedding=[], tags=[], source="", priority=0.5,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        candidates = [RecallResult(record=existing, score=0.85)]

        contradictions = detect_contradictions(
            "Python is not great for machine learning and AI applications",
            candidates,
            similarity_threshold=0.6,
        )
        assert len(contradictions) >= 1

    def test_no_contradiction_similar_topic(self):
        existing = MemoryRecord(
            id="existing1", content="Python is great for web development",
            l3_abstract="test", l2_facts=[], l1_compressed=b"",
            embedding=[], tags=[], source="", priority=0.5,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        candidates = [RecallResult(record=existing, score=0.7)]

        contradictions = detect_contradictions(
            "Python is great for data science",
            candidates,
            similarity_threshold=0.6,
        )
        # Similar but not contradictory
        assert len(contradictions) == 0

    def test_detect_opposite_pairs(self):
        existing = MemoryRecord(
            id="ex1", content="Always use type hints in Python code for better safety",
            l3_abstract="test", l2_facts=[], l1_compressed=b"",
            embedding=[], tags=[], source="", priority=0.5,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        candidates = [RecallResult(record=existing, score=0.8)]

        contradictions = detect_contradictions(
            "Never use type hints in Python code for better simplicity",
            candidates,
            similarity_threshold=0.6,
        )
        assert len(contradictions) >= 1


# --- PII Detection Tests ---


class TestPIIDetection:
    def test_detect_openai_key(self):
        matches = detect_pii("My key is sk-abcdefghijklmnopqrstuvwxyz1234")
        assert len(matches) >= 1
        assert any(m.category == "api_key" for m in matches)

    def test_detect_aws_key(self):
        matches = detect_pii("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert len(matches) >= 1
        assert any(m.category == "api_key" for m in matches)

    def test_detect_email(self):
        matches = detect_pii("Contact me at user@example.com")
        assert len(matches) >= 1
        assert any(m.category == "email" for m in matches)

    def test_detect_private_key(self):
        matches = detect_pii("-----BEGIN RSA PRIVATE KEY-----")
        assert len(matches) >= 1
        assert any(m.category == "private_key" for m in matches)

    def test_detect_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        matches = detect_pii(jwt)
        assert len(matches) >= 1
        assert any(m.category == "jwt" for m in matches)

    def test_no_pii_in_clean_text(self):
        matches = detect_pii("This is a normal memory about Python development")
        assert len(matches) == 0

    def test_redact_content(self):
        text = "My API key is sk-abcdefghijklmnopqrstuvwxyz1234"
        redacted = redact_content(text)
        assert "sk-" not in redacted
        assert "[API_KEY_REDACTED]" in redacted

    def test_pii_summary(self):
        matches = detect_pii("Contact sk-abc12345678901234567890 or user@test.com")
        summary = pii_summary(matches)
        assert "api_key" in summary
        assert "email" in summary

    def test_pii_summary_empty(self):
        assert "No sensitive data" in pii_summary([])

    def test_store_blocks_pii(self, tmp_db_pii_block):
        with pytest.raises(PIIDetectedError):
            tmp_db_pii_block.store("My key is sk-abcdefghijklmnopqrstuvwxyz1234")

    def test_store_redacts_pii(self, tmp_db_pii_redact):
        record = tmp_db_pii_redact.store("My key is sk-abcdefghijklmnopqrstuvwxyz1234")
        assert "sk-" not in record.content
        assert "REDACTED" in record.content

    def test_scan_pii(self, tmp_db):
        tmp_db.store("Contact user@example.com for help")
        results = tmp_db.scan_pii()
        assert len(results) >= 1
        assert results[0]["categories"] == ["email"]


# --- Data Retention Tests ---


class TestRetentionPolicy:
    def test_default_ttl_zero_never_expires(self):
        policy = RetentionPolicy()
        record = MemoryRecord(
            id="test", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        assert policy.effective_ttl(record) == 0

    def test_default_ttl(self):
        policy = RetentionPolicy(default_ttl_days=90)
        record = MemoryRecord(
            id="test", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert policy.effective_ttl(record) == 90

    def test_type_override(self):
        policy = RetentionPolicy(
            default_ttl_days=90,
            type_ttl={"volatile": 30},
        )
        record = MemoryRecord(
            id="test", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            memory_type="volatile",
        )
        assert policy.effective_ttl(record) == 30

    def test_tag_override_shortest_wins(self):
        policy = RetentionPolicy(
            default_ttl_days=90,
            tag_ttl={"temp": 7, "important": 365},
        )
        record = MemoryRecord(
            id="test", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=["temp", "important"],
            source="", priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert policy.effective_ttl(record) == 7  # Shortest wins

    def test_namespace_override(self):
        policy = RetentionPolicy(
            default_ttl_days=90,
            namespace_ttl={"scratch": 14},
        )
        record = MemoryRecord(
            id="test", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc), namespace="scratch",
        )
        assert policy.effective_ttl(record) == 14

    def test_find_expired_memories(self):
        policy = RetentionPolicy(default_ttl_days=30)
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        recent = now - timedelta(days=10)

        records = [
            MemoryRecord(
                id="old", content="old", l3_abstract="", l2_facts=[],
                l1_compressed=b"", embedding=[], tags=[], source="",
                priority=0.5, created_at=old, updated_at=old,
            ),
            MemoryRecord(
                id="recent", content="recent", l3_abstract="", l2_facts=[],
                l1_compressed=b"", embedding=[], tags=[], source="",
                priority=0.5, created_at=recent, updated_at=recent,
            ),
        ]
        expired = find_expired_memories(records, policy, now=now)
        assert len(expired) == 1
        assert expired[0][0].id == "old"

    def test_parse_retention_config(self):
        config = "default:90,type:volatile:30,tag:temp:7,ns:scratch:14"
        policy = parse_retention_config(config)
        assert policy.default_ttl_days == 90
        assert policy.type_ttl == {"volatile": 30}
        assert policy.tag_ttl == {"temp": 7}
        assert policy.namespace_ttl == {"scratch": 14}

    def test_parse_empty_config(self):
        policy = parse_retention_config("")
        assert policy.default_ttl_days == 0

    def test_store_purge_expired(self, tmp_db):
        # With default empty retention policy, nothing should expire
        results = tmp_db.purge_expired(dry_run=True)
        assert len(results) == 0


# --- Adaptive Re-ranking Tests ---


class TestAdaptiveReranking:
    def test_feedback_increases_priority(self):
        scorer = PriorityScorer(decay_half_life_days=30.0)
        record = MemoryRecord(
            id="test1", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        base = scorer.adjusted_priority(record)

        scorer.record_feedback("test1", useful=True)
        scorer.record_feedback("test1", useful=True)
        boosted = scorer.adjusted_priority(record)
        assert boosted > base

    def test_negative_feedback_decreases_priority(self):
        scorer = PriorityScorer(decay_half_life_days=30.0)
        record = MemoryRecord(
            id="test1", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        base = scorer.adjusted_priority(record)

        scorer.record_feedback("test1", useful=False)
        scorer.record_feedback("test1", useful=False)
        penalized = scorer.adjusted_priority(record)
        assert penalized < base

    def test_recency_boost(self):
        scorer = PriorityScorer(decay_half_life_days=30.0)
        now = datetime.now(timezone.utc)

        # Recently accessed
        recent = MemoryRecord(
            id="recent", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=now, updated_at=now,
            last_accessed=now,
        )

        # Not recently accessed
        old = MemoryRecord(
            id="old", content="test", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=now, updated_at=now,
            last_accessed=now - timedelta(days=30),
        )

        assert scorer.adjusted_priority(recent) > scorer.adjusted_priority(old)

    def test_feedback_state_persistence(self):
        scorer = PriorityScorer()
        scorer.record_feedback("id1", useful=True)
        scorer.record_feedback("id2", useful=False)
        state = scorer.save_feedback()

        scorer2 = PriorityScorer()
        scorer2.load_feedback(state)
        record1 = MemoryRecord(
            id="id1", content="", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        record2 = MemoryRecord(
            id="id2", content="", l3_abstract="", l2_facts=[],
            l1_compressed=b"", embedding=[], tags=[], source="",
            priority=0.5, created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # id1 should rank higher than id2
        assert scorer2.adjusted_priority(record1) > scorer2.adjusted_priority(record2)

    def test_store_feedback(self, tmp_db):
        """Test feedback through the store interface."""
        record = tmp_db.store("Test memory for feedback")
        tmp_db.record_feedback(record.id, useful=True)
        # No exception means success


# --- Integration Tests ---


class TestIntegration:
    def test_trust_report(self, tmp_db):
        record = tmp_db.store("Test memory from CLI source", source="cli")
        report = tmp_db.get_trust_report(record.id)
        assert "trust_score" in report
        assert "is_trusted" in report

    def test_pii_warn_mode(self, tmp_db):
        # In warn mode, PII is stored but warnings logged
        record = tmp_db.store("My email is test@example.com")
        assert "test@example.com" in record.content  # Not redacted in warn mode

    def test_all_features_store_recall_cycle(self, tmp_db):
        """End-to-end: store with auto-tag, recall with staleness, give feedback."""
        record = tmp_db.store(
            "I decided to use FastAPI for the REST API layer",
            source="design-doc",
        )
        assert record.memory_type == "decision"
        assert any(t in record.tags for t in ["python", "api"])

        results = tmp_db.recall("API framework decision")
        assert len(results) >= 1

        # Give feedback
        tmp_db.record_feedback(results[0].record.id, useful=True)

        # Trust report
        report = tmp_db.get_trust_report(record.id)
        assert report["is_trusted"]

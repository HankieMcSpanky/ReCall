"""Tests for session summary generation."""
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.session import generate_session_summary, _categorize_sentence
from neuropack.core.store import MemoryStore


def test_empty_session():
    result = generate_session_summary([])
    assert result["summary"] == "Empty session"
    assert result["investigated"] == []
    assert result["learned"] == []
    assert result["completed"] == []
    assert result["next_steps"] == []


def test_categorize_sentence():
    assert _categorize_sentence("I found a bug in the code") == "investigated"
    assert _categorize_sentence("I learned about Python") == "learned"
    assert _categorize_sentence("I fixed the authentication issue") == "completed"
    assert _categorize_sentence("Next we should add tests") == "next_steps"
    assert _categorize_sentence("Some random sentence") == "learned"  # default


def test_session_summary_with_memories(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        r1 = store.store(content="I explored the database module and found several issues.", tags=["dev"])
        r2 = store.store(content="I fixed the authentication bug in the login flow.", tags=["dev"])
        r3 = store.store(content="Next we should implement the API endpoints.", tags=["dev"])

        summary = store.session_summary([r1.id, r2.id, r3.id])
        assert "summary" in summary
        assert isinstance(summary["investigated"], list)
        assert isinstance(summary["completed"], list)
        assert isinstance(summary["next_steps"], list)
    finally:
        store.close()


def test_store_session_summary(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        r1 = store.store(content="Completed the migration to the new database schema.", tags=["dev"])
        record = store.store_session_summary([r1.id])
        assert "session-summary" in record.tags
        assert record.priority == 0.7
    finally:
        store.close()


def test_session_summary_deduplicates(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        # Store two memories with overlapping L3 abstracts
        r1 = store.store(content="I found a critical bug in the parser.", tags=["dev"])
        r2 = store.store(content="I also found a critical bug in the lexer component.", tags=["dev"])
        summary = store.session_summary([r1.id, r2.id])
        # Categories should not have exact duplicate entries
        for cat in ("investigated", "learned", "completed", "next_steps"):
            assert len(summary[cat]) == len(set(summary[cat]))
    finally:
        store.close()

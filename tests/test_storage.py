"""Tests for SQLite storage layer."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from neuropack.exceptions import MemoryNotFoundError
from neuropack.storage.database import Database
from neuropack.storage.repository import MemoryRepository
from neuropack.types import MemoryRecord


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    d.initialize_schema()
    yield d
    d.close()


@pytest.fixture
def repo(db):
    return MemoryRepository(db)


def _make_record(id_: str = "test-001", content: str = "Hello world") -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        content=content,
        l3_abstract="Greeting",
        l2_facts=["Says hello"],
        l1_compressed=b"\x00\x01\x02",
        embedding=[0.1] * 256,
        tags=["test"],
        source="unit-test",
        priority=0.5,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_insert_and_get(repo):
    record = _make_record()
    repo.insert(record)
    fetched = repo.get_by_id("test-001")
    assert fetched is not None
    assert fetched.id == "test-001"
    assert fetched.content == "Hello world"
    assert fetched.l3_abstract == "Greeting"
    assert fetched.l2_facts == ["Says hello"]
    assert fetched.tags == ["test"]


def test_get_nonexistent(repo):
    assert repo.get_by_id("nonexistent") is None


def test_delete(repo):
    repo.insert(_make_record())
    assert repo.delete("test-001") is True
    assert repo.get_by_id("test-001") is None


def test_delete_nonexistent(repo):
    assert repo.delete("nonexistent") is False


def test_update(repo):
    repo.insert(_make_record())
    updated = repo.update("test-001", priority=0.9, tags=["updated"])
    assert updated.priority == 0.9
    assert updated.tags == ["updated"]


def test_update_nonexistent(repo):
    with pytest.raises(MemoryNotFoundError):
        repo.update("nonexistent", priority=0.9)


def test_list_all(repo):
    repo.insert(_make_record("id-1", "First"))
    repo.insert(_make_record("id-2", "Second"))
    repo.insert(_make_record("id-3", "Third"))
    records = repo.list_all(limit=10)
    assert len(records) == 3


def test_list_with_tag_filter(repo):
    repo.insert(_make_record("id-1", "First"))
    r2 = MemoryRecord(
        id="id-2", content="Second", l3_abstract="S", l2_facts=[], l1_compressed=b"",
        embedding=[0.0] * 256, tags=["special"], source="", priority=0.5,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    repo.insert(r2)
    records = repo.list_all(limit=10, tag="special")
    assert len(records) == 1
    assert records[0].id == "id-2"


def test_count(repo):
    assert repo.count() == 0
    repo.insert(_make_record("id-1"))
    repo.insert(_make_record("id-2"))
    assert repo.count() == 2


def test_stats(repo):
    repo.insert(_make_record())
    s = repo.stats()
    assert s.total_memories == 1
    assert s.total_size_bytes > 0


def test_fts_search(repo):
    repo.insert(_make_record("id-1", "Python programming language"))
    repo.insert(_make_record("id-2", "Tokyo weather forecast"))
    results = repo.fts_search("Python")
    assert len(results) >= 1
    assert results[0][0] == "id-1"


def test_get_all_embeddings(repo):
    repo.insert(_make_record("id-1"))
    repo.insert(_make_record("id-2"))
    embeddings = repo.get_all_embeddings()
    assert len(embeddings) == 2
    assert embeddings[0][1].shape == (256,)


def test_touch(repo):
    repo.insert(_make_record())
    repo.touch("test-001")
    record = repo.get_by_id("test-001")
    assert record.access_count == 1
    assert record.last_accessed is not None


def test_metadata(repo):
    repo.save_metadata("test_key", "test_value")
    assert repo.load_metadata("test_key") == "test_value"
    assert repo.load_metadata("nonexistent") is None


def test_wal_mode(db):
    conn = db.connect()
    result = conn.execute("PRAGMA journal_mode").fetchone()
    assert dict(result)["journal_mode"] == "wal"

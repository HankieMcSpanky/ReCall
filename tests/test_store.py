"""Integration tests for MemoryStore."""
import pytest

from neuropack.exceptions import ContentTooLargeError, MemoryNotFoundError


def test_store_and_get(store):
    record = store.store("User prefers dark mode", tags=["prefs"])
    assert record.id
    assert record.l3_abstract
    assert record.l2_facts
    assert len(record.l1_compressed) > 0

    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.content == "User prefers dark mode"


def test_store_with_caller_provided_l3_l2(store):
    record = store.store(
        "Some content here",
        l3_override="Custom abstract",
        l2_override=["Fact one", "Fact two"],
    )
    assert record.l3_abstract == "Custom abstract"
    assert record.l2_facts == ["Fact one", "Fact two"]


def test_store_and_recall(populated_store):
    results = populated_store.recall("programming language")
    assert len(results) > 0
    # Python memory should rank high
    ids_with_python = [r.record.id for r in results if "Python" in r.record.content]
    assert len(ids_with_python) > 0


def test_forget(store):
    record = store.store("Temporary memory")
    assert store.forget(record.id) is True
    assert store.get(record.id) is None


def test_forget_nonexistent(store):
    assert store.forget("nonexistent-id") is False


def test_update_content(store):
    record = store.store("Original content", tags=["v1"])
    updated = store.update(record.id, content="Updated content")
    assert updated.content == "Updated content"
    # L3 should reflect new content
    assert updated.l3_abstract != ""


def test_update_tags_only(store):
    record = store.store("Some content", tags=["old"])
    updated = store.update(record.id, tags=["new", "tags"])
    assert updated.tags == ["new", "tags"]
    assert updated.content == "Some content"


def test_update_nonexistent(store):
    with pytest.raises(MemoryNotFoundError):
        store.update("nonexistent", tags=["x"])


def test_list(populated_store):
    records = populated_store.list(limit=3)
    assert len(records) == 3


def test_list_with_tag(populated_store):
    records = populated_store.list(tag="tech")
    assert all(any("tech" in t for t in r.tags) for r in records)


def test_stats(populated_store):
    s = populated_store.stats()
    assert s.total_memories == 5
    assert s.total_size_bytes > 0
    assert s.avg_compression_ratio > 0
    assert s.oldest is not None
    assert s.newest is not None


def test_content_too_large(store):
    store.config = store.config.model_copy(update={"max_content_size": 10})
    with pytest.raises(ContentTooLargeError):
        store.store("x" * 100)


def test_dedup_merges(store):
    r1 = store.store("User lives in Helsinki Finland", tags=["location"])
    r2 = store.store("User lives in Helsinki Finland", tags=["geo"])
    # Should merge (same text = cosine ~1.0)
    assert store.stats().total_memories == 1
    merged = store.get(r1.id)
    assert merged is not None
    assert "location" in merged.tags
    assert "geo" in merged.tags


def test_decompress(store):
    record = store.store("Original text for decompression test")
    result = store.decompress(record.l1_compressed)
    assert result == "Original text for decompression test"

from __future__ import annotations

from pathlib import Path

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
def config(tmp_db: str) -> NeuropackConfig:
    return NeuropackConfig(db_path=tmp_db, auth_token="test-token-123")


@pytest.fixture
def store(config: NeuropackConfig) -> MemoryStore:
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def populated_store(store: MemoryStore) -> MemoryStore:
    """Store with 5 pre-loaded memories for search tests."""
    store.store(
        "The quick brown fox jumps over the lazy dog",
        tags=["animals"],
        source="test",
    )
    store.store(
        "Python is a programming language created by Guido van Rossum. It supports multiple paradigms.",
        tags=["tech"],
        source="test",
    )
    store.store(
        "SQLite uses B-trees for indexing and supports full-text search via FTS5 extension.",
        tags=["tech", "database"],
        source="test",
    )
    store.store(
        "The weather in Tokyo is warm and humid in summer. Cherry blossoms bloom in spring.",
        tags=["travel"],
        source="test",
    )
    store.store(
        "Neural networks use backpropagation for training. Deep learning requires large datasets.",
        tags=["tech", "ml"],
        source="test",
    )
    return store

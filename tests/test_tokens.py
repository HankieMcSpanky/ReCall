"""Tests for token estimation and economics."""
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.core.tokens import estimate_tokens, estimate_tokens_for_list


def test_estimate_tokens_basic():
    # "hello world" = 11 chars -> 11//4 = 2
    assert estimate_tokens("hello world") == 2


def test_estimate_tokens_minimum():
    assert estimate_tokens("") == 1
    assert estimate_tokens("hi") == 1


def test_estimate_tokens_for_list():
    result = estimate_tokens_for_list(["fact one", "fact two"])
    assert result > 0


def test_estimate_tokens_for_empty_list():
    assert estimate_tokens_for_list([]) == 1


def test_store_populates_token_fields(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        text = "This is a moderately long piece of content for testing token estimation."
        record = store.store(content=text, tags=["test"])
        assert record.content_tokens > 0
        assert record.compressed_tokens > 0
    finally:
        store.close()


def test_token_stats(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(content="Memory one with some content", tags=["a"])
        store.store(content="Memory two with other content", tags=["b"])
        ts = store.token_stats()
        assert ts["total_content_tokens"] > 0
        assert ts["total_compressed_tokens"] > 0
        assert isinstance(ts["tokens_saved"], int)
        assert isinstance(ts["token_savings_ratio"], float)
    finally:
        store.close()


def test_token_savings_with_long_content(tmp_path):
    """Long content should show real token savings since L3+L2 < full content."""
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        long_text = (
            "Python is a high-level programming language known for its simplicity. "
            "It was created by Guido van Rossum and first released in 1991. "
            "Python supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming. It has a large standard "
            "library and an active community. Python is widely used in web development, "
            "data science, artificial intelligence, and automation."
        )
        record = store.store(content=long_text, tags=["tech"])
        # For long content, compressed tokens should be less than content tokens
        assert record.content_tokens > record.compressed_tokens
        ts = store.token_stats()
        assert ts["tokens_saved"] > 0
        assert ts["token_savings_ratio"] > 0
    finally:
        store.close()


def test_stats_includes_token_fields(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    store = MemoryStore(config)
    store.initialize()
    try:
        store.store(content="Some content for stats test", tags=["test"])
        s = store.stats()
        assert s.total_content_tokens > 0
        assert s.total_compressed_tokens > 0
        assert isinstance(s.token_savings_ratio, float)
    finally:
        store.close()

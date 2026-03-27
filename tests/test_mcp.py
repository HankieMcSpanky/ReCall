"""Tests for MCP server tools."""
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.mcp_server import server as mcp_module


@pytest.fixture(autouse=True)
def setup_store(tmp_path):
    """Set up the module-level store for MCP tools."""
    config = NeuropackConfig(db_path=str(tmp_path / "mcp_test.db"))
    store = MemoryStore(config)
    store.initialize()
    mcp_module._store = store
    yield
    store.close()
    mcp_module._store = None


def test_remember():
    result = mcp_module.remember(
        content="User prefers dark mode",
        tags=["prefs"],
        source="test",
    )
    assert result["status"] == "stored"
    assert result["id"]
    assert result["l3_abstract"]


def test_remember_with_caller_provided():
    result = mcp_module.remember(
        content="Some content",
        l3="Custom abstract",
        l2=["Fact 1"],
    )
    assert result["l3_abstract"] == "Custom abstract"


def test_recall():
    mcp_module.remember(content="Python is great for data science", tags=["tech"])
    result = mcp_module.recall(query="Python data science")
    assert result["count"] >= 1
    assert result["results"][0]["l3_abstract"]


def test_forget():
    result = mcp_module.remember(content="Temporary memory")
    mid = result["id"]
    forget_result = mcp_module.forget(memory_id=mid)
    assert forget_result["deleted"] is True


def test_list_memories():
    mcp_module.remember(content="Memory one")
    mcp_module.remember(content="Memory two, completely different")
    result = mcp_module.list_memories(limit=10)
    assert result["count"] >= 2


def test_memory_stats():
    mcp_module.remember(content="Stats test memory")
    result = mcp_module.memory_stats()
    assert result["total_memories"] >= 1

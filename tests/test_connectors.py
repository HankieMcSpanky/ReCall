"""Tests for data source connectors (Phase 6)."""
from __future__ import annotations

import json

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.io.connectors.structured import parse_csv, parse_json_array


@pytest.fixture
def store(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


# --- CSV connector ---

def test_parse_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "content,category,source\n"
        "Python is great for ML,tech,manual\n"
        "React is popular for frontend,web,manual\n"
        "x,,\n",  # too short, should be skipped
        encoding="utf-8",
    )

    memories = parse_csv(
        str(csv_file),
        content_column="content",
        tag_columns=["category"],
        source_column="source",
    )
    assert len(memories) == 2
    assert memories[0]["content"] == "Python is great for ML"
    assert "tech" in memories[0]["tags"]
    assert "csv" in memories[0]["tags"]
    assert memories[0]["source"] == "csv:manual"


def test_parse_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_csv(str(tmp_path / "nope.csv"))


# --- JSON array connector ---

def test_parse_json_array(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([
        {"content": "Memory about databases", "tags": ["tech"], "source": "notes"},
        {"content": "Memory about cooking", "tags": ["food"]},
        {"content": "x"},  # too short
    ]), encoding="utf-8")

    memories = parse_json_array(str(json_file))
    assert len(memories) == 2
    assert memories[0]["source"] == "notes"
    assert "json" in memories[0]["tags"]
    assert "tech" in memories[0]["tags"]


def test_parse_json_array_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_json_array(str(tmp_path / "nope.json"))


# --- Integration with store.import_memories ---

def test_import_csv_via_store(store, tmp_path):
    csv_file = tmp_path / "import.csv"
    csv_file.write_text(
        "content,tag\n"
        "Important discovery about quantum computing,science\n"
        "Neural networks learn from data,ai\n",
        encoding="utf-8",
    )

    count = store.import_memories(format="csv", path=str(csv_file))
    assert count == 2


def test_import_json_array_via_store(store, tmp_path):
    json_file = tmp_path / "import.json"
    json_file.write_text(json.dumps([
        {"content": "GraphQL vs REST comparison"},
        {"content": "Docker containerization benefits"},
    ]), encoding="utf-8")

    count = store.import_memories(format="json-array", path=str(json_file))
    assert count == 2


# --- PDF connector (import check, actual parsing requires pymupdf) ---

def test_pdf_import_error_no_file(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.import_memories(format="pdf", path=str(tmp_path / "nope.pdf"))


# --- Web connector (skip if trafilatura not installed) ---

def test_web_connector_import_error():
    """Web format requires trafilatura."""
    try:
        from neuropack.io.connectors.web import parse_url
        # If trafilatura is installed, this would try to fetch
        # We just verify the function exists
        assert callable(parse_url)
    except ImportError:
        pytest.skip("trafilatura not installed")

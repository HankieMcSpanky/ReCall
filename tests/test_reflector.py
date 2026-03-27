"""Tests for reflection and synthesis (Phase 4)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.reflector import MemoryReflector, _parse_json
from neuropack.core.store import MemoryStore


def test_parse_json_direct():
    assert _parse_json('{"key": "value"}') == {"key": "value"}


def test_parse_json_with_text():
    text = 'Here is the result: {"insight": "test", "patterns": []}'
    result = _parse_json(text)
    assert result["insight"] == "test"


def test_parse_json_failure():
    result = _parse_json("not json at all")
    assert "error" in result


def test_synthesize_no_memories():
    mock_llm = MagicMock()
    reflector = MemoryReflector(mock_llm)
    result = reflector.synthesize("test query", [])
    assert result["confidence"] == 0.0
    assert "No memories" in result["insight"]


def test_synthesize_with_memories():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"insight": "Python is widely used", "patterns": ["tech"], "confidence": 0.8}'

    reflector = MemoryReflector(mock_llm)
    memories = [
        {"id": "m1", "content": "Python is great", "l3_abstract": "Python praise", "tags": ["tech"]},
        {"id": "m2", "content": "Python for ML", "l3_abstract": "ML with Python", "tags": ["tech"]},
    ]
    result = reflector.synthesize("Python", memories)
    assert result["insight"] == "Python is widely used"
    assert result["source_ids"] == ["m1", "m2"]


def test_synthesize_llm_error():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("LLM down")

    reflector = MemoryReflector(mock_llm)
    memories = [{"id": "m1", "content": "test", "l3_abstract": "", "tags": []}]
    result = reflector.synthesize("query", memories)
    assert result["confidence"] == 0.0
    assert "m1" in result["source_ids"]


def test_reflect_with_related():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"reflection": "Key finding", "contradictions": [], "evolution": "stable"}'

    reflector = MemoryReflector(mock_llm)
    memory = {"id": "m1", "content": "React is fast", "l3_abstract": "React performance"}
    related = [{"id": "m2", "l3_abstract": "React 18 improvements"}]
    result = reflector.reflect(memory, related)
    assert result["reflection"] == "Key finding"


def test_recall_and_synthesize_no_llm(tmp_path):
    """Without LLM configured, synthesis returns a no-LLM message."""
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"))
    s = MemoryStore(config)
    s.initialize()
    s.store("Python is versatile.", tags=["tech"])

    result = s.recall_and_synthesize("Python", limit=5, synthesize=True)
    assert "results" in result
    assert result["synthesis"]["confidence"] == 0.0
    s.close()

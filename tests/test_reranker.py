"""Tests for cross-encoder / LLM reranking (Phase 3)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.search.reranker import LLMReranker, Reranker
from neuropack.types import MemoryRecord, RecallResult


def _make_result(content: str, score: float) -> RecallResult:
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id="test-" + content[:8],
        content=content,
        l3_abstract=content[:50],
        l2_facts=[content],
        l1_compressed=b"",
        embedding=[0.0] * 10,
        tags=["test"],
        source="test",
        priority=0.5,
        created_at=now,
        updated_at=now,
    )
    return RecallResult(record=record, score=score)


def test_llm_reranker_blends_scores():
    """LLM reranker blends RRF score with LLM relevance score."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "8"  # 8/10 relevance

    reranker = LLMReranker(mock_llm, weight=0.3)
    results = [
        _make_result("Python is great for scripting", 0.5),
        _make_result("Java is verbose but reliable", 0.4),
    ]

    reranked = reranker.rerank("scripting language", results, top_k=2)
    assert len(reranked) == 2
    # All results should have blended scores
    for r in reranked:
        assert r.score > 0


def test_llm_reranker_respects_top_k():
    """Reranker returns at most top_k results."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "5"

    reranker = LLMReranker(mock_llm, weight=0.3)
    results = [_make_result(f"item {i}", 0.5 - i * 0.1) for i in range(5)]

    reranked = reranker.rerank("query", results, top_k=2)
    assert len(reranked) <= 2


def test_llm_reranker_handles_llm_error():
    """If LLM fails, reranker uses fallback score of 0.5."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("LLM down")

    reranker = LLMReranker(mock_llm, weight=0.3)
    results = [_make_result("test content", 0.5)]

    reranked = reranker.rerank("query", results, top_k=1)
    assert len(reranked) == 1
    # Score should be blended with fallback 0.5
    expected = (1 - 0.3) * 0.5 + 0.3 * 0.5
    assert abs(reranked[0].score - expected) < 0.01


def test_reranker_off_by_default(tmp_path):
    """When reranker='off', no reranker is wired."""
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), reranker="off")
    s = MemoryStore(config)
    s.initialize()
    assert s._retriever._reranker is None
    s.close()


def test_reranker_empty_results():
    """Reranking empty list returns empty list."""
    mock_llm = MagicMock()
    reranker = LLMReranker(mock_llm, weight=0.3)
    assert reranker.rerank("query", [], top_k=10) == []

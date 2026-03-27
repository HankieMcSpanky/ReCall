"""Post-retrieval reranking: LLM-based or cross-encoder scoring."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from neuropack.types import RecallResult


class Reranker(ABC):
    """Base class for rerankers that rescore query-passage pairs."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RecallResult], top_k: int = 10
    ) -> list[RecallResult]:
        ...


class LLMReranker(Reranker):
    """Use an existing LLM provider to score relevance of each result."""

    def __init__(self, llm_provider, weight: float = 0.3):
        self._llm = llm_provider
        self._weight = weight

    def rerank(
        self, query: str, results: list[RecallResult], top_k: int = 10
    ) -> list[RecallResult]:
        if not results:
            return results

        # Score each candidate with the LLM
        scored: list[tuple[RecallResult, float]] = []
        for r in results[:top_k * 2]:  # only rerank top candidates
            rerank_score = self._score_pair(query, r.record.l3_abstract or r.record.content[:300])
            scored.append((r, rerank_score))

        # Blend: final = (1-w)*rrf + w*reranker
        reranked = []
        for r, rscore in scored:
            blended = (1 - self._weight) * r.score + self._weight * rscore
            reranked.append(RecallResult(
                record=r.record,
                score=blended,
                fts_rank=r.fts_rank,
                vec_score=r.vec_score,
                graph_score=r.graph_score,
                staleness_warning=r.staleness_warning,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def _score_pair(self, query: str, passage: str) -> float:
        """Ask LLM to rate relevance 0-10, return normalized score."""
        prompt = (
            f"Rate the relevance of this passage to the query on a scale of 0-10.\n"
            f"Query: {query}\n"
            f"Passage: {passage}\n"
            f"Reply with ONLY a number 0-10."
        )
        try:
            response = self._llm.complete(prompt)
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if match:
                return min(float(match.group(1)) / 10.0, 1.0)
        except Exception:
            pass
        return 0.5  # fallback neutral score


class CrossEncoderReranker(Reranker):
    """Use a sentence-transformers CrossEncoder for relevance scoring.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", weight: float = 0.3):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers: "
                "pip install sentence-transformers"
            )
        self._weight = weight

    def rerank(
        self, query: str, results: list[RecallResult], top_k: int = 10
    ) -> list[RecallResult]:
        if not results:
            return results

        candidates = results[:top_k * 2]
        pairs = [
            (query, r.record.l3_abstract or r.record.content[:512])
            for r in candidates
        ]

        scores = self._model.predict(pairs)
        # Normalize to 0-1
        min_s = min(scores) if len(scores) > 0 else 0
        max_s = max(scores) if len(scores) > 0 else 1
        span = max_s - min_s if max_s != min_s else 1.0

        reranked = []
        for r, raw_score in zip(candidates, scores):
            norm_score = (raw_score - min_s) / span
            blended = (1 - self._weight) * r.score + self._weight * norm_score
            reranked.append(RecallResult(
                record=r.record,
                score=blended,
                fts_rank=r.fts_rank,
                vec_score=r.vec_score,
                graph_score=r.graph_score,
                staleness_warning=r.staleness_warning,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

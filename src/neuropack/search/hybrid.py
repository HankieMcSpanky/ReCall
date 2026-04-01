from __future__ import annotations

from neuropack.config import NeuropackConfig
from neuropack.core.decay import MemoryDecay
from neuropack.core.events import EventExtractor
from neuropack.core.priority import PriorityScorer
from neuropack.embeddings.base import Embedder
from neuropack.exceptions import FTSQueryError
from neuropack.search.decomposer import QueryDecomposer
from neuropack.search.graph_retriever import GraphRetriever
from neuropack.search.reranker import Reranker
from neuropack.search.temporal import TemporalRetriever
from neuropack.search.vector_index import BruteForceIndex
from neuropack.storage.repository import MemoryRepository
from neuropack.types import RecallResult


class HybridRetriever:
    """Hybrid retrieval combining vector similarity, FTS5, knowledge graph, and temporal via RRF."""

    def __init__(
        self,
        repository: MemoryRepository,
        vector_index: BruteForceIndex,
        embedder: Embedder,
        priority_scorer: PriorityScorer,
        config: NeuropackConfig,
        graph_retriever: GraphRetriever | None = None,
        reranker: Reranker | None = None,
        temporal_retriever: TemporalRetriever | None = None,
    ):
        self._repo = repository
        self._index = vector_index
        self._embedder = embedder
        self._priority = priority_scorer
        self._config = config
        self._graph = graph_retriever
        self._reranker = reranker
        self._temporal = temporal_retriever
        self._decay = MemoryDecay(decay_rate=config.decay_rate)
        self._decomposer = QueryDecomposer() if config.query_decomposition else None
        self._events: EventExtractor | None = None  # wired in MemoryStore.initialize()

    def recall(
        self,
        query: str,
        limit: int = 20,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> list[RecallResult]:
        # Query decomposition: if enabled, run sub-queries and merge via RRF
        if self._decomposer is not None:
            sub_queries = self._decomposer.decompose(query)
            if len(sub_queries) > 1:
                return self._recall_decomposed(
                    sub_queries, limit=limit, tags=tags,
                    min_score=min_score, namespaces=namespaces,
                )

        return self._recall_single(
            query, limit=limit, tags=tags,
            min_score=min_score, namespaces=namespaces,
        )

    def _recall_decomposed(
        self,
        sub_queries: list[str],
        limit: int = 20,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> list[RecallResult]:
        """Run retrieval for each sub-query and merge results using RRF."""
        per_query_k = max(limit, 10)
        all_results_by_query: list[list[RecallResult]] = []

        for sq in sub_queries:
            results = self._recall_single(
                sq, limit=per_query_k, tags=tags,
                min_score=0.0, namespaces=namespaces,
            )
            all_results_by_query.append(results)

        # Build rank maps per sub-query (1-indexed) and fuse via RRF
        rrf_k = self._config.rrf_k
        scores: dict[str, float] = {}
        result_map: dict[str, RecallResult] = {}

        for query_results in all_results_by_query:
            for rank, r in enumerate(query_results, 1):
                mid = r.record.id
                scores[mid] = scores.get(mid, 0.0) + (1.0 / (rrf_k + rank))
                # Keep the result with the highest individual score
                if mid not in result_map or r.score > result_map[mid].score:
                    result_map[mid] = r

        # Sort by fused score and apply min_score filter
        fused: list[RecallResult] = []
        for mid, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if fused_score < min_score:
                continue
            original = result_map[mid]
            fused.append(RecallResult(
                record=original.record,
                score=fused_score,
                fts_rank=original.fts_rank,
                vec_score=original.vec_score,
                graph_score=original.graph_score,
                temporal_score=original.temporal_score,
                staleness_warning=original.staleness_warning,
            ))

        return fused[:limit]

    def _recall_single(
        self,
        query: str,
        limit: int = 20,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> list[RecallResult]:
        """Single-query retrieval using hybrid vector + FTS + graph + temporal."""
        candidate_k = limit * 5

        # Vector search
        query_vec = self._embedder.embed(query)
        vec_results = self._index.search(query_vec, k=candidate_k)

        # FTS5 search (fall back to vector-only on query errors)
        try:
            fts_results = self._repo.fts_search(query, limit=candidate_k)
        except FTSQueryError:
            fts_results = []

        # Graph-based retrieval (optional 3rd signal)
        graph_results: list[tuple[str, float]] = []
        if self._graph:
            try:
                graph_results = self._graph.retrieve(query, limit=candidate_k)
            except Exception:
                graph_results = []

        # Temporal retrieval (optional 4th signal — only active when query has date refs)
        temporal_results: list[tuple[str, float]] = []
        has_temporal = False
        if self._temporal:
            try:
                temporal_results = self._temporal.retrieve(
                    query, limit=candidate_k,
                    namespace=namespaces[0] if namespaces and len(namespaces) == 1 else None,
                )
                has_temporal = len(temporal_results) > 0
            except Exception:
                temporal_results = []

        # Event calendar retrieval (optional 5th signal)
        event_results: list[tuple[str, float]] = []
        has_events = False
        if self._events:
            try:
                ns_for_events = namespaces[0] if namespaces and len(namespaces) == 1 else None
                event_results = self._events.search_events(
                    query, limit=candidate_k, namespace=ns_for_events,
                )
                has_events = len(event_results) > 0
            except Exception:
                event_results = []

        # Build rank maps (1-indexed)
        vec_ranks: dict[str, tuple[int, float]] = {}
        for rank, (mid, score) in enumerate(vec_results, 1):
            vec_ranks[mid] = (rank, score)

        fts_ranks: dict[str, tuple[int, float]] = {}
        for rank, (mid, bm25) in enumerate(fts_results, 1):
            fts_ranks[mid] = (rank, bm25)

        graph_ranks: dict[str, tuple[int, float]] = {}
        for rank, (mid, gscore) in enumerate(graph_results, 1):
            graph_ranks[mid] = (rank, gscore)

        temporal_ranks: dict[str, tuple[int, float]] = {}
        for rank, (mid, tscore) in enumerate(temporal_results, 1):
            temporal_ranks[mid] = (rank, tscore)

        event_ranks: dict[str, tuple[int, float]] = {}
        for rank, (mid, escore) in enumerate(event_results, 1):
            event_ranks[mid] = (rank, escore)

        # 5-way Reciprocal Rank Fusion
        all_ids = (
            set(vec_ranks.keys())
            | set(fts_ranks.keys())
            | set(graph_ranks.keys())
            | set(temporal_ranks.keys())
            | set(event_ranks.keys())
        )
        rrf_k = self._config.rrf_k

        # Determine active weights
        w_graph = self._config.retrieval_weight_graph if self._graph else 0.0
        w_temporal = self._config.retrieval_weight_temporal if has_temporal else 0.0
        w_events = self._config.retrieval_weight_events if has_events else 0.0

        # Rebalance vec/fts weights to account for graph + temporal + events weights
        remaining = 1.0 - w_graph - w_temporal - w_events
        remaining = max(remaining, 0.0)  # safety clamp
        total_vf = self._config.retrieval_weight_vec + self._config.retrieval_weight_fts
        w_vec = self._config.retrieval_weight_vec / total_vf * remaining if total_vf > 0 else remaining / 2
        w_fts = self._config.retrieval_weight_fts / total_vf * remaining if total_vf > 0 else remaining / 2

        scored: list[tuple[str, float, float | None, float | None, float | None, float | None]] = []
        for mid in all_ids:
            vec_component = 0.0
            fts_component = 0.0
            graph_component = 0.0
            temporal_component = 0.0
            event_component = 0.0
            vec_score = None
            fts_rank_val = None
            graph_score_val = None
            temporal_score_val = None

            if mid in vec_ranks:
                rank, sim = vec_ranks[mid]
                vec_component = w_vec * (1.0 / (rrf_k + rank))
                vec_score = sim

            if mid in fts_ranks:
                rank, bm25 = fts_ranks[mid]
                fts_component = w_fts * (1.0 / (rrf_k + rank))
                fts_rank_val = bm25

            if mid in graph_ranks:
                rank, gscore = graph_ranks[mid]
                graph_component = w_graph * (1.0 / (rrf_k + rank))
                graph_score_val = gscore

            if mid in temporal_ranks:
                rank, tscore = temporal_ranks[mid]
                temporal_component = w_temporal * (1.0 / (rrf_k + rank))
                temporal_score_val = tscore

            if mid in event_ranks:
                rank, _escore = event_ranks[mid]
                event_component = w_events * (1.0 / (rrf_k + rank))

            rrf_score = vec_component + fts_component + graph_component + temporal_component + event_component
            scored.append((mid, rrf_score, vec_score, fts_rank_val, graph_score_val, temporal_score_val))

        # Fetch records, apply priority boost, filter
        results: list[RecallResult] = []
        for mid, rrf_score, vec_score, fts_rank_val, graph_score_val, temporal_score_val in scored:
            record = self._repo.get_by_id(mid)
            if record is None:
                continue

            # Namespace filter
            if namespaces:
                if record.namespace not in namespaces:
                    continue

            # Tag filter
            if tags:
                if not any(t in record.tags for t in tags):
                    continue

            # Priority boost
            priority_adj = self._priority.adjusted_priority(record)
            final_score = rrf_score * (0.5 + 0.5 * priority_adj)

            # Supersession demotion: reduce score of memories that have been superseded
            if record.superseded_by is not None:
                final_score *= 0.5

            if final_score < min_score:
                continue

            results.append(RecallResult(
                record=record,
                score=final_score,
                fts_rank=fts_rank_val,
                vec_score=vec_score,
                graph_score=graph_score_val,
                temporal_score=temporal_score_val,
            ))

        # Sort by final score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Cross-encoder reranking: take top N candidates, rescore, return top K
        if self._reranker and results and self._config.rerank_enabled:
            rerank_candidates = results[:self._config.rerank_top_n]
            results = self._reranker.rerank(query, rerank_candidates, top_k=limit)
        else:
            results = results[:limit]

        # Apply Ebbinghaus decay boost to final scores
        if self._config.decay_enabled and results:
            decayed: list[RecallResult] = []
            for r in results:
                boosted_score = self._decay.apply_decay_boost(
                    r.score,
                    r.record,
                    decay_weight=self._config.decay_weight,
                )
                decayed.append(RecallResult(
                    record=r.record,
                    score=boosted_score,
                    fts_rank=r.fts_rank,
                    vec_score=r.vec_score,
                    graph_score=r.graph_score,
                    temporal_score=r.temporal_score,
                    staleness_warning=r.staleness_warning,
                ))
            decayed.sort(key=lambda r: r.score, reverse=True)
            results = decayed

        # Touch accessed records (lightweight access recording)
        for r in results:
            self._repo.touch(r.record.id)

        return results

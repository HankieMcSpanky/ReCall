"""Memory consolidation: cluster similar memories and merge them into summaries.

Like human sleep consolidation -- reduces noise, strengthens signal.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np

from neuropack.types import ConsolidationResult, MemoryRecord


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Embeddings should already be L2-normalized
    return embeddings @ embeddings.T


def find_clusters(
    records: list[MemoryRecord],
    embeddings: np.ndarray,
    threshold: float = 0.80,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """Find clusters of similar memories using single-linkage agglomerative clustering.

    Returns list of clusters, each cluster is a list of indices into records.
    """
    n = len(records)
    if n < min_cluster_size:
        return []

    sim_matrix = _cosine_similarity_matrix(embeddings)

    # Union-Find for single-linkage clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Link items above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                # Only cluster within same namespace
                if records[i].namespace == records[j].namespace:
                    union(i, j)

    # Collect clusters
    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    # Filter by min size
    return [c for c in clusters_map.values() if len(c) >= min_cluster_size]


def summarize_cluster_extractive(records: list[MemoryRecord]) -> str:
    """Create a summary from a cluster using extractive approach (no LLM)."""
    # Pick the memory with highest priority as the "best" representative
    records_sorted = sorted(records, key=lambda r: r.priority, reverse=True)

    # Combine L3 abstracts
    abstracts = []
    seen = set()
    for r in records_sorted:
        if r.l3_abstract not in seen:
            abstracts.append(r.l3_abstract)
            seen.add(r.l3_abstract)

    # Combine unique L2 facts
    facts = []
    seen_facts: set[str] = set()
    for r in records_sorted:
        for fact in r.l2_facts:
            if fact not in seen_facts:
                facts.append(fact)
                seen_facts.add(fact)

    summary_parts = ["Consolidated from {} memories:".format(len(records))]
    summary_parts.extend(f"- {a}" for a in abstracts[:5])
    if facts:
        summary_parts.append("\nKey facts:")
        summary_parts.extend(f"- {f}" for f in facts[:10])

    return "\n".join(summary_parts)


def summarize_cluster_llm(records: list[MemoryRecord], llm_provider) -> str:
    """Create a summary from a cluster using LLM."""
    content_block = "\n---\n".join(
        f"Memory {i+1} (priority={r.priority}):\n{r.content[:500]}"
        for i, r in enumerate(records[:10])
    )
    prompt = (
        "Consolidate these related memories into a single, comprehensive summary. "
        "Preserve all key facts, decisions, and context. Remove redundancy.\n\n"
        f"{content_block}\n\n"
        "Write a clear consolidated summary:"
    )
    return llm_provider.complete(prompt)


class MemoryConsolidator:
    """Consolidate clusters of similar memories into summaries."""

    def __init__(self, store, config):
        self._store = store
        self._config = config

    def consolidate(
        self,
        namespace: str | None = None,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Run consolidation. Returns result with stats.

        If dry_run=True, reports what would be consolidated without modifying data.
        """
        repo = self._store._repo
        items = repo.get_all_with_embeddings(namespace=namespace)

        if not items:
            return ConsolidationResult(0, 0, 0)

        records = [r for r, _ in items]
        embeddings = np.vstack([e.reshape(1, -1) for _, e in items]).astype(np.float32)

        clusters = find_clusters(
            records, embeddings,
            threshold=self._config.consolidation_threshold,
            min_cluster_size=self._config.consolidation_min_cluster,
        )

        if not clusters or dry_run:
            total_memories = sum(len(c) for c in clusters)
            return ConsolidationResult(
                clusters_found=len(clusters),
                memories_consolidated=total_memories,
                summaries_created=0,
            )

        # Get LLM provider if available
        llm = None
        try:
            default_llm = self._store._llm_registry.get_default()
            if default_llm is not None:
                from neuropack.llm.provider import LLMProvider
                llm = LLMProvider(default_llm)
        except Exception:
            pass

        archived_ids: list[str] = []
        summaries_created = 0

        for cluster_indices in clusters:
            cluster_records = [records[i] for i in cluster_indices]

            # Generate summary
            if llm:
                try:
                    summary_text = summarize_cluster_llm(cluster_records, llm)
                except Exception:
                    summary_text = summarize_cluster_extractive(cluster_records)
            else:
                summary_text = summarize_cluster_extractive(cluster_records)

            # Merge tags from all cluster members
            all_tags = set()
            for r in cluster_records:
                all_tags.update(r.tags)
            all_tags.add("consolidated")

            # Use highest priority from cluster
            max_priority = max(r.priority for r in cluster_records)
            ns = cluster_records[0].namespace

            # Store the consolidated memory
            consolidated = self._store.store(
                content=summary_text,
                tags=list(all_tags),
                source="consolidation",
                priority=max_priority,
                namespace=ns,
            )
            summaries_created += 1

            # Mark originals as superseded
            for r in cluster_records:
                repo.save_version(r, reason="consolidation")
                repo.update(r.id, superseded_by=consolidated.id)
                archived_ids.append(r.id)

        return ConsolidationResult(
            clusters_found=len(clusters),
            memories_consolidated=len(archived_ids),
            summaries_created=summaries_created,
            archived_ids=archived_ids,
        )

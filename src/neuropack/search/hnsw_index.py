"""HNSW vector index using hnswlib. Falls back to BruteForceIndex if hnswlib is not installed."""
from __future__ import annotations

import numpy as np

from neuropack.search.vector_index import BruteForceIndex

try:
    import hnswlib

    _HAS_HNSWLIB = True
except ImportError:
    _HAS_HNSWLIB = False


class HNSWIndex:
    """Approximate nearest neighbor search using HNSW (hnswlib).

    Same interface as BruteForceIndex for drop-in replacement.
    Requires: pip install hnswlib
    """

    def __init__(self, dim: int = 256, max_elements: int = 100_000, ef_construction: int = 200, m: int = 16):
        if not _HAS_HNSWLIB:
            raise ImportError("hnswlib is required for HNSWIndex. Install with: pip install hnswlib")
        self._dim = dim
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._m = m
        self._ids: list[str] = []
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0
        self._index: hnswlib.Index | None = None

    def _ensure_index(self) -> hnswlib.Index:
        if self._index is None:
            self._index = hnswlib.Index(space="cosine", dim=self._dim)
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._m,
            )
            self._index.set_ef(50)
        return self._index

    def build(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Build index from list of (id, embedding) pairs."""
        self._ids = []
        self._id_to_label = {}
        self._next_label = 0
        self._index = None

        if not items:
            return

        # Detect dimension from first item
        self._dim = items[0][1].shape[0]
        index = self._ensure_index()

        # Resize if needed
        if len(items) > self._max_elements:
            self._max_elements = len(items) * 2
            index.resize_index(self._max_elements)

        vectors = np.vstack([emb.reshape(1, -1) for _, emb in items]).astype(np.float32)
        labels = list(range(len(items)))

        for i, (mid, _) in enumerate(items):
            self._ids.append(mid)
            self._id_to_label[mid] = i

        self._next_label = len(items)
        index.add_items(vectors, labels)

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Add a single item to the index."""
        if self._dim != embedding.shape[0] and self._index is not None:
            # Dimension mismatch -- rebuild needed
            return

        if self._dim == 256 and self._index is None:
            self._dim = embedding.shape[0]

        index = self._ensure_index()

        # Resize if at capacity
        if self._next_label >= self._max_elements:
            self._max_elements = self._max_elements * 2
            index.resize_index(self._max_elements)

        label = self._next_label
        self._next_label += 1
        self._ids.append(memory_id)
        self._id_to_label[memory_id] = label

        vec = embedding.reshape(1, -1).astype(np.float32)
        index.add_items(vec, [label])

    def remove(self, memory_id: str) -> None:
        """Mark an item as deleted. hnswlib supports lazy deletion."""
        if memory_id not in self._id_to_label:
            return
        label = self._id_to_label[memory_id]
        if self._index is not None:
            try:
                self._index.mark_deleted(label)
            except RuntimeError:
                pass  # Already deleted
        del self._id_to_label[memory_id]
        # Don't remove from _ids list to keep labels stable

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """Return top-k (id, cosine_similarity) pairs."""
        if self._index is None or not self._id_to_label:
            return []

        k = min(k, len(self._id_to_label))
        if k <= 0:
            return []

        query = query_vec.reshape(1, -1).astype(np.float32)
        try:
            labels, distances = self._index.knn_query(query, k=k)
        except RuntimeError:
            return []

        # Build label -> id reverse map
        label_to_id: dict[int, str] = {v: k_ for k_, v in self._id_to_label.items()}

        results: list[tuple[str, float]] = []
        for label, dist in zip(labels[0], distances[0]):
            mid = label_to_id.get(int(label))
            if mid is not None:
                # hnswlib cosine distance = 1 - cosine_similarity
                similarity = 1.0 - float(dist)
                results.append((mid, similarity))

        return results

    def __len__(self) -> int:
        return len(self._id_to_label)


def create_vector_index(dim: int = 256, use_hnsw: bool = True) -> BruteForceIndex | HNSWIndex:
    """Factory: create the best available vector index.

    Uses HNSW if hnswlib is installed, otherwise falls back to brute-force.
    """
    if use_hnsw and _HAS_HNSWLIB:
        return HNSWIndex(dim=dim)
    return BruteForceIndex()

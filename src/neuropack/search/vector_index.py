from __future__ import annotations

import numpy as np


class BruteForceIndex:
    """Brute-force cosine similarity search over L2-normalized vectors."""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def build(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Build index from list of (id, embedding) pairs."""
        if not items:
            self._ids = []
            self._matrix = None
            return
        self._ids = [id_ for id_, _ in items]
        self._matrix = np.vstack([emb.reshape(1, -1) for _, emb in items]).astype(np.float32)

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Add a single item to the index."""
        row = embedding.reshape(1, -1).astype(np.float32)
        self._ids.append(memory_id)
        if self._matrix is None:
            self._matrix = row
        else:
            self._matrix = np.vstack([self._matrix, row])

    def remove(self, memory_id: str) -> None:
        """Remove an item by ID."""
        if memory_id not in self._ids:
            return
        idx = self._ids.index(memory_id)
        self._ids.pop(idx)
        if self._matrix is not None:
            self._matrix = np.delete(self._matrix, idx, axis=0)
            if len(self._matrix) == 0:
                self._matrix = None

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """Return top-k (id, cosine_similarity) pairs. Vectors must be L2-normalized."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        query = query_vec.astype(np.float32).ravel()
        scores = self._matrix @ query

        k = min(k, len(scores))
        if k <= 0:
            return []

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        return [(self._ids[i], float(scores[i])) for i in top_k_idx]

    def __len__(self) -> int:
        return len(self._ids)

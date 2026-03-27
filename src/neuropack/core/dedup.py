from __future__ import annotations

import numpy as np

from neuropack.search.vector_index import BruteForceIndex


class Deduplicator:
    """Detects near-duplicate memories using cosine similarity threshold."""

    def __init__(self, vector_index: BruteForceIndex, threshold: float = 0.92):
        self._index = vector_index
        self.threshold = threshold

    def find_duplicate(self, embedding: np.ndarray) -> str | None:
        """Check if a near-duplicate exists. Returns memory ID if found, else None."""
        results = self._index.search(embedding, k=1)
        if results and results[0][1] >= self.threshold:
            return results[0][0]
        return None

"""Sentence-transformer embedder (384-dim dense vectors)."""
from __future__ import annotations

import numpy as np

from neuropack.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Dense embedder using sentence-transformers (all-MiniLM-L6-v2 by default, 384d)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for dense embeddings. "
                "Install with: pip install 'neuropack[transformers]'"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch-embed using sentence-transformers native batch encoding."""
        if not texts:
            return []
        # Replace empty strings so the model doesn't choke
        sanitized = [t if t.strip() else " " for t in texts]
        vectors = self._model.encode(
            sanitized, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=128, show_progress_bar=False,
        )
        result: list[np.ndarray] = []
        for i, vec in enumerate(vectors):
            if not texts[i].strip():
                result.append(np.zeros(self._dim, dtype=np.float32))
            else:
                result.append(vec.astype(np.float32))
        return result

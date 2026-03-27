from __future__ import annotations

import json
import math

import mmh3
import numpy as np

from neuropack.embeddings.base import Embedder
from neuropack.embeddings.tokenizer import bigrams, tokenize


class FeatureHashedTFIDF(Embedder):
    """256-dim TF-IDF embedder using feature hashing. No vocabulary storage needed."""

    def __init__(self, dim: int = 256, use_bigrams: bool = True):
        self._dim = dim
        self.use_bigrams = use_bigrams
        self._doc_count: int = 0
        self._term_doc_freq: dict[int, int] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Produce a dim-dimensional L2-normalized embedding."""
        tokens = tokenize(text)
        if self.use_bigrams:
            tokens = tokens + bigrams(tokens)

        if not tokens:
            return np.zeros(self._dim, dtype=np.float32)

        # Term frequency with signed hashing trick
        tf: dict[int, float] = {}
        for token in tokens:
            h = mmh3.hash(token, signed=False) % self._dim
            sign = 1.0 if mmh3.hash(token, seed=1, signed=True) >= 0 else -1.0
            tf[h] = tf.get(h, 0.0) + sign

        # Sublinear TF: sign * (1 + log(|count|))
        for h in tf:
            val = tf[h]
            if val != 0:
                tf[h] = math.copysign(1.0 + math.log(abs(val)), val)

        # Apply IDF weighting (default IDF=1.0 when no documents indexed yet)
        vec = np.zeros(self._dim, dtype=np.float32)
        for h, val in tf.items():
            if self._doc_count == 0:
                idf = 1.0
            else:
                idf = math.log(1.0 + self._doc_count / (1.0 + self._term_doc_freq.get(h, 0)))
            vec[h] = val * idf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def embed_with_prefix(self, prefix: str, content: str) -> np.ndarray:
        """Embed prefix + content together so TF-IDF vectors capture contextual terms."""
        combined = f"{prefix} {content}" if prefix else content
        return self.embed(combined)

    def update_idf(self, text: str) -> None:
        """Update document frequency counts after storing a new document."""
        tokens = tokenize(text)
        if self.use_bigrams:
            tokens = tokens + bigrams(tokens)

        seen_buckets: set[int] = set()
        for token in tokens:
            h = mmh3.hash(token, signed=False) % self._dim
            seen_buckets.add(h)

        for h in seen_buckets:
            self._term_doc_freq[h] = self._term_doc_freq.get(h, 0) + 1
        self._doc_count += 1

    def save_state(self) -> str:
        """Serialize IDF state to JSON string for persistence."""
        return json.dumps({
            "doc_count": self._doc_count,
            "term_doc_freq": {str(k): v for k, v in self._term_doc_freq.items()},
        })

    def load_state(self, state_json: str) -> None:
        """Restore IDF state from JSON string."""
        state = json.loads(state_json)
        self._doc_count = state["doc_count"]
        self._term_doc_freq = {int(k): v for k, v in state["term_doc_freq"].items()}

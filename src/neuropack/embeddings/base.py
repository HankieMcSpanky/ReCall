"""Abstract base class for all NeuroPack embedders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Base interface for text embedding."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Produce an L2-normalized embedding vector."""

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts at once.

        Default implementation calls embed() in a loop. Subclasses (e.g.,
        SentenceTransformerEmbedder, OpenAIEmbedder) should override this
        with a native batch implementation for much better performance.
        """
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""

    def update_idf(self, text: str) -> None:
        """Update document frequency stats (no-op by default)."""

    def save_state(self) -> str:
        """Serialize internal state for persistence. Returns empty JSON by default."""
        return "{}"

    def load_state(self, state_json: str) -> None:
        """Restore internal state from JSON string."""

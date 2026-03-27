"""OpenAI text-embedding-3-large embedder (API-based, no SDK dependency)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from neuropack.embeddings.base import Embedder


class OpenAIEmbedder(Embedder):
    """Dense embedder using OpenAI's embedding API via urllib.

    Default model: text-embedding-3-large (3072 native dimensions).
    Supports Matryoshka dimension truncation via the ``dimensions`` parameter
    (e.g., 1024 or 768 for faster search with minimal quality loss).

    Reads ``OPENAI_API_KEY`` from the environment.
    """

    _API_URL = "https://api.openai.com/v1/embeddings"
    _MAX_BATCH = 50  # Smaller batches to avoid token limits on large texts
    _MAX_RETRIES = 8
    _BATCH_SLEEP = 0.05  # Small sleep between batch API calls for rate-limit safety

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int = 1024,
        api_key: str = "",
    ):
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to OpenAIEmbedder."
            )

    @property
    def dim(self) -> int:
        return self._dimensions

    # ------------------------------------------------------------------
    # Single embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Produce an L2-normalized embedding vector for a single text."""
        if not text.strip():
            return np.zeros(self._dimensions, dtype=np.float32)
        # Sanitize: strip null bytes, cap length
        text = text.replace("\x00", "").strip()
        if len(text) > 32000:
            text = text[:32000]
        if not text:
            return np.zeros(self._dimensions, dtype=np.float32)
        try:
            result = self._call_api([text])
            return result[0]
        except urllib.error.HTTPError:
            # If the text causes a 400, return zero vector rather than crashing
            return np.zeros(self._dimensions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts efficiently, sending up to 100 per API call.

        Returns a list of L2-normalized np.ndarray vectors in the same order
        as the input texts.
        """
        if not texts:
            return []

        all_embeddings: list[np.ndarray | None] = [None] * len(texts)

        # Process in chunks of _MAX_BATCH
        for chunk_start in range(0, len(texts), self._MAX_BATCH):
            chunk_end = min(chunk_start + self._MAX_BATCH, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]

            # Sanitize: replace empty strings, strip null bytes, cap length
            sanitized = []
            for t in chunk_texts:
                t = t.replace("\x00", "").strip()
                if not t:
                    t = " "
                # Cap at ~8000 tokens (~32000 chars) to stay within model limit
                if len(t) > 32000:
                    t = t[:32000]
                sanitized.append(t)

            try:
                embeddings = self._call_api(sanitized)
            except urllib.error.HTTPError as exc:
                if exc.code == 400:
                    # Batch too large or invalid content — fall back to one-by-one
                    embeddings = []
                    for j, txt in enumerate(sanitized):
                        try:
                            emb = self._call_api([txt])
                            embeddings.extend(emb)
                        except Exception:
                            embeddings.append(np.zeros(self._dimensions, dtype=np.float32))
                else:
                    raise
            for i, emb in enumerate(embeddings):
                # If the original text was empty, return zero vector
                if not chunk_texts[i].strip():
                    emb = np.zeros(self._dimensions, dtype=np.float32)
                all_embeddings[chunk_start + i] = emb

            # Small sleep between batch calls for rate-limit safety
            if chunk_end < len(texts):
                time.sleep(self._BATCH_SLEEP)

        return [e for e in all_embeddings if e is not None]

    # ------------------------------------------------------------------
    # API call with retry logic
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        """Send a batch of texts to the OpenAI embeddings API.

        Returns a list of L2-normalized np.ndarray vectors matching the
        order of the input texts.
        """
        payload = json.dumps({
            "input": texts,
            "model": self._model,
            "dimensions": self._dimensions,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        for attempt in range(self._MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    self._API_URL, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                # The API returns {"data": [{"embedding": [...], "index": 0}, ...]}
                # Sort by index to guarantee order matches input
                data_items = sorted(body["data"], key=lambda d: d["index"])
                result: list[np.ndarray] = []
                for item in data_items:
                    vec = np.array(item["embedding"], dtype=np.float32)
                    # L2-normalize
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    result.append(vec)
                return result

            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self._MAX_RETRIES - 1:
                    # Rate limited -- back off with exponential delay
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            wait = min(float(retry_after) + 1, 120)
                        except ValueError:
                            wait = min(2 ** attempt + 1, 120)
                    else:
                        wait = min(2 ** attempt + 1, 120)
                    time.sleep(wait)
                    continue
                elif exc.code in (500, 502, 503) and attempt < self._MAX_RETRIES - 1:
                    time.sleep(min(2 ** attempt + 1, 60))
                    continue
                raise
            except urllib.error.URLError:
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(min(2 ** attempt + 1, 60))
                    continue
                raise

        # Should not reach here, but return zeros as a safety fallback
        return [np.zeros(self._dimensions, dtype=np.float32) for _ in texts]

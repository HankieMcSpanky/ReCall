"""Memory poisoning defense: provenance tracking, trust scoring, anomaly detection."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from neuropack.types import MemoryRecord


class TrustScorer:
    """Bayesian trust scoring per source.

    Each source starts at prior (default 0.5). Successful recall (accessed)
    increases trust; contradictions/reports decrease it.
    """

    def __init__(self, prior: float = 0.5, learning_rate: float = 0.05):
        self._prior = prior
        self._lr = learning_rate
        # source -> (successes, failures)
        self._counts: dict[str, tuple[int, int]] = {}

    def trust_score(self, source: str) -> float:
        """Bayesian trust: (alpha) / (alpha + beta) where alpha = prior + successes."""
        s, f = self._counts.get(source, (0, 0))
        alpha = self._prior * 10 + s
        beta = (1 - self._prior) * 10 + f
        return alpha / (alpha + beta)

    def record_success(self, source: str) -> None:
        s, f = self._counts.get(source, (0, 0))
        self._counts[source] = (s + 1, f)

    def record_failure(self, source: str) -> None:
        s, f = self._counts.get(source, (0, 0))
        self._counts[source] = (s, f + 1)

    def load_state(self, data: dict) -> None:
        self._counts = {k: tuple(v) for k, v in data.items()}

    def save_state(self) -> dict:
        return {k: list(v) for k, v in self._counts.items()}


class AnomalyDetector:
    """Detect memories that are outliers in embedding space.

    Uses distance from centroid as anomaly signal.
    """

    def __init__(self, threshold_sigma: float = 3.0):
        self._threshold = threshold_sigma
        self._centroid: np.ndarray | None = None
        self._mean_dist: float = 0.0
        self._std_dist: float = 1.0
        self._n: int = 0

    def fit(self, embeddings: list[tuple[str, np.ndarray]]) -> None:
        """Compute centroid and distance statistics from existing memories."""
        if not embeddings:
            return
        vecs = np.array([e for _, e in embeddings], dtype=np.float32)
        self._centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - self._centroid, axis=1)
        self._mean_dist = float(dists.mean())
        self._std_dist = float(dists.std()) if len(dists) > 1 else 1.0
        self._n = len(embeddings)

    def is_anomaly(self, embedding: np.ndarray) -> tuple[bool, float]:
        """Check if an embedding is an outlier. Returns (is_anomaly, z_score)."""
        if self._centroid is None or self._n < 10:
            return False, 0.0
        dist = float(np.linalg.norm(embedding - self._centroid))
        if self._std_dist == 0:
            return False, 0.0
        z = (dist - self._mean_dist) / self._std_dist
        return z > self._threshold, z

    def update(self, embedding: np.ndarray) -> None:
        """Incrementally update centroid (online mean)."""
        if self._centroid is None:
            self._centroid = embedding.copy()
            self._n = 1
            return
        self._n += 1
        self._centroid = self._centroid + (embedding - self._centroid) / self._n


def check_memory_trust(
    record: MemoryRecord,
    trust_scorer: TrustScorer,
    anomaly_detector: AnomalyDetector,
    embedding: np.ndarray | None = None,
) -> dict:
    """Evaluate a memory's trustworthiness. Returns trust report."""
    source = record.source or "unknown"
    trust = trust_scorer.trust_score(source)

    anomaly = False
    z_score = 0.0
    if embedding is not None:
        anomaly, z_score = anomaly_detector.is_anomaly(embedding)

    warnings = []
    if trust < 0.3:
        warnings.append(f"Low trust source '{source}' (score={trust:.2f})")
    if anomaly:
        warnings.append(f"Anomalous embedding (z={z_score:.1f})")

    return {
        "source": source,
        "trust_score": round(trust, 3),
        "is_anomaly": anomaly,
        "z_score": round(z_score, 2),
        "warnings": warnings,
        "is_trusted": trust >= 0.3 and not anomaly,
    }

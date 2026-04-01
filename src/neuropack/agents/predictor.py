"""Predictive Pre-loading — anticipate what the user will need next.

Based on: current time, recent queries, activity patterns, and entity
co-occurrence, predict which memories will be needed and pre-load them.

This is the prediction layer on top of the existing watcher infrastructure.

Usage:
    from neuropack.agents.predictor import MemoryPredictor

    predictor = MemoryPredictor(store)
    predictor.record_query("authentication bug")

    # Get predictions for what the user will ask next
    predictions = predictor.predict_next(top_k=5)
    for p in predictions:
        print(f"{p.query} (confidence: {p.confidence:.2f})")

    # Pre-load predicted memories into cache
    pre_loaded = predictor.pre_load()
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A predicted future query with confidence."""
    query: str
    confidence: float  # 0-1
    reason: str  # why we think this
    related_entities: list[str] = field(default_factory=list)


class MemoryPredictor:
    """Predicts what memories the user will need next."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._query_history: list[tuple[float, str]] = []  # (timestamp, query)
        self._entity_cooccurrence: dict[str, Counter] = defaultdict(Counter)
        self._time_patterns: dict[int, Counter] = defaultdict(Counter)  # hour -> topic counts
        self._session_topics: list[str] = []  # topics in current session

    def record_query(self, query: str) -> None:
        """Record a query for pattern learning."""
        now = time.time()
        self._query_history.append((now, query))

        # Track time-of-day patterns
        hour = datetime.now().hour
        words = self._extract_topics(query)
        for word in words:
            self._time_patterns[hour][word] += 1

        # Track entity co-occurrence (queries close in time share context)
        self._session_topics.extend(words)
        if len(self._session_topics) > 1:
            for i, w1 in enumerate(self._session_topics[-5:]):
                for w2 in self._session_topics[-5:]:
                    if w1 != w2:
                        self._entity_cooccurrence[w1][w2] += 1

    def predict_next(self, top_k: int = 5) -> list[Prediction]:
        """Predict what the user will query next."""
        predictions: list[Prediction] = []

        # Strategy 1: Co-occurrence — if user just asked about X, they often ask about Y
        if self._session_topics:
            recent_topic = self._session_topics[-1]
            if recent_topic in self._entity_cooccurrence:
                cooc = self._entity_cooccurrence[recent_topic].most_common(3)
                for topic, count in cooc:
                    if count >= 2:
                        predictions.append(Prediction(
                            query=topic,
                            confidence=min(count / 10, 0.9),
                            reason=f"Often discussed with '{recent_topic}'",
                            related_entities=[recent_topic, topic],
                        ))

        # Strategy 2: Time-of-day — certain topics are discussed at certain times
        hour = datetime.now().hour
        if hour in self._time_patterns:
            top_topics = self._time_patterns[hour].most_common(3)
            for topic, count in top_topics:
                if count >= 3 and topic not in [p.query for p in predictions]:
                    predictions.append(Prediction(
                        query=topic,
                        confidence=min(count / 15, 0.7),
                        reason=f"Usually discussed around {hour}:00",
                        related_entities=[topic],
                    ))

        # Strategy 3: Sequence patterns — detect A→B→C query chains
        if len(self._query_history) >= 3:
            recent = [q for _, q in self._query_history[-10:]]
            # Find bigrams
            bigrams: Counter = Counter()
            for i in range(len(recent) - 1):
                t1 = self._extract_topics(recent[i])
                t2 = self._extract_topics(recent[i + 1])
                for w1 in t1:
                    for w2 in t2:
                        if w1 != w2:
                            bigrams[(w1, w2)] += 1

            last_topics = self._extract_topics(recent[-1])
            for (t1, t2), count in bigrams.most_common(5):
                if t1 in last_topics and count >= 2:
                    if t2 not in [p.query for p in predictions]:
                        predictions.append(Prediction(
                            query=t2,
                            confidence=min(count / 5, 0.8),
                            reason=f"After '{t1}', you usually look at '{t2}'",
                            related_entities=[t1, t2],
                        ))

        # Sort by confidence
        predictions.sort(key=lambda p: -p.confidence)
        return predictions[:top_k]

    def pre_load(self, top_k: int = 5) -> list[Any]:
        """Pre-load predicted memories into the store's cache."""
        predictions = self.predict_next(top_k)
        pre_loaded = []
        for pred in predictions:
            try:
                results = self._store.recall(query=pred.query, limit=3)
                pre_loaded.extend(results)
            except Exception:
                pass
        return pre_loaded

    def get_context_hint(self) -> str:
        """Get a one-line hint about the user's current context.

        Useful for injecting into LLM system prompts.
        """
        if not self._session_topics:
            return ""
        recent = list(dict.fromkeys(reversed(self._session_topics[-5:])))
        return f"User is currently focused on: {', '.join(recent[:3])}"

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        """Extract meaningful topic words from text."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "what",
            "how", "when", "where", "which", "who", "why", "that",
            "this", "with", "from", "about", "into", "for", "and",
            "but", "not", "you", "your", "my", "our", "their",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in stop][:5]

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from neuropack.types import MemoryRecord


class PriorityScorer:
    """Computes time-decayed, access-boosted, feedback-adjusted priority for ranking."""

    def __init__(self, decay_half_life_days: float = 30.0):
        self.decay_hl = decay_half_life_days
        # Feedback scores: memory_id -> cumulative feedback value
        self._feedback: dict[str, float] = {}

    def adjusted_priority(
        self, record: MemoryRecord, now: datetime | None = None
    ) -> float:
        if now is None:
            now = datetime.now(timezone.utc)

        # Time decay
        age_seconds = (now - record.created_at).total_seconds()
        age_days = max(age_seconds / 86400.0, 0.0)
        decay = 0.5 ** (age_days / self.decay_hl) if self.decay_hl > 0 else 1.0

        # Access frequency boost
        access_boost = 1.0 + math.log(1.0 + record.access_count) / 10.0

        # Recency boost: recently accessed memories get a bump
        recency_boost = 1.0
        if record.last_accessed is not None:
            last_days = (now - record.last_accessed).total_seconds() / 86400.0
            recency_boost = 1.0 + 0.2 * (0.5 ** (last_days / 7.0))  # 7-day half-life

        # Feedback adjustment: +/- from user feedback
        feedback_adj = self._feedback.get(record.id, 0.0)
        feedback_mult = 1.0 + 0.1 * math.tanh(feedback_adj)  # Bounded [-0.1, +0.1]

        return record.priority * decay * access_boost * recency_boost * feedback_mult

    def record_feedback(self, memory_id: str, useful: bool) -> None:
        """Record user feedback for a memory result."""
        delta = 1.0 if useful else -1.0
        self._feedback[memory_id] = self._feedback.get(memory_id, 0.0) + delta

    def load_feedback(self, data: str) -> None:
        """Load feedback state from JSON string."""
        if data:
            self._feedback = json.loads(data)

    def save_feedback(self) -> str:
        """Save feedback state as JSON string."""
        return json.dumps(self._feedback)

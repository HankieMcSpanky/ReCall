"""Ebbinghaus-inspired memory decay with ACT-R base-level activation."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from neuropack.types import MemoryRecord


class MemoryDecay:
    """Computes memory strength and recall probability using ACT-R activation
    and Ebbinghaus forgetting curve models.

    The core idea: memories accessed more frequently and at increasing intervals
    are "stronger" — just like human memory consolidation.
    """

    def __init__(self, decay_rate: float = 0.5):
        self._decay_rate = decay_rate

    def compute_strength(
        self,
        access_times: list[datetime],
        created_at: datetime,
        current_time: datetime | None = None,
    ) -> float:
        """ACT-R base-level activation: B = ln(sum((t_now - t_j)^(-d)))

        Args:
            access_times: List of datetime objects when the memory was accessed.
            created_at: When the memory was first created.
            current_time: Current time (defaults to now UTC).

        Returns:
            A float score where higher = stronger memory.
        """
        now = current_time or datetime.now(timezone.utc)

        # If never accessed, use created_at as the single access time
        times = access_times if access_times else [created_at]

        activation_sum = 0.0
        for t_j in times:
            # Ensure timezone-aware comparison
            if t_j.tzinfo is None:
                t_j = t_j.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            delta_seconds = (now - t_j).total_seconds()
            if delta_seconds <= 0:
                delta_seconds = 1.0  # Avoid zero/negative time deltas

            # Convert to hours for more reasonable scale
            delta_hours = delta_seconds / 3600.0
            activation_sum += delta_hours ** (-self._decay_rate)

        if activation_sum <= 0:
            return -10.0  # Very weak memory

        return math.log(activation_sum)

    def compute_recall_probability(
        self,
        last_accessed: datetime,
        strength: float,
        current_time: datetime | None = None,
    ) -> float:
        """Ebbinghaus forgetting curve: R = e^(-t/S)

        Args:
            last_accessed: When the memory was last accessed.
            strength: The memory strength (from compute_strength).
            current_time: Current time (defaults to now UTC).

        Returns:
            Recall probability between 0 and 1.
        """
        now = current_time or datetime.now(timezone.utc)

        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta_hours = (now - last_accessed).total_seconds() / 3600.0
        if delta_hours <= 0:
            return 1.0

        # Use exp(strength) as the effective stability S to keep S positive
        stability = math.exp(strength) if strength > -20 else 1e-9
        if stability <= 0:
            stability = 1e-9

        recall_prob = math.exp(-delta_hours / stability)
        return max(0.0, min(1.0, recall_prob))

    def apply_decay_boost(
        self,
        base_score: float,
        memory: MemoryRecord,
        decay_weight: float = 0.1,
        current_time: datetime | None = None,
    ) -> float:
        """Combine retrieval score with memory strength for decay-boosted ranking.

        Frequently accessed memories get a subtle boost. The decay_weight
        controls how much influence strength has on the final score.

        Args:
            base_score: The original retrieval score (e.g. from RRF + reranking).
            memory: The MemoryRecord being scored.
            decay_weight: How much decay boost influences the final score.
            current_time: Current time (defaults to now UTC).

        Returns:
            The boosted score: base_score * (1 + strength * decay_weight).
        """
        now = current_time or datetime.now(timezone.utc)

        # Build access times from available data
        access_times: list[datetime] = []
        if memory.last_accessed is not None:
            # We approximate: the memory was accessed access_count times,
            # with the most recent being last_accessed and creation being
            # the first. We use these two anchors for the activation formula.
            access_times.append(memory.created_at)
            if memory.access_count > 1 and memory.last_accessed != memory.created_at:
                access_times.append(memory.last_accessed)
                # For intermediate accesses, interpolate evenly between
                # created_at and last_accessed to approximate the spacing
                if memory.access_count > 2:
                    created_ts = memory.created_at.timestamp()
                    last_ts = memory.last_accessed.timestamp()
                    for i in range(1, min(memory.access_count - 1, 10)):
                        frac = i / (memory.access_count - 1)
                        interp_ts = created_ts + frac * (last_ts - created_ts)
                        interp_dt = datetime.fromtimestamp(interp_ts, tz=timezone.utc)
                        access_times.append(interp_dt)
        else:
            access_times = [memory.created_at]

        strength = self.compute_strength(access_times, memory.created_at, now)

        # Normalize strength to a non-negative boost factor
        # Clamp strength so the boost stays reasonable
        clamped_strength = max(0.0, strength)
        boost = 1.0 + (clamped_strength * decay_weight)

        return base_score * boost

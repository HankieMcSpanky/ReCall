"""Forgetting Curve — Ebbinghaus-style memory relevance decay.

Each memory has a "strength" that:
- Starts at 1.0 when created
- Decays exponentially over time (half-life based on type)
- Gets reinforced (+0.3) every time it's accessed
- Memories with high access_count decay slower (learned stability)

Usage:
    from neuropack.agents.forgetting_curve import ForgettingCurve
    curve = ForgettingCurve()
    strength = curve.compute_strength(memory_record)
    # 0.0-1.0, used to boost/demote recall scores
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from neuropack.types import MemoryRecord

# Half-lives in days by (memory_type, staleness)
_BASE_HALF_LIVES: dict[str, float] = {
    "fact": 365.0,
    "preference": 180.0,
    "procedure": 120.0,
    "decision": 60.0,
    "observation": 30.0,
    "general": 90.0,
    "code": 90.0,
}

_STALENESS_MULTIPLIER: dict[str, float] = {
    "stable": 1.0,
    "semi-stable": 0.5,   # halves the half-life
    "volatile": 0.15,     # ~14 days for a 90-day base
}

# Reinforcement: each access reduces effective age by this fraction
_ACCESS_DECAY_REDUCTION = 0.20
# Cap on total reinforcement boost (multiplicative on half-life)
_MAX_ACCESS_BOOST = 3.0


class ForgettingCurve:
    """Ebbinghaus-inspired memory strength calculator."""

    @staticmethod
    def get_half_life(memory_type: str, staleness: str) -> float:
        """Return half-life in days for a given memory type and staleness."""
        base = _BASE_HALF_LIVES.get(memory_type, 90.0)
        mult = _STALENESS_MULTIPLIER.get(staleness, 1.0)
        return base * mult

    def compute_strength(self, record: MemoryRecord, now: datetime | None = None) -> float:
        """Compute current strength (0.0-1.0) for a single memory record."""
        if now is None:
            now = datetime.now(timezone.utc)

        created = record.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        age_days = max((now - created).total_seconds() / 86400.0, 0.0)

        # Each access reduces the effective age by 20%, compounding
        access_count = record.access_count or 0
        effective_age = age_days * ((1.0 - _ACCESS_DECAY_REDUCTION) ** min(access_count, 30))

        # Access count also extends the half-life (learned stability)
        half_life = self.get_half_life(record.memory_type, record.staleness)
        stability_boost = 1.0 + 0.1 * min(access_count, 20)
        stability_boost = min(stability_boost, _MAX_ACCESS_BOOST)
        effective_half_life = half_life * stability_boost

        if effective_half_life <= 0:
            return 0.0

        # Exponential decay: S = 2^(-t / half_life)
        strength = math.pow(2.0, -effective_age / effective_half_life)
        return max(0.0, min(1.0, strength))

    def compute_batch(self, records: list[Any], now: datetime | None = None) -> dict[str, float]:
        """Batch compute strengths. Returns {memory_id: strength}."""
        if now is None:
            now = datetime.now(timezone.utc)
        return {r.id: self.compute_strength(r, now) for r in records}

    def should_consolidate(self, record: MemoryRecord, now: datetime | None = None) -> bool:
        """True if strength < 0.1 — candidate for archival / consolidation."""
        return self.compute_strength(record, now) < 0.1

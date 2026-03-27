"""Staleness detection: identify memories that may be outdated."""
from __future__ import annotations

from datetime import datetime, timezone

from neuropack.types import MemoryRecord


def staleness_age_days(record: MemoryRecord) -> float:
    """Days since the memory was last updated."""
    now = datetime.now(timezone.utc)
    return (now - record.updated_at).total_seconds() / 86400


def check_staleness(
    record: MemoryRecord,
    volatile_days: int = 30,
    semi_stable_days: int = 90,
) -> str | None:
    """Check if a memory is potentially stale. Returns a warning string or None."""
    age = staleness_age_days(record)

    if record.superseded_by:
        return f"Superseded by {record.superseded_by}"

    if record.staleness == "volatile" and age > volatile_days:
        return f"Volatile memory is {int(age)} days old (threshold: {volatile_days}d)"

    if record.staleness == "semi-stable" and age > semi_stable_days:
        return f"Semi-stable memory is {int(age)} days old (threshold: {semi_stable_days}d)"

    # Stable memories don't get staleness warnings
    return None


def get_stale_summary(
    records: list[MemoryRecord],
    volatile_days: int = 30,
    semi_stable_days: int = 90,
) -> list[dict]:
    """Return a list of stale memories with their warnings."""
    stale = []
    for record in records:
        warning = check_staleness(record, volatile_days, semi_stable_days)
        if warning:
            stale.append({
                "id": record.id,
                "l3_abstract": record.l3_abstract,
                "staleness": record.staleness,
                "memory_type": record.memory_type,
                "age_days": int(staleness_age_days(record)),
                "warning": warning,
                "tags": record.tags,
            })
    return stale

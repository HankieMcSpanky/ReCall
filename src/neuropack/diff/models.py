from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class MemoryChangeSummary:
    """Summary of a newly created memory."""
    id: str
    l3_abstract: str
    tags: list[str]
    created_at: datetime
    source: str


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    """Summary of a memory that was updated in the time range."""
    id: str
    old_l3: str
    new_l3: str
    old_tags: list[str]
    new_tags: list[str]
    changed_at: datetime


@dataclass(frozen=True, slots=True)
class DiffStats:
    """Aggregate statistics for a diff."""
    added: int
    modified: int
    deleted: int
    topics_added: list[str]
    topics_removed: list[str]


@dataclass(frozen=True, slots=True)
class MemoryDiff:
    """The complete diff between two points in time."""
    since: datetime
    until: datetime
    new_memories: list[MemoryChangeSummary]
    updated_memories: list[MemoryUpdate]
    deleted_ids: list[str]
    stats: DiffStats


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """A single period in a knowledge timeline."""
    period: str
    period_label: str
    added: int
    modified: int
    deleted: int
    top_tags: list[str]

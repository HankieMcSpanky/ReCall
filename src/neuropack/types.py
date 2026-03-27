from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Valid memory type values
MEMORY_TYPES = ("general", "fact", "decision", "preference", "procedure", "observation", "code")

# Staleness categories
STALENESS_CATEGORIES = ("stable", "semi-stable", "volatile")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    l3_abstract: str
    l2_facts: list[str]
    l1_compressed: bytes
    embedding: list[float]
    tags: list[str]
    source: str
    priority: float
    created_at: datetime
    updated_at: datetime
    namespace: str = "default"
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    content_tokens: int = 0
    compressed_tokens: int = 0
    memory_type: str = "general"
    staleness: str = "stable"
    superseded_by: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RecallResult:
    record: MemoryRecord
    score: float
    fts_rank: Optional[float] = None
    vec_score: Optional[float] = None
    graph_score: Optional[float] = None
    temporal_score: Optional[float] = None
    staleness_warning: Optional[str] = None


@dataclass
class StoreStats:
    total_memories: int
    total_size_bytes: int
    avg_compression_ratio: float
    oldest: Optional[datetime] = None
    newest: Optional[datetime] = None
    total_content_tokens: int = 0
    total_compressed_tokens: int = 0
    token_savings_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class CompressedMemory:
    l3: str
    l2: list[str]
    l1: bytes


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    """A previous version of a memory, saved before update/merge."""
    memory_id: str
    version: int
    content: str
    l3_abstract: str
    tags: list[str]
    saved_at: datetime
    reason: str = "update"


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation."""
    clusters_found: int
    memories_consolidated: int
    summaries_created: int
    archived_ids: list[str] = field(default_factory=list)


# --- Workspace Types ---

WORKSPACE_STATUSES = ("active", "completed", "archived")
TASK_STATUSES = ("open", "claimed", "blocked", "done")


@dataclass(frozen=True, slots=True)
class Workspace:
    """A shared collaboration workspace."""
    id: str
    name: str
    goal: str
    status: str  # active, completed, archived
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"


@dataclass(frozen=True, slots=True)
class WorkspaceMember:
    """An agent participating in a workspace."""
    workspace_id: str
    agent_name: str
    role: str  # owner, member
    joined_at: datetime
    last_seen: datetime


@dataclass(frozen=True, slots=True)
class WorkspaceTask:
    """A task on the workspace board."""
    id: str
    workspace_id: str
    title: str
    description: str
    status: str  # open, claimed, blocked, done
    created_by: str
    assigned_to: Optional[str]
    blocked_by: list[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


@dataclass(frozen=True, slots=True)
class Handoff:
    """A structured handoff between agents."""
    id: str
    workspace_id: str
    from_agent: str
    to_agent: Optional[str]
    task_id: Optional[str]
    summary: str
    context: dict  # {findings, decisions, open_questions}
    memory_ids: list[str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Decision:
    """A logged decision with rationale."""
    id: str
    workspace_id: str
    title: str
    rationale: str
    decided_by: str
    alternatives: list[str]
    related_task_id: Optional[str]
    created_at: datetime

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# Valid event types
EVENT_TYPES = (
    "file_modified",
    "file_created",
    "git_commit",
    "git_branch",
    "git_diff",
    "terminal_command",
)


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    """An event observed by one of the watchers."""
    type: str
    path: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

"""Agent-managed memory lifecycle: promote, demote, archive, pin."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from neuropack.storage.database import Database
from neuropack.storage.repository import MemoryRepository


class AgentMemoryManager:
    """Allows agents to control their own memory lifecycle.

    Operations: promote (boost priority), demote (lower priority),
    archive (save version + tag), pin (never auto-purge), create_working_memory.
    """

    def __init__(
        self,
        db: Database,
        repo: MemoryRepository,
        agent_name: str,
        store_fn=None,
    ):
        self._db = db
        self._repo = repo
        self._agent = agent_name
        self._store_fn = store_fn  # callable to create new memories

    def promote(self, memory_id: str, priority: float = 0.8) -> bool:
        """Boost a memory's priority and mark as stable."""
        record = self._repo.get_by_id(memory_id)
        if record is None or record.namespace != self._agent:
            return False

        conn = self._db.connect()
        conn.execute(
            "UPDATE memories SET priority = ?, staleness = 'stable' WHERE id = ?",
            (max(priority, record.priority), memory_id),
        )

        # Add 'promoted' tag if not present
        tags = list(record.tags)
        if "promoted" not in tags:
            tags.append("promoted")
            conn.execute(
                "UPDATE memories SET tags = ? WHERE id = ?",
                (json.dumps(tags), memory_id),
            )

        conn.commit()
        return True

    def demote(self, memory_id: str, priority: float = 0.2) -> bool:
        """Lower a memory's priority and mark as volatile."""
        record = self._repo.get_by_id(memory_id)
        if record is None or record.namespace != self._agent:
            return False

        conn = self._db.connect()
        conn.execute(
            "UPDATE memories SET priority = ?, staleness = 'volatile' WHERE id = ?",
            (min(priority, record.priority), memory_id),
        )
        conn.commit()
        return True

    def archive(self, memory_id: str, reason: str = "archived") -> bool:
        """Save a version snapshot, then tag as archived."""
        record = self._repo.get_by_id(memory_id)
        if record is None or record.namespace != self._agent:
            return False

        conn = self._db.connect()
        now = datetime.now(timezone.utc).isoformat()

        # Get next version number
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) as v FROM memory_versions WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        next_version = dict(row)["v"] + 1

        # Save version
        conn.execute(
            """INSERT INTO memory_versions (memory_id, version, content, l3_abstract, tags, saved_at, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                next_version,
                record.content,
                record.l3_abstract,
                json.dumps(record.tags),
                now,
                reason,
            ),
        )

        # Tag as archived and lower priority
        tags = list(record.tags)
        if "archived" not in tags:
            tags.append("archived")
        conn.execute(
            "UPDATE memories SET tags = ?, priority = 0.1, staleness = 'volatile' WHERE id = ?",
            (json.dumps(tags), memory_id),
        )

        conn.commit()
        return True

    def pin(self, memory_id: str) -> bool:
        """Pin a memory: set to max priority, stable, tagged 'pinned'."""
        record = self._repo.get_by_id(memory_id)
        if record is None or record.namespace != self._agent:
            return False

        conn = self._db.connect()
        tags = list(record.tags)
        if "pinned" not in tags:
            tags.append("pinned")

        conn.execute(
            "UPDATE memories SET priority = 1.0, staleness = 'stable', tags = ? WHERE id = ?",
            (json.dumps(tags), memory_id),
        )
        conn.commit()
        return True

    def get_pinned(self) -> list[dict]:
        """List all pinned memories for this agent."""
        conn = self._db.connect()
        rows = conn.execute(
            """SELECT id, l3_abstract, priority, tags FROM memories
               WHERE namespace = ? AND tags LIKE '%pinned%'
               ORDER BY priority DESC""",
            (self._agent,),
        ).fetchall()

        return [
            {
                "id": dict(r)["id"],
                "l3_abstract": dict(r)["l3_abstract"],
                "priority": dict(r)["priority"],
                "tags": json.loads(dict(r)["tags"]),
            }
            for r in rows
        ]

    def create_working_memory(self, content: str) -> str | None:
        """Create a volatile, low-priority memory for temporary working state."""
        if not self._store_fn:
            return None

        record = self._store_fn(
            content=content,
            tags=["working_memory"],
            source=f"agent:{self._agent}",
            namespace=self._agent,
            priority=0.2,
        )
        # Mark as volatile
        conn = self._db.connect()
        conn.execute(
            "UPDATE memories SET staleness = 'volatile', memory_type = 'observation' WHERE id = ?",
            (record.id,),
        )
        conn.commit()
        return record.id

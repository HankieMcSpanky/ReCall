"""Immutable audit trail for mutations and auth events."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from neuropack.storage.database import Database


class AuditLogger:
    """Logs all mutations and auth events to the audit_log table."""

    def __init__(self, db: Database):
        self._db = db

    def log(
        self,
        action: str,
        actor: str = "system",
        memory_id: str | None = None,
        namespace: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Write an audit log entry."""
        now = datetime.now(timezone.utc).isoformat()
        entry_id = uuid.uuid4().hex
        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO audit_log (id, timestamp, action, actor, memory_id, namespace, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id,
                    now,
                    action,
                    actor,
                    memory_id,
                    namespace,
                    json.dumps(details) if details else None,
                ),
            )

    def query(
        self,
        action: str | None = None,
        actor: str | None = None,
        memory_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log entries with optional filters."""
        conn = self._db.connect()
        conditions: list[str] = []
        params: list[object] = []

        if action:
            conditions.append("action = ?")
            params.append(action)
        if actor:
            conditions.append("actor = ?")
            params.append(actor)
        if memory_id:
            conditions.append("memory_id = ?")
            params.append(memory_id)

        query = "SELECT * FROM audit_log"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["details"] = json.loads(d["details"]) if d["details"] else None
            results.append(d)
        return results

    def count(self) -> int:
        """Total number of audit log entries."""
        conn = self._db.connect()
        row = conn.execute("SELECT COUNT(*) as cnt FROM audit_log").fetchone()
        return dict(row)["cnt"]

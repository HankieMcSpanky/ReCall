"""Shared Workspace for multi-agent collaboration.

Provides coordination primitives: workspaces, task board, structured handoffs,
decision log, and progressive context injection for late-joining agents.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from neuropack.audit import AuditLogger
from neuropack.core.tokens import estimate_tokens
from neuropack.exceptions import TaskClaimError, WorkspaceError
from neuropack.storage.database import Database
from neuropack.types import (
    Decision,
    Handoff,
    Workspace,
    WorkspaceMember,
    WorkspaceTask,
)


class WorkspaceManager:
    """Coordinates multi-agent collaboration via shared workspaces."""

    def __init__(
        self,
        db: Database,
        audit: AuditLogger,
        absorb_fn: Callable[..., None] | None = None,
    ):
        self._db = db
        self._audit = audit
        self._absorb_fn = absorb_fn

    def _ws_namespace(self, workspace_id: str) -> str:
        return f"workspace.{workspace_id}"

    # ---- Workspace lifecycle ----

    def create_workspace(
        self, name: str, goal: str, created_by: str = "system"
    ) -> Workspace:
        """Create a new workspace and auto-join the creator as owner."""
        ws_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO workspaces (id, name, goal, status, created_at, updated_at, created_by)
                   VALUES (?, ?, ?, 'active', ?, ?, ?)""",
                (ws_id, name, goal, now_iso, now_iso, created_by),
            )
            # Auto-join creator as owner
            conn.execute(
                """INSERT INTO workspace_members (workspace_id, agent_name, role, joined_at, last_seen)
                   VALUES (?, ?, 'owner', ?, ?)""",
                (ws_id, created_by, now_iso, now_iso),
            )

        self._audit.log(
            "workspace_create", actor=created_by,
            namespace=self._ws_namespace(ws_id),
            details={"name": name, "goal": goal},
        )

        return Workspace(
            id=ws_id, name=name, goal=goal, status="active",
            created_at=now, updated_at=now, created_by=created_by,
        )

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        return Workspace(
            id=d["id"], name=d["name"], goal=d["goal"], status=d["status"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            created_by=d["created_by"],
        )

    def list_workspaces(self, status: str | None = None) -> list[Workspace]:
        conn = self._db.connect()
        if status:
            rows = conn.execute(
                "SELECT * FROM workspaces WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM workspaces ORDER BY created_at DESC"
            ).fetchall()
        return [
            Workspace(
                id=dict(r)["id"], name=dict(r)["name"], goal=dict(r)["goal"],
                status=dict(r)["status"],
                created_at=datetime.fromisoformat(dict(r)["created_at"]),
                updated_at=datetime.fromisoformat(dict(r)["updated_at"]),
                created_by=dict(r)["created_by"],
            )
            for r in rows
        ]

    def close_workspace(self, workspace_id: str, agent_name: str) -> Workspace:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE workspaces SET status = 'completed', updated_at = ? WHERE id = ?",
                (now_iso, workspace_id),
            )
        self._audit.log(
            "workspace_close", actor=agent_name,
            namespace=self._ws_namespace(workspace_id),
        )
        ws = self.get_workspace(workspace_id)
        if ws is None:
            raise WorkspaceError(f"Workspace {workspace_id} not found")
        return ws

    # ---- Membership ----

    def join_workspace(self, workspace_id: str, agent_name: str) -> WorkspaceMember:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        ws = self.get_workspace(workspace_id)
        if ws is None:
            raise WorkspaceError(f"Workspace {workspace_id} not found")

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO workspace_members
                   (workspace_id, agent_name, role, joined_at, last_seen)
                   VALUES (?, ?, 'member', ?, ?)""",
                (workspace_id, agent_name, now_iso, now_iso),
            )

        self._audit.log(
            "workspace_join", actor=agent_name,
            namespace=self._ws_namespace(workspace_id),
        )

        return WorkspaceMember(
            workspace_id=workspace_id, agent_name=agent_name,
            role="member", joined_at=now, last_seen=now,
        )

    def leave_workspace(self, workspace_id: str, agent_name: str) -> bool:
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM workspace_members WHERE workspace_id = ? AND agent_name = ?",
                (workspace_id, agent_name),
            )
            deleted = cursor.rowcount > 0

        if deleted:
            self._audit.log(
                "workspace_leave", actor=agent_name,
                namespace=self._ws_namespace(workspace_id),
            )
        return deleted

    def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT * FROM workspace_members WHERE workspace_id = ? ORDER BY joined_at",
            (workspace_id,),
        ).fetchall()
        return [
            WorkspaceMember(
                workspace_id=dict(r)["workspace_id"],
                agent_name=dict(r)["agent_name"],
                role=dict(r)["role"],
                joined_at=datetime.fromisoformat(dict(r)["joined_at"]),
                last_seen=datetime.fromisoformat(dict(r)["last_seen"]),
            )
            for r in rows
        ]

    def heartbeat(self, workspace_id: str, agent_name: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_members SET last_seen = ? WHERE workspace_id = ? AND agent_name = ?",
                (now_iso, workspace_id, agent_name),
            )

    # ---- Task Board ----

    def create_task(
        self,
        workspace_id: str,
        title: str,
        description: str = "",
        created_by: str = "system",
    ) -> WorkspaceTask:
        task_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO workspace_tasks
                   (id, workspace_id, title, description, status, created_by,
                    blocked_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'open', ?, '[]', ?, ?)""",
                (task_id, workspace_id, title, description, created_by, now_iso, now_iso),
            )

        self._audit.log(
            "task_create", actor=created_by,
            namespace=self._ws_namespace(workspace_id),
            details={"task_id": task_id, "title": title},
        )

        return WorkspaceTask(
            id=task_id, workspace_id=workspace_id, title=title,
            description=description, status="open", created_by=created_by,
            assigned_to=None, blocked_by=[], created_at=now, updated_at=now,
        )

    def claim_task(self, task_id: str, agent_name: str) -> WorkspaceTask:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM workspace_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise WorkspaceError(f"Task {task_id} not found")

            d = dict(row)
            if d["status"] == "claimed" and d["assigned_to"] != agent_name:
                raise TaskClaimError(task_id, d["assigned_to"])
            if d["status"] == "done":
                raise WorkspaceError(f"Task {task_id} is already done")

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE workspace_tasks SET status = 'claimed', assigned_to = ?, updated_at = ? WHERE id = ?",
                (agent_name, now_iso, task_id),
            )

        self._audit.log(
            "task_claim", actor=agent_name,
            namespace=self._ws_namespace(d["workspace_id"]),
            details={"task_id": task_id, "title": d["title"]},
        )

        return self.get_task(task_id)  # type: ignore

    def complete_task(self, task_id: str, agent_name: str) -> WorkspaceTask:
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM workspace_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise WorkspaceError(f"Task {task_id} not found")

            d = dict(row)
            conn.execute(
                "UPDATE workspace_tasks SET status = 'done', completed_at = ?, updated_at = ? WHERE id = ?",
                (now_iso, now_iso, task_id),
            )

            # Auto-unblock tasks that were blocked by this one
            blocked_rows = conn.execute(
                "SELECT id, blocked_by FROM workspace_tasks WHERE workspace_id = ? AND status = 'blocked'",
                (d["workspace_id"],),
            ).fetchall()
            for br in blocked_rows:
                bd = dict(br)
                blocked_list = json.loads(bd["blocked_by"])
                if task_id in blocked_list:
                    blocked_list.remove(task_id)
                    if not blocked_list:
                        conn.execute(
                            "UPDATE workspace_tasks SET status = 'open', blocked_by = '[]', updated_at = ? WHERE id = ?",
                            (now_iso, bd["id"]),
                        )
                    else:
                        conn.execute(
                            "UPDATE workspace_tasks SET blocked_by = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(blocked_list), now_iso, bd["id"]),
                        )

        self._audit.log(
            "task_complete", actor=agent_name,
            namespace=self._ws_namespace(d["workspace_id"]),
            details={"task_id": task_id, "title": d["title"]},
        )

        # Auto-absorb task completion into agent's personal namespace
        if self._absorb_fn:
            absorb_text = f"[Task completed] {d['title']}"
            if d.get("description"):
                absorb_text += f": {d['description']}"
            self._absorb_fn(
                agent_name=agent_name, content=absorb_text,
                tags=["workspace_learning", "task_complete"],
                source=f"workspace:{d['workspace_id']}",
            )

        return self.get_task(task_id)  # type: ignore

    def block_task(
        self, task_id: str, blocked_by_task_ids: list[str], agent_name: str
    ) -> WorkspaceTask:
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM workspace_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise WorkspaceError(f"Task {task_id} not found")

            conn.execute(
                "UPDATE workspace_tasks SET status = 'blocked', blocked_by = ?, updated_at = ? WHERE id = ?",
                (json.dumps(blocked_by_task_ids), now_iso, task_id),
            )

        self._audit.log(
            "task_block", actor=agent_name,
            namespace=self._ws_namespace(dict(row)["workspace_id"]),
            details={"task_id": task_id, "blocked_by": blocked_by_task_ids},
        )

        return self.get_task(task_id)  # type: ignore

    def list_tasks(
        self,
        workspace_id: str,
        status: str | None = None,
        assigned_to: str | None = None,
    ) -> list[WorkspaceTask]:
        conn = self._db.connect()
        query = "SELECT * FROM workspace_tasks WHERE workspace_id = ?"
        params: list[object] = [workspace_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if assigned_to:
            query += " AND assigned_to = ?"
            params.append(assigned_to)
        query += " ORDER BY created_at"
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_task(dict(r)) for r in rows]

    def get_task(self, task_id: str) -> WorkspaceTask | None:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT * FROM workspace_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(dict(row))

    def _row_to_task(self, d: dict) -> WorkspaceTask:
        return WorkspaceTask(
            id=d["id"], workspace_id=d["workspace_id"],
            title=d["title"], description=d["description"],
            status=d["status"], created_by=d["created_by"],
            assigned_to=d["assigned_to"],
            blocked_by=json.loads(d["blocked_by"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d["completed_at"] else None,
        )

    # ---- Handoffs ----

    def post_handoff(
        self,
        workspace_id: str,
        from_agent: str,
        summary: str,
        context: dict,
        memory_ids: list[str] | None = None,
        to_agent: str | None = None,
        task_id: str | None = None,
    ) -> Handoff:
        handoff_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO workspace_handoffs
                   (id, workspace_id, from_agent, to_agent, task_id, summary, context, memory_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    handoff_id, workspace_id, from_agent, to_agent, task_id,
                    summary, json.dumps(context), json.dumps(memory_ids or []), now_iso,
                ),
            )

        self._audit.log(
            "handoff_post", actor=from_agent,
            namespace=self._ws_namespace(workspace_id),
            details={"handoff_id": handoff_id, "to_agent": to_agent, "task_id": task_id},
        )

        # Auto-absorb handoff learnings into agent's personal namespace
        if self._absorb_fn:
            absorb_text = f"[Workspace handoff] {summary}"
            if context.get("findings"):
                absorb_text += f"\nFindings: {', '.join(str(f) for f in context['findings'])}"
            if context.get("decisions"):
                absorb_text += f"\nDecisions: {', '.join(str(d) for d in context['decisions'])}"
            self._absorb_fn(
                agent_name=from_agent, content=absorb_text,
                tags=["workspace_learning", "handoff"],
                source=f"workspace:{workspace_id}",
            )

        return Handoff(
            id=handoff_id, workspace_id=workspace_id, from_agent=from_agent,
            to_agent=to_agent, task_id=task_id, summary=summary,
            context=context, memory_ids=memory_ids or [], created_at=now,
        )

    def get_handoffs(
        self,
        workspace_id: str,
        for_agent: str | None = None,
        limit: int = 20,
    ) -> list[Handoff]:
        conn = self._db.connect()
        if for_agent:
            rows = conn.execute(
                """SELECT * FROM workspace_handoffs
                   WHERE workspace_id = ? AND (to_agent IS NULL OR to_agent = ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, for_agent, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM workspace_handoffs
                   WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, limit),
            ).fetchall()
        return [self._row_to_handoff(dict(r)) for r in rows]

    def _row_to_handoff(self, d: dict) -> Handoff:
        return Handoff(
            id=d["id"], workspace_id=d["workspace_id"],
            from_agent=d["from_agent"], to_agent=d["to_agent"],
            task_id=d["task_id"], summary=d["summary"],
            context=json.loads(d["context"]),
            memory_ids=json.loads(d["memory_ids"]),
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    # ---- Decisions ----

    def log_decision(
        self,
        workspace_id: str,
        title: str,
        rationale: str,
        decided_by: str,
        alternatives: list[str] | None = None,
        related_task_id: str | None = None,
    ) -> Decision:
        dec_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO workspace_decisions
                   (id, workspace_id, title, rationale, decided_by, alternatives, related_task_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dec_id, workspace_id, title, rationale, decided_by,
                    json.dumps(alternatives or []), related_task_id, now_iso,
                ),
            )

        self._audit.log(
            "decision_log", actor=decided_by,
            namespace=self._ws_namespace(workspace_id),
            details={"decision_id": dec_id, "title": title},
        )

        # Auto-absorb decision into agent's personal namespace
        if self._absorb_fn:
            self._absorb_fn(
                agent_name=decided_by,
                content=f"[Decision] {title}: {rationale}",
                tags=["workspace_learning", "decision"],
                source=f"workspace:{workspace_id}",
            )

        return Decision(
            id=dec_id, workspace_id=workspace_id, title=title,
            rationale=rationale, decided_by=decided_by,
            alternatives=alternatives or [], related_task_id=related_task_id,
            created_at=now,
        )

    def get_decisions(self, workspace_id: str, limit: int = 50) -> list[Decision]:
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT * FROM workspace_decisions WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
        return [
            Decision(
                id=dict(r)["id"], workspace_id=dict(r)["workspace_id"],
                title=dict(r)["title"], rationale=dict(r)["rationale"],
                decided_by=dict(r)["decided_by"],
                alternatives=json.loads(dict(r)["alternatives"]),
                related_task_id=dict(r)["related_task_id"],
                created_at=datetime.fromisoformat(dict(r)["created_at"]),
            )
            for r in rows
        ]

    # ---- Activity Feed ----

    def activity_feed(self, workspace_id: str, limit: int = 50) -> list[dict]:
        """Get activity feed from existing audit_log."""
        ns = self._ws_namespace(workspace_id)
        conn = self._db.connect()
        rows = conn.execute(
            """SELECT * FROM audit_log WHERE namespace = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (ns, limit),
        ).fetchall()
        return [
            {
                "id": dict(r)["id"],
                "timestamp": dict(r)["timestamp"],
                "action": dict(r)["action"],
                "actor": dict(r)["actor"],
                "details": json.loads(dict(r)["details"]) if dict(r)["details"] else None,
            }
            for r in rows
        ]

    # ---- Context Injection ----

    def get_catchup_context(
        self,
        workspace_id: str,
        agent_name: str,
        token_budget: int = 4000,
    ) -> dict:
        """Progressive context injection for a late-joining agent.

        Returns L3 overview (always), L2 details (if budget allows),
        and memory IDs for full content (remaining budget).
        """
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            raise WorkspaceError(f"Workspace {workspace_id} not found")

        members = self.list_members(workspace_id)
        tasks = self.list_tasks(workspace_id)
        decisions = self.get_decisions(workspace_id, limit=20)

        # L3: always included
        l3 = {
            "workspace": {
                "name": workspace.name,
                "goal": workspace.goal,
                "status": workspace.status,
                "members": [m.agent_name for m in members],
            },
            "task_summary": {
                "total": len(tasks),
                "open": sum(1 for t in tasks if t.status == "open"),
                "claimed": sum(1 for t in tasks if t.status == "claimed"),
                "done": sum(1 for t in tasks if t.status == "done"),
                "blocked": sum(1 for t in tasks if t.status == "blocked"),
                "tasks": [
                    {"id": t.id, "title": t.title, "status": t.status,
                     "assigned_to": t.assigned_to}
                    for t in tasks
                ],
            },
            "decision_titles": [d.title for d in decisions],
        }

        used_tokens = estimate_tokens(json.dumps(l3))
        remaining = token_budget - used_tokens

        # L2: handoff summaries + decision rationales
        l2 = None
        if remaining > 200:
            handoffs = self.get_handoffs(workspace_id, limit=10)
            l2 = {
                "handoffs": [
                    {"from": h.from_agent, "summary": h.summary,
                     "open_questions": h.context.get("open_questions", [])}
                    for h in handoffs
                ],
                "decisions": [
                    {"title": d.title, "rationale": d.rationale,
                     "decided_by": d.decided_by}
                    for d in decisions[:10]
                ],
            }
            l2_tokens = estimate_tokens(json.dumps(l2))
            if l2_tokens <= remaining:
                remaining -= l2_tokens
            else:
                # Trim to fit
                l2 = {
                    "handoffs": [
                        {"from": h.from_agent, "summary": h.summary}
                        for h in handoffs[:5]
                    ],
                }
                remaining -= estimate_tokens(json.dumps(l2))

        # Full: memory IDs from handoffs
        memory_ids_for_full: list[str] = []
        if remaining > 100:
            handoffs = self.get_handoffs(workspace_id, limit=50)
            all_ids: set[str] = set()
            for h in handoffs:
                all_ids.update(h.memory_ids)
            memory_ids_for_full = list(all_ids)

        # Update heartbeat
        self.heartbeat(workspace_id, agent_name)

        return {
            "token_budget": token_budget,
            "tokens_used": token_budget - remaining,
            "l3_overview": l3,
            "l2_details": l2,
            "memory_ids_for_full_context": memory_ids_for_full,
        }

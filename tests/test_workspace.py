"""Tests for shared workspace collaboration feature."""
from __future__ import annotations

import json

import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.core.workspace import WorkspaceManager
from neuropack.exceptions import TaskClaimError, WorkspaceError
from neuropack.types import (
    Decision,
    Handoff,
    Workspace,
    WorkspaceMember,
    WorkspaceTask,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    config = NeuropackConfig(db_path=db_path, auto_tag=False, contradiction_check=False)
    s = MemoryStore(config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def ws(store):
    """Fixture that returns a workspace manager."""
    return store.workspace


@pytest.fixture
def workspace_id(ws):
    """Fixture that creates a workspace and returns its ID."""
    w = ws.create_workspace("Test Project", "Build something cool", created_by="alice")
    return w.id


# --- Workspace Lifecycle ---


class TestWorkspaceLifecycle:
    def test_create_workspace(self, ws):
        w = ws.create_workspace("My Project", "Solve a problem", created_by="alice")
        assert isinstance(w, Workspace)
        assert w.name == "My Project"
        assert w.goal == "Solve a problem"
        assert w.status == "active"
        assert w.created_by == "alice"

    def test_get_workspace(self, ws, workspace_id):
        w = ws.get_workspace(workspace_id)
        assert w is not None
        assert w.name == "Test Project"

    def test_get_nonexistent_workspace(self, ws):
        assert ws.get_workspace("nonexistent") is None

    def test_list_workspaces(self, ws, workspace_id):
        ws.create_workspace("Second", "Another goal")
        workspaces = ws.list_workspaces()
        assert len(workspaces) >= 2

    def test_list_workspaces_by_status(self, ws, workspace_id):
        active = ws.list_workspaces(status="active")
        assert len(active) >= 1
        completed = ws.list_workspaces(status="completed")
        assert len(completed) == 0

    def test_close_workspace(self, ws, workspace_id):
        w = ws.close_workspace(workspace_id, "alice")
        assert w.status == "completed"

    def test_close_nonexistent(self, ws):
        with pytest.raises(WorkspaceError):
            ws.close_workspace("nonexistent", "alice")


# --- Membership ---


class TestMembership:
    def test_creator_auto_joins(self, ws, workspace_id):
        members = ws.list_members(workspace_id)
        assert len(members) == 1
        assert members[0].agent_name == "alice"
        assert members[0].role == "owner"

    def test_join_workspace(self, ws, workspace_id):
        member = ws.join_workspace(workspace_id, "bob")
        assert isinstance(member, WorkspaceMember)
        assert member.agent_name == "bob"
        assert member.role == "member"

        members = ws.list_members(workspace_id)
        assert len(members) == 2

    def test_join_nonexistent(self, ws):
        with pytest.raises(WorkspaceError):
            ws.join_workspace("nonexistent", "bob")

    def test_leave_workspace(self, ws, workspace_id):
        ws.join_workspace(workspace_id, "bob")
        assert ws.leave_workspace(workspace_id, "bob")
        members = ws.list_members(workspace_id)
        names = [m.agent_name for m in members]
        assert "bob" not in names

    def test_leave_not_member(self, ws, workspace_id):
        assert not ws.leave_workspace(workspace_id, "nobody")

    def test_heartbeat(self, ws, workspace_id):
        # Should not raise
        ws.heartbeat(workspace_id, "alice")


# --- Task Board ---


class TestTaskBoard:
    def test_create_task(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research auth options", "Look into OAuth2 vs JWT")
        assert isinstance(task, WorkspaceTask)
        assert task.title == "Research auth options"
        assert task.status == "open"
        assert task.assigned_to is None

    def test_list_tasks(self, ws, workspace_id):
        ws.create_task(workspace_id, "Task 1")
        ws.create_task(workspace_id, "Task 2")
        tasks = ws.list_tasks(workspace_id)
        assert len(tasks) == 2

    def test_list_tasks_by_status(self, ws, workspace_id):
        t1 = ws.create_task(workspace_id, "Task 1")
        ws.create_task(workspace_id, "Task 2")
        ws.claim_task(t1.id, "bob")

        open_tasks = ws.list_tasks(workspace_id, status="open")
        assert len(open_tasks) == 1
        claimed = ws.list_tasks(workspace_id, status="claimed")
        assert len(claimed) == 1

    def test_claim_task(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        claimed = ws.claim_task(task.id, "bob")
        assert claimed.status == "claimed"
        assert claimed.assigned_to == "bob"

    def test_claim_already_claimed_by_other(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        ws.claim_task(task.id, "bob")
        with pytest.raises(TaskClaimError):
            ws.claim_task(task.id, "charlie")

    def test_claim_own_task_ok(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        ws.claim_task(task.id, "bob")
        # Re-claiming your own task should succeed
        reclaimed = ws.claim_task(task.id, "bob")
        assert reclaimed.assigned_to == "bob"

    def test_claim_done_task_fails(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        ws.claim_task(task.id, "bob")
        ws.complete_task(task.id, "bob")
        with pytest.raises(WorkspaceError):
            ws.claim_task(task.id, "charlie")

    def test_complete_task(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        ws.claim_task(task.id, "bob")
        done = ws.complete_task(task.id, "bob")
        assert done.status == "done"
        assert done.completed_at is not None

    def test_complete_unblocks_dependents(self, ws, workspace_id):
        t1 = ws.create_task(workspace_id, "Prerequisite")
        t2 = ws.create_task(workspace_id, "Dependent")

        # Block t2 on t1
        ws.block_task(t2.id, [t1.id], "alice")
        blocked = ws.get_task(t2.id)
        assert blocked.status == "blocked"

        # Complete t1 -- should unblock t2
        ws.claim_task(t1.id, "alice")
        ws.complete_task(t1.id, "alice")
        unblocked = ws.get_task(t2.id)
        assert unblocked.status == "open"

    def test_block_task(self, ws, workspace_id):
        t1 = ws.create_task(workspace_id, "Blocker")
        t2 = ws.create_task(workspace_id, "Blocked")
        blocked = ws.block_task(t2.id, [t1.id], "alice")
        assert blocked.status == "blocked"
        assert t1.id in blocked.blocked_by

    def test_get_task(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        fetched = ws.get_task(task.id)
        assert fetched is not None
        assert fetched.title == "Research"

    def test_get_nonexistent_task(self, ws):
        assert ws.get_task("nonexistent") is None


# --- Handoffs ---


class TestHandoffs:
    def test_post_handoff(self, ws, workspace_id):
        handoff = ws.post_handoff(
            workspace_id, "alice",
            summary="Finished auth research",
            context={
                "findings": ["OAuth2 is best for SPAs"],
                "decisions": ["Use PKCE flow"],
                "open_questions": ["Which provider?"],
            },
            memory_ids=["mem1", "mem2"],
        )
        assert isinstance(handoff, Handoff)
        assert handoff.from_agent == "alice"
        assert "OAuth2 is best" in handoff.context["findings"][0]
        assert len(handoff.memory_ids) == 2

    def test_get_handoffs(self, ws, workspace_id):
        ws.post_handoff(workspace_id, "alice", "Summary 1", {})
        ws.post_handoff(workspace_id, "bob", "Summary 2", {})
        handoffs = ws.get_handoffs(workspace_id)
        assert len(handoffs) == 2

    def test_get_handoffs_for_agent(self, ws, workspace_id):
        ws.post_handoff(workspace_id, "alice", "For bob", {}, to_agent="bob")
        ws.post_handoff(workspace_id, "alice", "For anyone", {})
        ws.post_handoff(workspace_id, "alice", "For charlie", {}, to_agent="charlie")

        bob_handoffs = ws.get_handoffs(workspace_id, for_agent="bob")
        # Should get "For bob" + "For anyone" (to_agent IS NULL)
        assert len(bob_handoffs) == 2

    def test_handoff_with_task_link(self, ws, workspace_id):
        task = ws.create_task(workspace_id, "Research")
        handoff = ws.post_handoff(
            workspace_id, "alice", "Done researching",
            context={}, task_id=task.id,
        )
        assert handoff.task_id == task.id


# --- Decisions ---


class TestDecisions:
    def test_log_decision(self, ws, workspace_id):
        dec = ws.log_decision(
            workspace_id,
            title="Use PostgreSQL",
            rationale="Better for complex queries than SQLite",
            decided_by="alice",
            alternatives=["SQLite", "MySQL"],
        )
        assert isinstance(dec, Decision)
        assert dec.title == "Use PostgreSQL"
        assert len(dec.alternatives) == 2

    def test_get_decisions(self, ws, workspace_id):
        ws.log_decision(workspace_id, "Decision 1", "Reason 1", "alice")
        ws.log_decision(workspace_id, "Decision 2", "Reason 2", "bob")
        decisions = ws.get_decisions(workspace_id)
        assert len(decisions) == 2


# --- Activity Feed ---


class TestActivityFeed:
    def test_activity_feed(self, ws, workspace_id):
        # Creating workspace already logs an event
        ws.join_workspace(workspace_id, "bob")
        ws.create_task(workspace_id, "Task 1", created_by="bob")

        feed = ws.activity_feed(workspace_id)
        assert len(feed) >= 3  # workspace_create + workspace_join + task_create
        actions = [e["action"] for e in feed]
        assert "workspace_create" in actions
        assert "workspace_join" in actions
        assert "task_create" in actions


# --- Context Injection ---


class TestContextInjection:
    def test_catchup_context_basic(self, ws, workspace_id):
        ws.join_workspace(workspace_id, "bob")
        ws.create_task(workspace_id, "Task 1", created_by="alice")
        ws.log_decision(workspace_id, "Use Python", "Best for ML", "alice")

        ctx = ws.get_catchup_context(workspace_id, "bob")
        assert "l3_overview" in ctx
        assert ctx["l3_overview"]["workspace"]["name"] == "Test Project"
        assert ctx["l3_overview"]["task_summary"]["total"] == 1
        assert "Use Python" in ctx["l3_overview"]["decision_titles"]

    def test_catchup_with_handoffs(self, ws, workspace_id):
        ws.post_handoff(workspace_id, "alice", "Did research", {
            "findings": ["Found X"],
            "open_questions": ["What about Y?"],
        })

        ctx = ws.get_catchup_context(workspace_id, "bob", token_budget=4000)
        assert ctx["l2_details"] is not None
        assert len(ctx["l2_details"]["handoffs"]) == 1

    def test_catchup_small_budget(self, ws, workspace_id):
        # With tiny budget, should still include L3
        ctx = ws.get_catchup_context(workspace_id, "alice", token_budget=50)
        assert "l3_overview" in ctx


# --- Store Integration ---


class TestStoreIntegration:
    def test_workspace_property(self, store):
        assert store.workspace is not None

    def test_workspace_catchup_resolves_memories(self, store):
        ws = store.workspace.create_workspace("Test", "Goal", created_by="alice")

        # Store a memory and reference it in a handoff
        mem = store.store("Detailed research findings", tags=["workspace"])
        store.workspace.post_handoff(
            ws.id, "alice", "Research done",
            context={"findings": ["Found something"]},
            memory_ids=[mem.id],
        )

        ctx = store.workspace_catchup(ws.id, "bob", token_budget=4000)
        assert "full_memories" in ctx
        # The memory should be resolved
        if ctx["full_memories"]:
            assert any(m["id"] == mem.id for m in ctx["full_memories"])

    def test_full_collaboration_flow(self, store):
        """End-to-end: create workspace, add tasks, claim, handoff, catchup."""
        ws_mgr = store.workspace

        # 1. Alice creates workspace
        ws = ws_mgr.create_workspace("Auth System", "Build user authentication", "alice")

        # 2. Alice creates tasks
        t1 = ws_mgr.create_task(ws.id, "Research auth options", created_by="alice")
        t2 = ws_mgr.create_task(ws.id, "Implement chosen auth", created_by="alice")
        ws_mgr.block_task(t2.id, [t1.id], "alice")

        # 3. Alice claims and works on research
        ws_mgr.claim_task(t1.id, "alice")
        mem = store.store("OAuth2 with PKCE is best for SPAs", tags=["auth"])

        # 4. Alice posts handoff
        ws_mgr.post_handoff(
            ws.id, "alice",
            summary="OAuth2 PKCE is the way to go",
            context={
                "findings": ["PKCE prevents auth code interception"],
                "decisions": ["Use OAuth2 PKCE"],
                "open_questions": ["Which identity provider?"],
            },
            memory_ids=[mem.id],
            task_id=t1.id,
        )

        # 5. Alice logs decision
        ws_mgr.log_decision(
            ws.id, "Use OAuth2 with PKCE",
            "Prevents auth code interception, works in browser",
            "alice", alternatives=["Basic auth", "API keys"],
        )

        # 6. Alice completes task (should unblock t2)
        ws_mgr.complete_task(t1.id, "alice")
        t2_refreshed = ws_mgr.get_task(t2.id)
        assert t2_refreshed.status == "open"

        # 7. Bob joins and gets caught up
        ws_mgr.join_workspace(ws.id, "bob")
        ctx = store.workspace_catchup(ws.id, "bob", token_budget=4000)

        assert ctx["l3_overview"]["workspace"]["goal"] == "Build user authentication"
        assert ctx["l3_overview"]["task_summary"]["done"] == 1
        assert ctx["l3_overview"]["task_summary"]["open"] == 1
        assert "Use OAuth2 with PKCE" in ctx["l3_overview"]["decision_titles"]

        # 8. Bob can see handoffs with open questions
        assert ctx["l2_details"] is not None
        handoffs = ctx["l2_details"]["handoffs"]
        assert len(handoffs) >= 1
        assert "Which identity provider?" in handoffs[0]["open_questions"]

        # 9. Bob claims the unblocked task
        ws_mgr.claim_task(t2.id, "bob")
        t2_claimed = ws_mgr.get_task(t2.id)
        assert t2_claimed.assigned_to == "bob"

        # 10. Activity feed shows the full history
        feed = ws_mgr.activity_feed(ws.id)
        actions = [e["action"] for e in feed]
        assert "workspace_create" in actions
        assert "task_create" in actions
        assert "task_claim" in actions
        assert "handoff_post" in actions
        assert "decision_log" in actions
        assert "task_complete" in actions
        assert "workspace_join" in actions


# --- Auto-Absorb: Agent ↔ Workspace Bridge ---


class TestAutoAbsorb:
    """Test that workspace actions auto-absorb learnings into agent namespaces."""

    def test_handoff_absorbs_to_agent(self, store):
        ws_mgr = store.workspace
        ws = ws_mgr.create_workspace("Project", "Goal", created_by="alice")

        ws_mgr.post_handoff(
            ws.id, "alice", "Found the root cause",
            context={"findings": ["Bug in auth module"], "decisions": ["Fix with patch"]},
        )

        # Alice's namespace should now have the absorbed learning
        alice_mems = store.list(limit=100, namespace="alice")
        ws_mems = [m for m in alice_mems if "workspace_learning" in (m.tags or [])]
        assert len(ws_mems) == 1
        assert "Found the root cause" in ws_mems[0].content
        assert "Bug in auth module" in ws_mems[0].content
        assert "handoff" in ws_mems[0].tags

    def test_task_complete_absorbs_to_agent(self, store):
        ws_mgr = store.workspace
        ws = ws_mgr.create_workspace("Project", "Goal", created_by="bob")
        task = ws_mgr.create_task(ws.id, "Implement caching", "Add Redis layer")
        ws_mgr.claim_task(task.id, "bob")
        ws_mgr.complete_task(task.id, "bob")

        bob_mems = store.list(limit=100, namespace="bob")
        ws_mems = [m for m in bob_mems if "task_complete" in (m.tags or [])]
        assert len(ws_mems) == 1
        assert "Implement caching" in ws_mems[0].content

    def test_decision_absorbs_to_agent(self, store):
        ws_mgr = store.workspace
        ws = ws_mgr.create_workspace("Project", "Goal", created_by="charlie")

        ws_mgr.log_decision(
            ws.id, "Use PostgreSQL", "Better for complex queries",
            decided_by="charlie", alternatives=["MySQL", "SQLite"],
        )

        charlie_mems = store.list(limit=100, namespace="charlie")
        ws_mems = [m for m in charlie_mems if "decision" in (m.tags or [])]
        assert len(ws_mems) == 1
        assert "Use PostgreSQL" in ws_mems[0].content
        assert "Better for complex queries" in ws_mems[0].content

    def test_agent_recall_finds_workspace_learnings(self, store):
        ws_mgr = store.workspace
        ws = ws_mgr.create_workspace("Auth", "Build auth", created_by="alice")
        ws_mgr.post_handoff(
            ws.id, "alice", "OAuth2 PKCE is best",
            context={"findings": ["PKCE prevents interception"]},
        )

        results = store.agent_recall("alice", "OAuth2", limit=5)
        assert len(results) >= 1
        assert any("OAuth2" in r.record.content for r in results)

    def test_agent_expertise_includes_workspace_data(self, store):
        ws_mgr = store.workspace
        ws = ws_mgr.create_workspace("Auth", "Build auth", created_by="alice")
        task = ws_mgr.create_task(ws.id, "Research")
        ws_mgr.claim_task(task.id, "alice")
        ws_mgr.complete_task(task.id, "alice")
        ws_mgr.post_handoff(ws.id, "alice", "Done", context={})
        ws_mgr.log_decision(ws.id, "Use JWT", "Simple", decided_by="alice")

        profile = store.agent_expertise("alice")
        assert profile["agent"] == "alice"
        assert profile["workspace_learnings"] >= 3  # handoff + task + decision
        assert profile["tasks_completed"] >= 1
        assert profile["handoffs_posted"] >= 1
        assert profile["decisions_made"] >= 1
        assert len(profile["workspaces"]) >= 1
        assert profile["workspaces"][0]["name"] == "Auth"

    def test_auto_absorb_disabled(self, tmp_path):
        """When workspace_auto_absorb=False, no learnings are absorbed."""
        from neuropack.config import NeuropackConfig
        from neuropack.core.store import MemoryStore

        config = NeuropackConfig(
            db_path=str(tmp_path / "test.db"),
            auto_tag=False, contradiction_check=False,
            workspace_auto_absorb=False,
        )
        s = MemoryStore(config)
        s.initialize()
        try:
            ws = s.workspace.create_workspace("P", "G", created_by="dana")
            s.workspace.post_handoff(ws.id, "dana", "Summary", context={})

            dana_mems = s.list(limit=100, namespace="dana")
            ws_mems = [m for m in dana_mems if "workspace_learning" in (m.tags or [])]
            assert len(ws_mems) == 0
        finally:
            s.close()

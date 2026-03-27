"""CLI commands for shared workspace collaboration."""
from __future__ import annotations

import json

import click

from neuropack.core.store import MemoryStore


@click.group("workspace")
def workspace_group() -> None:
    """Shared workspace commands for multi-agent collaboration."""
    pass


@workspace_group.command("create")
@click.argument("name")
@click.option("--goal", "-g", default="", help="Workspace goal")
@click.option("--agent", "-a", default="cli-user", help="Creator agent name")
@click.pass_context
def ws_create(ctx: click.Context, name: str, goal: str, agent: str) -> None:
    """Create a new collaboration workspace."""
    ms: MemoryStore = ctx.obj["store"]
    ws = ms.workspace.create_workspace(name, goal, created_by=agent)
    click.echo(json.dumps({
        "id": ws.id, "name": ws.name, "goal": ws.goal, "status": ws.status,
    }, indent=2))


@workspace_group.command("list")
@click.option("--status", "-s", default=None,
              type=click.Choice(["active", "completed", "archived"]))
@click.pass_context
def ws_list(ctx: click.Context, status: str | None) -> None:
    """List workspaces."""
    ms: MemoryStore = ctx.obj["store"]
    workspaces = ms.workspace.list_workspaces(status=status)
    if not workspaces:
        click.echo("No workspaces found.")
        return
    for w in workspaces:
        click.echo(json.dumps({
            "id": w.id, "name": w.name, "goal": w.goal,
            "status": w.status, "created_by": w.created_by,
        }))


@workspace_group.command("join")
@click.argument("workspace_id")
@click.option("--agent", "-a", required=True, help="Agent name")
@click.pass_context
def ws_join(ctx: click.Context, workspace_id: str, agent: str) -> None:
    """Join a workspace and get caught up."""
    ms: MemoryStore = ctx.obj["store"]
    ms.workspace.join_workspace(workspace_id, agent)
    ctx_data = ms.workspace_catchup(workspace_id, agent, token_budget=2000)
    click.echo(f"Joined workspace {workspace_id}")
    click.echo(json.dumps(ctx_data["l3_overview"], indent=2))


@workspace_group.command("tasks")
@click.argument("workspace_id")
@click.option("--status", "-s", default=None,
              type=click.Choice(["open", "claimed", "blocked", "done"]))
@click.pass_context
def ws_tasks(ctx: click.Context, workspace_id: str, status: str | None) -> None:
    """List tasks on the workspace board."""
    ms: MemoryStore = ctx.obj["store"]
    tasks = ms.workspace.list_tasks(workspace_id, status=status)
    if not tasks:
        click.echo("No tasks.")
        return
    for t in tasks:
        click.echo(json.dumps({
            "id": t.id, "title": t.title, "status": t.status,
            "assigned_to": t.assigned_to,
        }))


@workspace_group.command("task-create")
@click.argument("workspace_id")
@click.argument("title")
@click.option("--desc", "-d", default="", help="Task description")
@click.option("--agent", "-a", default="cli-user")
@click.pass_context
def ws_task_create(
    ctx: click.Context, workspace_id: str, title: str, desc: str, agent: str
) -> None:
    """Create a task on the workspace board."""
    ms: MemoryStore = ctx.obj["store"]
    task = ms.workspace.create_task(workspace_id, title, desc, created_by=agent)
    click.echo(json.dumps({"id": task.id, "title": task.title, "status": task.status}, indent=2))


@workspace_group.command("task-claim")
@click.argument("task_id")
@click.option("--agent", "-a", required=True)
@click.pass_context
def ws_task_claim(ctx: click.Context, task_id: str, agent: str) -> None:
    """Claim a task (prevents others from working on it)."""
    ms: MemoryStore = ctx.obj["store"]
    task = ms.workspace.claim_task(task_id, agent)
    click.echo(f"Claimed: {task.title} (assigned to {task.assigned_to})")


@workspace_group.command("task-done")
@click.argument("task_id")
@click.option("--agent", "-a", required=True)
@click.pass_context
def ws_task_done(ctx: click.Context, task_id: str, agent: str) -> None:
    """Mark a task as done."""
    ms: MemoryStore = ctx.obj["store"]
    task = ms.workspace.complete_task(task_id, agent)
    click.echo(f"Completed: {task.title}")


@workspace_group.command("handoffs")
@click.argument("workspace_id")
@click.option("--agent", "-a", default=None, help="Filter to this agent")
@click.option("--limit", "-l", default=20, type=int)
@click.pass_context
def ws_handoffs(ctx: click.Context, workspace_id: str, agent: str | None, limit: int) -> None:
    """View handoff records."""
    ms: MemoryStore = ctx.obj["store"]
    handoffs = ms.workspace.get_handoffs(workspace_id, for_agent=agent, limit=limit)
    if not handoffs:
        click.echo("No handoffs.")
        return
    for h in handoffs:
        click.echo(json.dumps({
            "id": h.id,
            "from": h.from_agent,
            "to": h.to_agent,
            "summary": h.summary,
            "open_questions": h.context.get("open_questions", []),
            "created_at": h.created_at.isoformat(),
        }, indent=2))
        click.echo("---")


@workspace_group.command("decisions")
@click.argument("workspace_id")
@click.pass_context
def ws_decisions(ctx: click.Context, workspace_id: str) -> None:
    """View decision log."""
    ms: MemoryStore = ctx.obj["store"]
    decisions = ms.workspace.get_decisions(workspace_id)
    if not decisions:
        click.echo("No decisions logged.")
        return
    for d in decisions:
        click.echo(json.dumps({
            "title": d.title,
            "rationale": d.rationale,
            "decided_by": d.decided_by,
            "alternatives": d.alternatives,
            "created_at": d.created_at.isoformat(),
        }, indent=2))
        click.echo("---")


@workspace_group.command("activity")
@click.argument("workspace_id")
@click.option("--limit", "-l", default=30, type=int)
@click.pass_context
def ws_activity(ctx: click.Context, workspace_id: str, limit: int) -> None:
    """View workspace activity feed."""
    ms: MemoryStore = ctx.obj["store"]
    feed = ms.workspace.activity_feed(workspace_id, limit=limit)
    if not feed:
        click.echo("No activity.")
        return
    for entry in feed:
        click.echo(f"[{entry['timestamp']}] {entry['actor']}: {entry['action']}")
        if entry.get("details"):
            click.echo(f"  {json.dumps(entry['details'])}")


@workspace_group.command("catchup")
@click.argument("workspace_id")
@click.option("--agent", "-a", required=True)
@click.option("--budget", "-b", default=4000, type=int, help="Token budget")
@click.pass_context
def ws_catchup(ctx: click.Context, workspace_id: str, agent: str, budget: int) -> None:
    """Get progressive context for a workspace (L3 overview, L2 details, full memories)."""
    ms: MemoryStore = ctx.obj["store"]
    ctx_data = ms.workspace_catchup(workspace_id, agent, token_budget=budget)
    click.echo(json.dumps(ctx_data, indent=2))

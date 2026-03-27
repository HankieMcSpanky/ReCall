"""Multi-agent learning commands: np agent subgroup."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import click

from neuropack.core.store import MemoryStore

# Keywords for auto-tagging
_MISTAKE_KEYWORDS = {"mistake", "error", "loss", "lost", "failed", "failure", "wrong", "bad"}
_WIN_KEYWORDS = {"success", "profit", "win", "won", "gained", "gain", "correct", "good"}


def _auto_tag(content: str) -> str:
    """Detect win/mistake/observation from content."""
    lower = content.lower()
    words = set(lower.split())
    if words & _MISTAKE_KEYWORDS:
        return "mistake"
    if words & _WIN_KEYWORDS:
        return "win"
    return "observation"


@click.group("agent")
def agent_group() -> None:
    """Multi-agent learning commands."""
    pass


@agent_group.command("create")
@click.argument("name")
@click.pass_context
def agent_create(ctx: click.Context, name: str) -> None:
    """Create a new agent namespace."""
    ms: MemoryStore = ctx.obj["store"]
    meta_key = f"agent:{name}"
    existing = ms._repo.load_metadata(meta_key)
    if existing:
        click.echo(f"Agent '{name}' already exists.", err=True)
        return

    now = datetime.now(timezone.utc).isoformat()
    ms._repo.save_metadata(meta_key, json.dumps({"created_at": now}))
    click.echo(f"Created agent '{name}' (namespace: {name})")


@agent_group.command("list")
@click.pass_context
def agent_list(ctx: click.Context) -> None:
    """List all agents with memory counts."""
    ms: MemoryStore = ctx.obj["store"]
    conn = ms._db.connect()
    rows = conn.execute(
        "SELECT key, value FROM metadata WHERE key LIKE 'agent:%'"
    ).fetchall()

    if not rows:
        click.echo("No agents created. Run: np agent create <name>")
        return

    agents = []
    for row in rows:
        d = dict(row)
        name = d["key"].replace("agent:", "", 1)
        meta = json.loads(d["value"])
        count = ms._repo.count_by_namespace(name)
        agents.append({
            "name": name,
            "memories": count,
            "created_at": meta.get("created_at", ""),
        })

    click.echo(json.dumps(agents, indent=2))


@agent_group.command("log")
@click.argument("name")
@click.argument("text")
@click.pass_context
def agent_log(ctx: click.Context, name: str, text: str) -> None:
    """Log an observation for an agent (auto-tags win/mistake/observation)."""
    ms: MemoryStore = ctx.obj["store"]
    tag = _auto_tag(text)
    record = ms.store(
        content=text,
        tags=[tag],
        source=f"agent:{name}",
        namespace=name,
    )
    click.echo(json.dumps({
        "id": record.id,
        "agent": name,
        "tag": tag,
        "l3": record.l3_abstract,
    }, indent=2))


@agent_group.command("mistakes")
@click.argument("name")
@click.option("--limit", "-l", default=20, type=int)
@click.pass_context
def agent_mistakes(ctx: click.Context, name: str, limit: int) -> None:
    """Show memories tagged 'mistake' for an agent."""
    ms: MemoryStore = ctx.obj["store"]
    records = ms.list(limit=limit, tag="mistake", namespace=name)
    if not records:
        click.echo(f"No mistakes logged for agent '{name}'.")
        return
    for r in records:
        click.echo(json.dumps({
            "id": r.id,
            "content": r.content[:200],
            "created": r.created_at.isoformat(),
        }))


@agent_group.command("wins")
@click.argument("name")
@click.option("--limit", "-l", default=20, type=int)
@click.pass_context
def agent_wins(ctx: click.Context, name: str, limit: int) -> None:
    """Show memories tagged 'win' for an agent."""
    ms: MemoryStore = ctx.obj["store"]
    records = ms.list(limit=limit, tag="win", namespace=name)
    if not records:
        click.echo(f"No wins logged for agent '{name}'.")
        return
    for r in records:
        click.echo(json.dumps({
            "id": r.id,
            "content": r.content[:200],
            "created": r.created_at.isoformat(),
        }))


@agent_group.command("share")
@click.argument("name")
@click.argument("memory_id")
@click.pass_context
def agent_share(ctx: click.Context, name: str, memory_id: str) -> None:
    """Share a memory to the 'shared' namespace."""
    ms: MemoryStore = ctx.obj["store"]
    try:
        record = ms.share_memory(memory_id, "shared")
        click.echo(json.dumps({
            "shared_id": record.id,
            "from_agent": name,
            "namespace": "shared",
        }, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@agent_group.command("learn")
@click.argument("name")
@click.option("--limit", "-l", default=20, type=int)
@click.pass_context
def agent_learn(ctx: click.Context, name: str, limit: int) -> None:
    """Show shared memories the agent hasn't seen yet."""
    ms: MemoryStore = ctx.obj["store"]
    shared = ms.list(limit=limit, namespace="shared")
    # Filter out memories that originated from this agent
    learnings = [r for r in shared if r.source != f"agent:{name}"]
    if not learnings:
        click.echo(f"No new shared learnings for agent '{name}'.")
        return
    for r in learnings:
        click.echo(json.dumps({
            "id": r.id,
            "l3": r.l3_abstract,
            "tags": r.tags,
            "source": r.source,
        }))


@agent_group.command("scoreboard")
@click.pass_context
def agent_scoreboard(ctx: click.Context) -> None:
    """Rank agents by win/mistake ratio."""
    ms: MemoryStore = ctx.obj["store"]
    conn = ms._db.connect()
    rows = conn.execute(
        "SELECT key FROM metadata WHERE key LIKE 'agent:%'"
    ).fetchall()

    if not rows:
        click.echo("No agents created.")
        return

    board = []
    for row in rows:
        name = dict(row)["key"].replace("agent:", "", 1)
        wins = ms.list(limit=1000, tag="win", namespace=name)
        mistakes = ms.list(limit=1000, tag="mistake", namespace=name)
        total = len(wins) + len(mistakes)
        ratio = len(wins) / total if total > 0 else 0.0
        board.append({
            "agent": name,
            "wins": len(wins),
            "mistakes": len(mistakes),
            "total": ms._repo.count_by_namespace(name),
            "win_ratio": round(ratio, 2),
        })

    board.sort(key=lambda x: x["win_ratio"], reverse=True)
    click.echo(json.dumps(board, indent=2))


@agent_group.command("recall")
@click.argument("name")
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int)
@click.pass_context
def agent_recall(ctx: click.Context, name: str, query: str, limit: int) -> None:
    """Search an agent's memory including workspace learnings."""
    ms: MemoryStore = ctx.obj["store"]
    results = ms.agent_recall(name, query, limit=limit)
    if not results:
        click.echo(f"No results for agent '{name}'.")
        return
    for r in results:
        click.echo(json.dumps({
            "id": r.record.id,
            "l3": r.record.l3_abstract,
            "tags": r.record.tags,
            "source": r.record.source,
            "score": round(r.score, 3),
        }))


@agent_group.command("expertise")
@click.argument("name")
@click.pass_context
def agent_expertise(ctx: click.Context, name: str) -> None:
    """Show an agent's expertise profile: workspace history, wins, learnings."""
    ms: MemoryStore = ctx.obj["store"]
    profile = ms.agent_expertise(name)
    click.echo(json.dumps(profile, indent=2))


@agent_group.command("promote")
@click.argument("name")
@click.argument("memory_id")
@click.option("--priority", "-p", default=0.8, type=float)
@click.pass_context
def agent_promote(ctx: click.Context, name: str, memory_id: str, priority: float) -> None:
    """Promote a memory: boost priority and mark stable."""
    ms: MemoryStore = ctx.obj["store"]
    mgr = ms.agent_memory(name)
    ok = mgr.promote(memory_id, priority=priority)
    click.echo(json.dumps({"promoted": ok, "memory_id": memory_id}))


@agent_group.command("demote")
@click.argument("name")
@click.argument("memory_id")
@click.option("--priority", "-p", default=0.2, type=float)
@click.pass_context
def agent_demote(ctx: click.Context, name: str, memory_id: str, priority: float) -> None:
    """Demote a memory: lower priority and mark volatile."""
    ms: MemoryStore = ctx.obj["store"]
    mgr = ms.agent_memory(name)
    ok = mgr.demote(memory_id, priority=priority)
    click.echo(json.dumps({"demoted": ok, "memory_id": memory_id}))


@agent_group.command("archive")
@click.argument("name")
@click.argument("memory_id")
@click.option("--reason", "-r", default="archived")
@click.pass_context
def agent_archive(ctx: click.Context, name: str, memory_id: str, reason: str) -> None:
    """Archive a memory: save version, tag archived, lower priority."""
    ms: MemoryStore = ctx.obj["store"]
    mgr = ms.agent_memory(name)
    ok = mgr.archive(memory_id, reason=reason)
    click.echo(json.dumps({"archived": ok, "memory_id": memory_id}))


@agent_group.command("pin")
@click.argument("name")
@click.argument("memory_id")
@click.pass_context
def agent_pin(ctx: click.Context, name: str, memory_id: str) -> None:
    """Pin a memory: max priority, never auto-purge."""
    ms: MemoryStore = ctx.obj["store"]
    mgr = ms.agent_memory(name)
    ok = mgr.pin(memory_id)
    click.echo(json.dumps({"pinned": ok, "memory_id": memory_id}))

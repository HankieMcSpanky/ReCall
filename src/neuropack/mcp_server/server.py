from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from neuropack.config import NeuropackConfig
from neuropack.core.privacy import strip_private_from_preview
from neuropack.core.store import MemoryStore

mcp = FastMCP("ReCall Memory Server")

_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    assert _store is not None, "Store not initialized"
    return _store


@mcp.tool()
def remember(
    content: str,
    tags: list[str] | None = None,
    source: str = "",
    priority: float = 0.5,
    l3: str | None = None,
    l2: list[str] | None = None,
    namespace: str | None = None,
    memory_type: str | None = None,
    staleness: str | None = None,
) -> dict:
    """Store a new memory in ReCall.

    Args:
        content: The text content to remember
        tags: Optional categorization tags
        source: Where this memory came from
        priority: Importance from 0.0 to 1.0
        l3: Optional caller-provided one-line abstract
        l2: Optional caller-provided list of key facts
        namespace: Optional namespace (default: config namespace)
        memory_type: Type classification (decision, fact, preference, procedure, code, observation, general)
        staleness: Expected staleness (stable, semi-stable, volatile)
    """
    store = _get_store()
    record = store.store(
        content=content,
        tags=tags or [],
        source=source,
        priority=priority,
        l3_override=l3,
        l2_override=l2,
        namespace=namespace,
        memory_type=memory_type,
        staleness=staleness,
    )
    return {
        "id": record.id,
        "l3_abstract": record.l3_abstract,
        "memory_type": record.memory_type,
        "staleness": record.staleness,
        "namespace": record.namespace,
        "status": "stored",
    }


@mcp.tool()
def recall(
    query: str,
    limit: int = 10,
    tags: list[str] | None = None,
    namespace: str | None = None,
    token_budget: int | None = None,
) -> dict:
    """Search memories by semantic and keyword similarity.

    Args:
        query: Natural language search query
        limit: Maximum number of results (default 10)
        tags: Filter by these tags
        namespace: Search in specific namespace (default: searches current + shared)
        token_budget: Optional max tokens for results (returns fewer results to fit)
    """
    store = _get_store()
    namespaces = [namespace] if namespace else None
    results = store.recall(
        query=query, limit=limit, tags=tags,
        namespaces=namespaces, token_budget=token_budget,
    )
    return {
        "count": len(results),
        "results": [
            {
                "id": r.record.id,
                "l3_abstract": r.record.l3_abstract,
                "l2_facts": r.record.l2_facts,
                "content_preview": strip_private_from_preview(r.record.content[:300])[:200],
                "score": round(r.score, 4),
                "tags": r.record.tags,
                "memory_type": r.record.memory_type,
                "namespace": r.record.namespace,
                "staleness_warning": r.staleness_warning,
            }
            for r in results
        ],
    }


@mcp.tool()
def forget(memory_id: str) -> dict:
    """Delete a memory by its ID.

    Args:
        memory_id: The UUID of the memory to delete
    """
    store = _get_store()
    deleted = store.forget(memory_id)
    return {"deleted": deleted, "id": memory_id}


@mcp.tool()
def list_memories(
    limit: int = 20,
    offset: int = 0,
    tag: str | None = None,
    namespace: str | None = None,
) -> dict:
    """List stored memories with pagination.

    Args:
        limit: Number of memories to return
        offset: Number of memories to skip
        tag: Optional tag filter
        namespace: Optional namespace filter
    """
    store = _get_store()
    records = store.list(limit=limit, offset=offset, tag=tag, namespace=namespace)
    return {
        "count": len(records),
        "memories": [
            {
                "id": r.id,
                "l3_abstract": r.l3_abstract,
                "tags": r.tags,
                "priority": r.priority,
                "namespace": r.namespace,
            }
            for r in records
        ],
    }


@mcp.tool()
def memory_stats() -> dict:
    """Get statistics about the memory store including token savings."""
    store = _get_store()
    s = store.stats()
    return {
        "total_memories": s.total_memories,
        "total_size_bytes": s.total_size_bytes,
        "avg_compression_ratio": round(s.avg_compression_ratio, 2),
        "total_content_tokens": s.total_content_tokens,
        "total_compressed_tokens": s.total_compressed_tokens,
        "token_savings_ratio": s.token_savings_ratio,
    }


@mcp.tool()
def context_summary(
    limit: int = 50,
    tags: list[str] | None = None,
    namespace: str | None = None,
) -> dict:
    """Get a compact index of all memories -- IDs, one-line abstracts, and tags.

    Use this FIRST to see what memories are available, then call fetch_details
    for the ones you need full content from.

    Args:
        limit: Maximum number of memories to index
        tags: Optional tag filter
        namespace: Optional namespace filter
    """
    store = _get_store()
    summaries = store.context_summary(limit=limit, tags=tags, namespace=namespace)
    return {
        "count": len(summaries),
        "memories": summaries,
        "hint": "Call fetch_details with specific IDs to get full content",
    }


@mcp.tool()
def fetch_details(memory_ids: list[str]) -> dict:
    """Fetch full content and L2 facts for specific memories by ID.

    Use after context_summary to retrieve detailed information for selected
    memories. Avoids loading all memories into context.

    Args:
        memory_ids: List of memory IDs from context_summary results
    """
    store = _get_store()
    details = store.fetch_details(memory_ids)
    return {
        "count": len(details),
        "memories": details,
    }


@mcp.tool()
def session_summary(memory_ids: list[str], store_as_memory: bool = False) -> dict:
    """Generate a structured summary of the current session.

    Takes memory IDs created during the session and produces a structured
    summary with categories: investigated, learned, completed, next_steps.

    Args:
        memory_ids: List of memory IDs from this session
        store_as_memory: If True, also stores the summary as a memory with tag "session-summary"
    """
    store = _get_store()
    summary = store.session_summary(memory_ids)
    result = {"summary": summary}
    if store_as_memory:
        record = store.store_session_summary(memory_ids)
        result["stored_id"] = record.id
    return result


@mcp.tool()
def generate_context(
    limit: int = 50,
    tags: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """Generate a CLAUDE.md-style context file from recent memories.

    Creates a markdown summary with recent activity, key facts, and tag cloud.
    Optionally writes to a file path.

    Args:
        limit: Maximum memories to include
        tags: Optional tag filter
        output_path: If set, write the markdown to this file path
    """
    store = _get_store()
    markdown = store.generate_context(limit=limit, tags=tags)
    result: dict = {"markdown": markdown, "memory_count": limit}
    if output_path:
        from pathlib import Path

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        result["written_to"] = str(path)
    return result


# --- New tools: Namespaces ---


@mcp.tool()
def list_namespaces() -> dict:
    """List all namespaces with memory counts.

    Shows how many memories exist in each namespace.
    """
    store = _get_store()
    ns = store.list_namespaces()
    return {"namespaces": ns}


@mcp.tool()
def share_memory(memory_id: str, target_namespace: str) -> dict:
    """Copy a memory to another namespace for cross-agent sharing.

    Args:
        memory_id: The memory ID to share
        target_namespace: The namespace to copy the memory into
    """
    store = _get_store()
    record = store.share_memory(memory_id, target_namespace)
    return {
        "new_id": record.id,
        "namespace": record.namespace,
        "status": "shared",
    }


# --- New tools: Knowledge Graph ---


@mcp.tool()
def query_entity(name: str, as_of: str = "") -> dict:
    """Look up an entity and its relationships in the knowledge graph.

    Args:
        name: The entity name to look up
        as_of: Optional ISO timestamp to filter facts valid at that point in time
    """
    store = _get_store()
    return store.query_entity(name, as_of=as_of or None)


@mcp.tool()
def get_current_facts(entity_name: str) -> dict:
    """Get only currently valid facts for an entity (excludes superseded/expired).

    Args:
        entity_name: The entity name to look up
    """
    store = _get_store()
    return store.get_current_facts(entity_name)


@mcp.tool()
def fact_timeline(entity_name: str) -> dict:
    """Show how facts about an entity evolved over time.

    Args:
        entity_name: The entity name to look up
    """
    store = _get_store()
    return store.fact_timeline(entity_name)


@mcp.tool()
def search_entities(query: str, limit: int = 20) -> dict:
    """Search for entities by name in the knowledge graph.

    Args:
        query: Search query for entity names
        limit: Maximum results (default 20)
    """
    store = _get_store()
    results = store.search_entities(query, limit=limit)
    return {"count": len(results), "entities": results}


# --- New tools: Import/Export ---


@mcp.tool()
def import_memories(format: str, file_path: str) -> dict:
    """Import memories from a local file or URL.

    Args:
        format: File format - chatgpt, claude, markdown, jsonl, pdf, web, csv, or json-array
        file_path: Path to the file/directory (or URL for 'web' format)
    """
    store = _get_store()
    count = store.import_memories(format=format, path=file_path)
    return {"imported": count, "format": format}


@mcp.tool()
def export_memories(
    format: str,
    file_path: str,
    tags: list[str] | None = None,
) -> dict:
    """Export memories to a file.

    Args:
        format: Export format - jsonl, markdown, or json
        file_path: Path to write the export
        tags: Optional tag filter
    """
    store = _get_store()
    count = store.export_memories(format=format, path=file_path, tags=tags)
    return {"exported": count, "format": format, "path": file_path}


# --- New tools: Obsidian ---


@mcp.tool()
def obsidian_sync(
    vault_path: str,
    direction: str = "both",
) -> dict:
    """Sync memories with an Obsidian vault.

    Args:
        vault_path: Path to the Obsidian vault root
        direction: Sync direction - 'to', 'from', or 'both'
    """
    from neuropack.io.obsidian import ObsidianSync

    store = _get_store()
    sync = ObsidianSync(vault_path=vault_path, store=store)

    if direction == "to":
        count = sync.sync_to_vault()
        return {"exported": count, "direction": "to"}
    elif direction == "from":
        count = sync.sync_from_vault()
        return {"imported": count, "direction": "from"}
    else:
        result = sync.full_sync()
        return {**result, "direction": "both"}


# --- New tools: Training Data ---


@mcp.tool()
def export_training(
    format: str,
    file_path: str,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Export memories as training data for fine-tuning.

    Args:
        format: Training format - openai, alpaca, qa, or embeddings
        file_path: Path to write the training data
        tags: Optional tag filter
        limit: Maximum records to export
    """
    store = _get_store()
    count = store.export_training(format=format, path=file_path, tags=tags, limit=limit)
    return {"exported": count, "format": format, "path": file_path}


# --- New tools: LLM Registry ---


@mcp.tool()
def list_llms() -> dict:
    """List all configured LLM providers (API keys are masked).

    Returns names, providers, models, and default status.
    """
    store = _get_store()
    configs = store._llm_registry.list_all()
    return {
        "count": len(configs),
        "llms": [
            {
                "name": c.name,
                "provider": c.provider,
                "model": c.model,
                "base_url": c.base_url or None,
                "api_key": c.masked_key() or None,
                "is_default": c.is_default,
            }
            for c in configs
        ],
    }


@mcp.tool()
def test_llm(name: str) -> dict:
    """Test an LLM connection with a simple prompt.

    Args:
        name: The name of the LLM config to test
    """
    store = _get_store()
    return store._llm_registry.test_connection(name)


# --- New tools: Multi-Agent Learning ---


@mcp.tool()
def agent_log(agent_name: str, content: str) -> dict:
    """Log an observation for an agent. Auto-tags as win/mistake/observation.

    Content containing words like 'profit', 'win', 'gained' gets tagged 'win'.
    Content containing 'loss', 'error', 'mistake', 'failed' gets tagged 'mistake'.
    Everything else is tagged 'observation'.

    Args:
        agent_name: The agent's namespace name
        content: The observation/learning to log
    """
    from neuropack.cli.agents import _auto_tag

    store = _get_store()
    tag = _auto_tag(content)
    record = store.store(
        content=content,
        tags=[tag],
        source=f"agent:{agent_name}",
        namespace=agent_name,
    )
    return {
        "id": record.id,
        "agent": agent_name,
        "tag": tag,
        "l3_abstract": record.l3_abstract,
    }


@mcp.tool()
def agent_scoreboard() -> dict:
    """Get agent rankings by win/mistake ratio.

    Returns all agents sorted by win ratio (wins / (wins + mistakes)).
    """
    store = _get_store()
    conn = store._db.connect()
    rows = conn.execute(
        "SELECT key FROM metadata WHERE key LIKE 'agent:%'"
    ).fetchall()

    board = []
    for row in rows:
        name = dict(row)["key"].replace("agent:", "", 1)
        wins = store.list(limit=1000, tag="win", namespace=name)
        mistakes = store.list(limit=1000, tag="mistake", namespace=name)
        total = len(wins) + len(mistakes)
        ratio = len(wins) / total if total > 0 else 0.0
        board.append({
            "agent": name,
            "wins": len(wins),
            "mistakes": len(mistakes),
            "total": store._repo.count_by_namespace(name),
            "win_ratio": round(ratio, 2),
        })

    board.sort(key=lambda x: x["win_ratio"], reverse=True)
    return {"agents": board}


# --- New tools: Consolidation, Backup, Staleness, Versioning ---


@mcp.tool()
def consolidate_memories(dry_run: bool = True) -> dict:
    """Consolidate similar memories into summaries to reduce redundancy.

    Args:
        dry_run: If True, shows what would be consolidated without making changes
    """
    store = _get_store()
    result = store.consolidate(dry_run=dry_run)
    return {
        "clusters_found": result.clusters_found,
        "memories_consolidated": result.memories_consolidated,
        "summaries_created": result.summaries_created,
        "archived_ids": result.archived_ids,
        "dry_run": dry_run,
    }


@mcp.tool()
def backup_store(backup_dir: str | None = None) -> dict:
    """Create a backup of the memory database.

    Args:
        backup_dir: Optional directory for backup (default: ~/.recall/backups)
    """
    store = _get_store()
    path = store.backup(backup_dir=backup_dir)
    return {"path": path, "status": "backup_created"}


@mcp.tool()
def get_stale_memories(limit: int = 20) -> dict:
    """Find memories that may be outdated based on their staleness category.

    Args:
        limit: Maximum number of stale memories to return
    """
    store = _get_store()
    stale = store.get_stale_memories(limit=limit)
    return {"count": len(stale), "stale_memories": stale}


@mcp.tool()
def memory_versions(memory_id: str) -> dict:
    """Get version history of a memory (previous content before updates).

    Args:
        memory_id: The memory ID to get versions for
    """
    store = _get_store()
    versions = store.get_versions(memory_id)
    return {
        "memory_id": memory_id,
        "version_count": len(versions),
        "versions": [
            {
                "version": v.version,
                "content_preview": v.content[:200],
                "l3_abstract": v.l3_abstract,
                "saved_at": v.saved_at.isoformat(),
                "reason": v.reason,
            }
            for v in versions
        ],
    }


@mcp.tool()
def inspect_memory(memory_id: str) -> dict:
    """Get full details of a memory including type, staleness, and trust.

    Args:
        memory_id: The memory ID to inspect
    """
    store = _get_store()
    record = store.get(memory_id)
    if record is None:
        return {"error": f"Memory {memory_id} not found"}
    trust = store.get_trust_report(memory_id)
    return {
        "id": record.id,
        "content": strip_private_from_preview(record.content[:500]),
        "l3_abstract": record.l3_abstract,
        "l2_facts": record.l2_facts,
        "tags": record.tags,
        "memory_type": record.memory_type,
        "staleness": record.staleness,
        "priority": record.priority,
        "source": record.source,
        "namespace": record.namespace,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "access_count": record.access_count,
        "superseded_by": record.superseded_by,
        "trust": trust,
    }


# --- New tools: Feedback, PII, Retention ---


@mcp.tool()
def memory_feedback(memory_id: str, useful: bool) -> dict:
    """Record feedback on a recall result to improve future ranking.

    Args:
        memory_id: The memory ID to rate
        useful: True if the result was helpful, False if not
    """
    store = _get_store()
    store.record_feedback(memory_id, useful)
    return {"memory_id": memory_id, "feedback": "positive" if useful else "negative", "status": "recorded"}


@mcp.tool()
def scan_pii(limit: int = 100) -> dict:
    """Scan memories for PII and secrets (API keys, emails, etc).

    Args:
        limit: Maximum memories to scan
    """
    store = _get_store()
    results = store.scan_pii(limit=limit)
    return {
        "scanned": limit,
        "findings": len(results),
        "memories_with_pii": results,
    }


@mcp.tool()
def purge_expired(dry_run: bool = True) -> dict:
    """Purge memories that have exceeded their retention policy.

    Args:
        dry_run: If True, shows what would be purged without deleting
    """
    store = _get_store()
    results = store.purge_expired(dry_run=dry_run)
    return {
        "expired_count": len(results),
        "expired_memories": results,
        "dry_run": dry_run,
    }


# --- Workspace Collaboration ---


@mcp.tool()
def workspace_create(name: str, goal: str, agent_name: str = "system") -> dict:
    """Create a shared workspace for multi-agent collaboration on a problem.

    Args:
        name: Human-readable workspace name
        goal: What this workspace aims to accomplish
        agent_name: The agent creating the workspace (auto-joins as owner)
    """
    store = _get_store()
    ws = store.workspace.create_workspace(name, goal, created_by=agent_name)
    return {"id": ws.id, "name": ws.name, "goal": ws.goal, "status": "created"}


@mcp.tool()
def workspace_join(workspace_id: str, agent_name: str) -> dict:
    """Join a workspace and receive catchup context on what has happened so far.

    Args:
        workspace_id: The workspace to join
        agent_name: Your agent identifier
    """
    store = _get_store()
    store.workspace.join_workspace(workspace_id, agent_name)
    catchup = store.workspace_catchup(workspace_id, agent_name)
    return {"joined": True, "workspace_id": workspace_id, "catchup": catchup}


@mcp.tool()
def workspace_list(status: str | None = None) -> dict:
    """List all workspaces.

    Args:
        status: Optional filter: active, completed, archived
    """
    store = _get_store()
    workspaces = store.workspace.list_workspaces(status=status)
    return {
        "count": len(workspaces),
        "workspaces": [
            {"id": w.id, "name": w.name, "goal": w.goal,
             "status": w.status, "created_by": w.created_by}
            for w in workspaces
        ],
    }


@mcp.tool()
def workspace_task_create(
    workspace_id: str, title: str, description: str = "", agent_name: str = "system"
) -> dict:
    """Create a task on the workspace board.

    Args:
        workspace_id: The workspace this task belongs to
        title: Short task title
        description: What needs to be done
        agent_name: Agent creating the task
    """
    store = _get_store()
    task = store.workspace.create_task(workspace_id, title, description, created_by=agent_name)
    return {"id": task.id, "title": task.title, "status": task.status}


@mcp.tool()
def workspace_task_claim(task_id: str, agent_name: str) -> dict:
    """Claim a task so other agents know you are working on it. Prevents duplicate work.

    Args:
        task_id: The task to claim
        agent_name: Your agent identifier
    """
    store = _get_store()
    task = store.workspace.claim_task(task_id, agent_name)
    return {"id": task.id, "title": task.title, "status": task.status,
            "assigned_to": task.assigned_to}


@mcp.tool()
def workspace_task_complete(task_id: str, agent_name: str) -> dict:
    """Mark a task as done. Automatically unblocks dependent tasks.

    Args:
        task_id: The task to complete
        agent_name: Your agent identifier
    """
    store = _get_store()
    task = store.workspace.complete_task(task_id, agent_name)
    return {"id": task.id, "title": task.title, "status": "done"}


@mcp.tool()
def workspace_task_list(workspace_id: str, status: str | None = None) -> dict:
    """List tasks on the workspace board.

    Args:
        workspace_id: The workspace to list tasks for
        status: Optional filter: open, claimed, blocked, done
    """
    store = _get_store()
    tasks = store.workspace.list_tasks(workspace_id, status=status)
    return {
        "count": len(tasks),
        "tasks": [
            {"id": t.id, "title": t.title, "status": t.status,
             "assigned_to": t.assigned_to, "description": t.description[:200]}
            for t in tasks
        ],
    }


@mcp.tool()
def workspace_handoff(
    workspace_id: str,
    agent_name: str,
    summary: str,
    findings: list[str] | None = None,
    decisions: list[str] | None = None,
    open_questions: list[str] | None = None,
    memory_ids: list[str] | None = None,
    to_agent: str | None = None,
    task_id: str | None = None,
) -> dict:
    """Post a structured handoff when you finish a subtask. Ensures the next agent has full context.

    Args:
        workspace_id: The workspace
        agent_name: Your agent identifier
        summary: One-paragraph summary of your work
        findings: Concrete facts discovered
        decisions: Decisions you made and why
        open_questions: Unresolved items for the next agent
        memory_ids: ReCall memory IDs with detailed content
        to_agent: Specific agent to hand off to (None = anyone)
        task_id: Related task ID
    """
    store = _get_store()
    context = {
        "findings": findings or [],
        "decisions": decisions or [],
        "open_questions": open_questions or [],
    }
    handoff = store.workspace.post_handoff(
        workspace_id, agent_name, summary, context,
        memory_ids=memory_ids, to_agent=to_agent, task_id=task_id,
    )
    return {"id": handoff.id, "summary": handoff.summary, "status": "posted"}


@mcp.tool()
def workspace_decide(
    workspace_id: str,
    title: str,
    rationale: str,
    agent_name: str,
    alternatives: list[str] | None = None,
    task_id: str | None = None,
) -> dict:
    """Log a decision with rationale so all agents know WHY a choice was made.

    Args:
        workspace_id: The workspace
        title: What was decided
        rationale: Why this decision was made
        agent_name: Who made the decision
        alternatives: Rejected alternatives
        task_id: Related task ID
    """
    store = _get_store()
    decision = store.workspace.log_decision(
        workspace_id, title, rationale, agent_name,
        alternatives=alternatives, related_task_id=task_id,
    )
    return {"id": decision.id, "title": decision.title, "status": "logged"}


@mcp.tool()
def workspace_activity(workspace_id: str, limit: int = 30) -> dict:
    """Get the workspace activity feed -- who did what and when.

    Args:
        workspace_id: The workspace
        limit: Maximum entries (default 30)
    """
    store = _get_store()
    feed = store.workspace.activity_feed(workspace_id, limit=limit)
    return {"count": len(feed), "activity": feed}


@mcp.tool()
def workspace_catchup(
    workspace_id: str, agent_name: str, token_budget: int = 4000
) -> dict:
    """Get caught up on a workspace. Returns progressive context within your token budget.

    Use this when joining a workspace or returning after being away. Returns:
    - L3 overview (goal, task board, decision titles) -- always included
    - L2 details (handoff summaries, decision rationales) -- if budget allows
    - Full memories linked by handoffs -- remaining budget

    Args:
        workspace_id: The workspace to catch up on
        agent_name: Your agent identifier
        token_budget: Max tokens for catchup context (default 4000)
    """
    store = _get_store()
    return store.workspace_catchup(workspace_id, agent_name, token_budget)


@mcp.tool()
def recall_and_synthesize(query: str, limit: int = 10, synthesize: bool = True) -> dict:
    """Recall memories and generate a synthesized insight across them.

    Uses LLM to find patterns, connections, and unified insights from
    multiple recalled memories. Requires an LLM provider to be configured.

    Args:
        query: What to search for
        limit: Maximum memories to recall (default 10)
        synthesize: Whether to generate LLM synthesis (default True)
    """
    store = _get_store()
    return store.recall_and_synthesize(query, limit=limit, synthesize=synthesize)


@mcp.tool()
def agent_recall(agent_name: str, query: str, limit: int = 10) -> dict:
    """Search an agent's personal memory, including workspace learnings absorbed over time.

    Use this to recall what a specific agent has learned across all their workspaces
    and direct observations.

    Args:
        agent_name: The agent whose memory to search
        query: What to search for
        limit: Max results (default 10)
    """
    store = _get_store()
    results = store.agent_recall(agent_name, query, limit=limit)
    return {
        "agent": agent_name,
        "count": len(results),
        "results": [
            {
                "id": r.record.id,
                "l3_abstract": r.record.l3_abstract,
                "tags": r.record.tags,
                "source": r.record.source,
                "score": round(r.score, 3),
                "created_at": r.record.created_at.isoformat(),
            }
            for r in results
        ],
    }


@mcp.tool()
def agent_expertise(agent_name: str) -> dict:
    """Get an agent's expertise profile: wins, mistakes, workspace participation, and learnings.

    Use this to understand what an agent is good at and what workspaces they've contributed to.

    Args:
        agent_name: The agent to profile
    """
    store = _get_store()
    return store.agent_expertise(agent_name)


@mcp.tool()
def agent_promote(agent_name: str, memory_id: str, priority: float = 0.8) -> dict:
    """Promote a memory in an agent's namespace: boost priority and mark stable.

    Args:
        agent_name: The agent who owns the memory
        memory_id: The memory to promote
        priority: New priority floor (default 0.8)
    """
    store = _get_store()
    mgr = store.agent_memory(agent_name)
    ok = mgr.promote(memory_id, priority=priority)
    return {"promoted": ok, "memory_id": memory_id, "agent": agent_name}


@mcp.tool()
def agent_demote(agent_name: str, memory_id: str, priority: float = 0.2) -> dict:
    """Demote a memory: lower priority and mark as volatile.

    Args:
        agent_name: The agent who owns the memory
        memory_id: The memory to demote
        priority: New priority ceiling (default 0.2)
    """
    store = _get_store()
    mgr = store.agent_memory(agent_name)
    ok = mgr.demote(memory_id, priority=priority)
    return {"demoted": ok, "memory_id": memory_id, "agent": agent_name}


@mcp.tool()
def agent_archive(agent_name: str, memory_id: str, reason: str = "archived") -> dict:
    """Archive a memory: save version snapshot, tag as archived, lower priority.

    Args:
        agent_name: The agent who owns the memory
        memory_id: The memory to archive
        reason: Reason for archiving (default "archived")
    """
    store = _get_store()
    mgr = store.agent_memory(agent_name)
    ok = mgr.archive(memory_id, reason=reason)
    return {"archived": ok, "memory_id": memory_id, "agent": agent_name}


@mcp.tool()
def agent_pin(agent_name: str, memory_id: str) -> dict:
    """Pin a memory: max priority, never auto-purge, always in context.

    Args:
        agent_name: The agent who owns the memory
        memory_id: The memory to pin
    """
    store = _get_store()
    mgr = store.agent_memory(agent_name)
    ok = mgr.pin(memory_id)
    return {"pinned": ok, "memory_id": memory_id, "agent": agent_name}


# --- Memory Diffing & Time Travel ---


@mcp.tool()
def memory_diff(since: str, until: str = "") -> dict:
    """Show what changed in your knowledge between two time points.

    Returns new memories added, memories updated, and memories deleted
    in the specified time range, with aggregate statistics.

    Args:
        since: Start time - e.g. "last week", "3 days ago", "yesterday", "2026-03-01"
        until: End time (default: now) - same formats as since
    """
    store = _get_store()
    return store.diff(since=since, until=until or None)


@mcp.tool()
def recall_as_of(query: str, as_of: str, limit: int = 10) -> dict:
    """Recall memories as they existed at a past point in time.

    Searches for memories that existed at the given time and reconstructs
    their historical state using version history. Useful for understanding
    what you knew at a specific moment.

    Args:
        query: What to search for
        as_of: Point in time - e.g. "last week", "2026-03-01", "yesterday"
        limit: Maximum results (default 10)
    """
    store = _get_store()
    results = store.recall_as_of(query=query, as_of=as_of, limit=limit)
    return {"query": query, "as_of": as_of, "count": len(results), "results": results}


@mcp.tool()
def knowledge_timeline(entity: str = "", tag: str = "", granularity: str = "day") -> dict:
    """Show how your knowledge evolved over time as a timeline.

    Groups memory additions, modifications, and deletions by time period
    with the most active tags per period. Filter by entity or tag to focus
    on specific topics.

    Args:
        entity: Optional entity name to filter by (from knowledge graph)
        tag: Optional tag to filter by
        granularity: Time grouping - "day", "week", or "month"
    """
    store = _get_store()
    entries = store.knowledge_timeline(
        entity=entity or None,
        tag=tag or None,
        granularity=granularity,
    )
    return {"granularity": granularity, "count": len(entries), "timeline": entries}


# --- Developer DNA Profile ---


@mcp.tool()
def developer_profile(namespace: str | None = None) -> dict:
    """Get your full Developer DNA profile -- coding style, patterns, and preferences.

    Analyzes your stored memories to build a statistical profile of your development
    habits including naming conventions, architecture patterns, error handling style,
    preferred libraries, code style, review themes, and things you avoid.

    Args:
        namespace: Optional namespace to scope the profile to
    """
    store = _get_store()
    return store.get_developer_profile(namespace=namespace)


@mcp.tool()
def coding_style(aspect: str, namespace: str | None = None) -> dict:
    """Query a specific aspect of your developer profile.

    Available aspects: naming_conventions, architecture_patterns, error_handling,
    preferred_libraries, code_style, review_feedback, anti_patterns.

    Args:
        aspect: The profile section to query
        namespace: Optional namespace to scope the profile to
    """
    store = _get_store()
    return store.query_coding_style(aspect, namespace=namespace)


@mcp.tool()
def rebuild_profile(namespace: str | None = None) -> dict:
    """Force a full rebuild of your Developer DNA profile.

    Re-analyzes all memories to update your coding style profile.
    Useful after storing many new code-related memories.

    Args:
        namespace: Optional namespace to scope the rebuild to
    """
    store = _get_store()
    return store.rebuild_developer_profile(namespace=namespace)


# --- Anticipatory Context ---


@mcp.tool()
def anticipatory_context(token_budget: int = 4000) -> dict:
    """Get pre-loaded anticipatory context from the background watcher.

    The watcher monitors file changes, git activity, and terminal commands
    to pre-load relevant memories before you need them. Returns memories
    that are likely relevant to your current work.

    Args:
        token_budget: Maximum tokens for the context (default 4000)
    """
    store = _get_store()
    items = store.get_anticipatory_context(token_budget=token_budget)
    return {
        "count": len(items),
        "items": items,
        "hint": "Start the watcher with 'np watch <dirs>' if no items are returned",
    }


@mcp.tool()
def watcher_status() -> dict:
    """Check if the anticipatory context watcher is running and see its stats.

    Returns whether the watcher is active, what directories it monitors,
    and cache statistics (hit rate, cached queries, total results).
    """
    store = _get_store()
    return store.watcher_status()


# --- LLM Proxy Status ---


@mcp.tool()
def proxy_status() -> dict:
    """Check if the ReCall LLM proxy is running and how many calls have been captured.

    Returns the number of memories tagged with 'llm-call', grouped by provider,
    along with total token usage tracked in metadata.
    """
    store = _get_store()

    # Count llm-call tagged memories
    all_calls = store.list(limit=10000, tag="llm-call")
    by_provider: dict[str, int] = {}
    total_tokens = 0

    for mem in all_calls:
        for tag in mem.tags:
            if tag.startswith("provider-"):
                provider = tag.replace("provider-", "", 1)
                by_provider[provider] = by_provider.get(provider, 0) + 1

    return {
        "total_calls_captured": len(all_calls),
        "by_provider": by_provider,
        "hint": "Start the proxy with: np proxy --port 8741",
    }


def main():
    global _store
    config = NeuropackConfig()
    _store = MemoryStore(config)
    _store.initialize()
    try:
        mcp.run(transport="stdio")
    finally:
        _store.close()


if __name__ == "__main__":
    main()

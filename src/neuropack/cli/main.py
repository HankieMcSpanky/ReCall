from __future__ import annotations

import json
import sys

import click

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


@click.group()
@click.option("--db", envvar="NEUROPACK_DB_PATH", default=None, help="Database path")
@click.option("--namespace", "-n", default=None, help="Namespace for operations")
@click.pass_context
def cli(ctx: click.Context, db: str | None, namespace: str | None) -> None:
    """ReCall - The universal memory layer for AI."""
    kwargs = {}
    if db:
        kwargs["db_path"] = db
    if namespace:
        kwargs["namespace"] = namespace
    config = NeuropackConfig(**kwargs) if kwargs else NeuropackConfig()
    store = MemoryStore(config)
    store.initialize()
    ctx.ensure_object(dict)
    ctx.obj["store"] = store
    ctx.obj["config"] = config
    ctx.call_on_close(store.close)

    # First-run hint (only once, only for read-like commands)
    invoked = ctx.invoked_subcommand
    if invoked in ("list", "recall", "stats", "namespaces"):
        from neuropack.cli.onboarding import check_first_run
        check_first_run(store)


@cli.command()
@click.argument("text")
@click.option("--tag", "-t", multiple=True, help="Tags for categorization")
@click.option("--source", "-s", default="", help="Source identifier")
@click.option("--priority", "-p", default=0.5, type=float, help="Priority 0.0-1.0")
@click.pass_context
def store(ctx: click.Context, text: str, tag: tuple[str, ...], source: str, priority: float) -> None:
    """Store a new memory."""
    ms: MemoryStore = ctx.obj["store"]
    record = ms.store(content=text, tags=list(tag), source=source, priority=priority)
    click.echo(json.dumps({
        "id": record.id,
        "l3": record.l3_abstract,
        "tags": record.tags,
        "namespace": record.namespace,
        "compression": {
            "original": len(text),
            "l1": len(record.l1_compressed),
        },
    }, indent=2))


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int, help="Max results")
@click.option("--tag", "-t", multiple=True, help="Filter by tag")
@click.option("--min-score", default=0.0, type=float, help="Minimum score threshold")
@click.option("--synthesize", is_flag=True, help="Generate LLM synthesis across results")
@click.option("--as-of", default=None, help="Recall memories as they existed at this time (e.g. 'last week', '2026-03-01')")
@click.pass_context
def recall(ctx: click.Context, query: str, limit: int, tag: tuple[str, ...], min_score: float, synthesize: bool, as_of: str | None) -> None:
    """Search memories by semantic and keyword similarity."""
    ms: MemoryStore = ctx.obj["store"]

    if as_of:
        results = ms.recall_as_of(query=query, as_of=as_of, limit=limit)
        if not results:
            click.echo("No memories found for that time.")
            return
        for r in results:
            click.echo(json.dumps(r, indent=2))
            click.echo("---")
        return

    if synthesize:
        result = ms.recall_and_synthesize(query=query, limit=limit, synthesize=True)
        for r in result.get("results", []):
            click.echo(json.dumps({
                "id": r["id"],
                "score": round(r["score"], 4),
                "l3": r["l3_abstract"],
                "tags": r["tags"],
            }, indent=2))
            click.echo("---")
        if "synthesis" in result:
            click.echo("\n=== Synthesis ===")
            click.echo(json.dumps(result["synthesis"], indent=2))
        return

    results = ms.recall(query=query, limit=limit, tags=list(tag) or None, min_score=min_score)
    if not results:
        click.echo("No memories found.")
        return
    for r in results:
        click.echo(json.dumps({
            "id": r.record.id,
            "score": round(r.score, 4),
            "l3": r.record.l3_abstract,
            "tags": r.record.tags,
            "namespace": r.record.namespace,
            "vec_score": round(r.vec_score, 4) if r.vec_score is not None else None,
            "fts_rank": round(r.fts_rank, 4) if r.fts_rank is not None else None,
        }, indent=2))
        click.echo("---")


@cli.command("list")
@click.option("--limit", "-l", default=20, type=int, help="Number of memories")
@click.option("--offset", default=0, type=int, help="Skip N memories")
@click.option("--tag", "-t", default=None, help="Filter by tag")
@click.pass_context
def list_memories(ctx: click.Context, limit: int, offset: int, tag: str | None) -> None:
    """List stored memories."""
    ms: MemoryStore = ctx.obj["store"]
    records = ms.list(limit=limit, offset=offset, tag=tag)
    if not records:
        click.echo("No memories stored.")
        return
    for r in records:
        click.echo(json.dumps({
            "id": r.id,
            "l3": r.l3_abstract,
            "tags": r.tags,
            "priority": r.priority,
            "namespace": r.namespace,
            "created": r.created_at.isoformat(),
        }))


@cli.command()
@click.argument("memory_id")
@click.option("--history", is_flag=True, help="Show version history")
@click.pass_context
def inspect(ctx: click.Context, memory_id: str, history: bool) -> None:
    """Show full details of a memory including all compression levels."""
    ms: MemoryStore = ctx.obj["store"]
    record = ms.get(memory_id)
    if record is None:
        click.echo(f"Memory {memory_id} not found.", err=True)
        sys.exit(1)

    if history:
        versions = ms.get_versions(memory_id)
        if not versions:
            click.echo("No version history.")
        else:
            for v in versions:
                click.echo(json.dumps({
                    "version": v.version,
                    "reason": v.reason,
                    "saved_at": v.saved_at.isoformat(),
                    "l3_abstract": v.l3_abstract,
                    "tags": v.tags,
                    "content_preview": v.content[:200],
                }, indent=2))
                click.echo("---")
        return

    raw_text = ms.decompress(record.l1_compressed)
    click.echo(json.dumps({
        "id": record.id,
        "l3_abstract": record.l3_abstract,
        "l2_facts": record.l2_facts,
        "raw_text": raw_text,
        "tags": record.tags,
        "source": record.source,
        "priority": record.priority,
        "namespace": record.namespace,
        "memory_type": record.memory_type,
        "staleness": record.staleness,
        "superseded_by": record.superseded_by,
        "access_count": record.access_count,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "sizes": {
            "raw": len(raw_text),
            "l1_compressed": len(record.l1_compressed),
            "l3": len(record.l3_abstract),
        },
    }, indent=2))


@cli.command()
@click.argument("memory_id")
@click.pass_context
def forget(ctx: click.Context, memory_id: str) -> None:
    """Delete a memory by ID."""
    ms: MemoryStore = ctx.obj["store"]
    deleted = ms.forget(memory_id)
    if deleted:
        click.echo(f"Deleted memory {memory_id}")
    else:
        click.echo(f"Memory {memory_id} not found.", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show store statistics."""
    ms: MemoryStore = ctx.obj["store"]
    s = ms.stats()
    click.echo(json.dumps({
        "total_memories": s.total_memories,
        "total_size_bytes": s.total_size_bytes,
        "avg_compression_ratio": round(s.avg_compression_ratio, 2),
        "oldest": s.oldest.isoformat() if s.oldest else None,
        "newest": s.newest.isoformat() if s.newest else None,
    }, indent=2))


@cli.command("session-summary")
@click.argument("memory_ids", nargs=-1, required=True)
@click.option("--store/--no-store", "do_store", default=False, help="Store summary as a memory")
@click.pass_context
def session_summary(ctx: click.Context, memory_ids: tuple[str, ...], do_store: bool) -> None:
    """Generate a session summary from memory IDs."""
    ms: MemoryStore = ctx.obj["store"]
    summary = ms.session_summary(list(memory_ids))
    click.echo(json.dumps(summary, indent=2))
    if do_store:
        record = ms.store_session_summary(list(memory_ids))
        click.echo(f"\nStored as memory: {record.id}")


@cli.command("generate-context")
@click.option("--output", "-o", default="./CLAUDE.md", help="Output file path")
@click.option("--limit", "-l", default=50, type=int, help="Max memories to include")
@click.option("--tag", "-t", multiple=True, help="Filter by tags")
@click.option("--watch", is_flag=False, flag_value=30, default=None, type=int,
              help="Regenerate every N seconds (default 30)")
@click.pass_context
def generate_context(
    ctx: click.Context, output: str, limit: int, tag: tuple[str, ...], watch: int | None
) -> None:
    """Generate a CLAUDE.md context file from recent memories."""
    import time
    from pathlib import Path

    ms: MemoryStore = ctx.obj["store"]
    tags = list(tag) if tag else None

    def _generate() -> None:
        md = ms.generate_context(limit=limit, tags=tags)
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        click.echo(f"Written to {path} ({len(md)} bytes)")

    _generate()

    if watch is not None:
        click.echo(f"Watching... regenerating every {watch}s (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(watch)
                _generate()
        except KeyboardInterrupt:
            click.echo("\nStopped watching.")


def _get_lan_ip() -> str:
    """Detect LAN IP address with fallback."""
    import socket

    # Method 1: UDP connect trick (works on most systems)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 2: Enumerate host addresses
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "0.0.0.0"


@cli.command()
@click.option("--port", default=7341, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--expose", is_flag=True, default=False, help="Bind to 0.0.0.0 for LAN access (phone, other devices)")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str, expose: bool) -> None:
    """Start the HTTP API server."""
    import uvicorn

    from neuropack.api.app import create_app

    if expose:
        host = "0.0.0.0"

    config: NeuropackConfig = ctx.obj["config"]
    config = config.model_copy(update={"api_port": port, "api_host": host})

    store: MemoryStore = ctx.obj["store"]

    # Auto-create API key if exposing without auth
    if expose and not config.auth_token:
        try:
            from neuropack.auth.keys import APIKeyManager

            key_mgr = APIKeyManager(store._db)
            existing = key_mgr.list_keys()
            if not existing:
                raw_key = key_mgr.create_key("mobile-access", scopes=["read", "write"])
                click.echo(f"\n  Auto-created API key for LAN access:")
                click.echo(f"  {raw_key}\n")
            else:
                raw_key = None
        except Exception as e:
            click.echo(f"  Warning: Could not auto-create API key: {e}", err=True)
            raw_key = None

        lan_ip = _get_lan_ip()
        click.echo(f"  Mobile UI: http://{lan_ip}:{port}/mobile")
        if raw_key:
            click.echo(f"  Enter the API key above on first use.\n")
        else:
            click.echo(f"  (API key already configured)\n")

    # Close the CLI store since the server will create its own
    store.close()

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--port", default=7341, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.pass_context
def app(ctx: click.Context, port: int, host: str) -> None:
    """Launch ReCall desktop app with GUI dashboard."""
    ctx.obj["store"].close()

    from neuropack.desktop.launcher import NeuropackDesktop

    config: NeuropackConfig = ctx.obj["config"]
    desktop = NeuropackDesktop(host=host, port=port, db_path=config.db_path)
    desktop.run()


# --- New commands: Namespaces ---


@cli.command("namespaces")
@click.pass_context
def namespaces(ctx: click.Context) -> None:
    """List all namespaces with memory counts."""
    ms: MemoryStore = ctx.obj["store"]
    ns_list = ms.list_namespaces()
    if not ns_list:
        click.echo("No namespaces found.")
        return
    click.echo(json.dumps(ns_list, indent=2))


# --- New commands: Knowledge Graph ---


@cli.group("graph")
def graph_group() -> None:
    """Knowledge graph commands."""
    pass


@graph_group.command("entity")
@click.argument("name")
@click.pass_context
def graph_entity(ctx: click.Context, name: str) -> None:
    """Show entity and its relationships."""
    ms: MemoryStore = ctx.obj["store"]
    result = ms.query_entity(name)
    click.echo(json.dumps(result, indent=2))


@graph_group.command("search")
@click.argument("query")
@click.option("--limit", "-l", default=20, type=int, help="Max results")
@click.pass_context
def graph_search(ctx: click.Context, query: str, limit: int) -> None:
    """Search entities by name."""
    ms: MemoryStore = ctx.obj["store"]
    results = ms.search_entities(query, limit=limit)
    click.echo(json.dumps(results, indent=2))


# --- New commands: Import/Export ---


@cli.command("import")
@click.argument("file")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["chatgpt", "claude", "markdown", "jsonl", "pdf", "web", "csv", "json-array"]),
              help="Import format")
@click.option("--tag", "-t", default="imported", help="Source tag to add")
@click.pass_context
def import_cmd(ctx: click.Context, file: str, fmt: str, tag: str) -> None:
    """Import memories from external formats."""
    ms: MemoryStore = ctx.obj["store"]
    count = ms.import_memories(format=fmt, path=file, source_tag=tag)
    click.echo(f"Imported {count} memories from {fmt} format")


@cli.command("export")
@click.argument("path")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["jsonl", "markdown", "json"]),
              help="Export format")
@click.option("--tag", "-t", multiple=True, help="Filter by tags")
@click.option("--limit", "-l", default=None, type=int, help="Max records")
@click.pass_context
def export_cmd(ctx: click.Context, path: str, fmt: str, tag: tuple[str, ...], limit: int | None) -> None:
    """Export memories to a file or directory."""
    ms: MemoryStore = ctx.obj["store"]
    tags = list(tag) if tag else None
    count = ms.export_memories(format=fmt, path=path, tags=tags, limit=limit)
    click.echo(f"Exported {count} memories to {path}")


@cli.command("export-training")
@click.argument("path")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["openai", "alpaca", "qa", "embeddings"]),
              help="Training data format")
@click.option("--tag", "-t", multiple=True, help="Filter by tags")
@click.option("--limit", "-l", default=None, type=int, help="Max records")
@click.pass_context
def export_training_cmd(ctx: click.Context, path: str, fmt: str, tag: tuple[str, ...], limit: int | None) -> None:
    """Export memories as training data for fine-tuning."""
    ms: MemoryStore = ctx.obj["store"]
    tags = list(tag) if tag else None
    count = ms.export_training(format=fmt, path=path, tags=tags, limit=limit)
    click.echo(f"Exported {count} records to {path} ({fmt} format)")


# --- New commands: Obsidian ---


@cli.group("obsidian")
def obsidian_group() -> None:
    """Obsidian vault sync commands."""
    pass


@obsidian_group.command("sync")
@click.argument("vault_path")
@click.option("--direction", "-d", default="both",
              type=click.Choice(["to", "from", "both"]),
              help="Sync direction")
@click.pass_context
def obsidian_sync(ctx: click.Context, vault_path: str, direction: str) -> None:
    """Sync memories with an Obsidian vault."""
    from neuropack.io.obsidian import ObsidianSync

    ms: MemoryStore = ctx.obj["store"]
    sync = ObsidianSync(vault_path=vault_path, store=ms)

    if direction == "to":
        count = sync.sync_to_vault()
        click.echo(f"Exported {count} memories to vault")
    elif direction == "from":
        count = sync.sync_from_vault()
        click.echo(f"Imported {count} memories from vault")
    else:
        result = sync.full_sync()
        click.echo(f"Exported {result['exported']}, imported {result['imported']}")


@obsidian_group.command("status")
@click.argument("vault_path")
@click.pass_context
def obsidian_status(ctx: click.Context, vault_path: str) -> None:
    """Show sync status with an Obsidian vault."""
    from pathlib import Path

    ms: MemoryStore = ctx.obj["store"]
    vault = Path(vault_path)
    sync_dir = vault / "ReCall"

    vault_count = len(list(sync_dir.glob("*.md"))) if sync_dir.exists() else 0
    memory_count = ms.stats().total_memories

    click.echo(json.dumps({
        "vault_path": str(vault),
        "vault_files": vault_count,
        "total_memories": memory_count,
        "sync_folder": str(sync_dir),
    }, indent=2))


# --- Onboarding commands ---


@cli.command("init")
@click.pass_context
def init_cmd(ctx: click.Context) -> None:
    """Interactive setup wizard."""
    from neuropack.cli.onboarding import run_init
    config: NeuropackConfig = ctx.obj["config"]
    # Close the store created by cli() -- init creates its own
    ctx.obj["store"].close()
    run_init(config)


@cli.command("doctor")
@click.pass_context
def doctor_cmd(ctx: click.Context) -> None:
    """Run health check diagnostics."""
    from neuropack.cli.onboarding import run_doctor
    ms: MemoryStore = ctx.obj["store"]
    run_doctor(ms)


# --- LLM management commands ---


@cli.group("llm")
def llm_group() -> None:
    """LLM provider management."""
    pass


@llm_group.command("add")
@click.argument("name")
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic", "gemini", "openai-compatible", "ollama"]),
              prompt="Provider")
@click.option("--api-key", "-k", default="", help="API key")
@click.option("--model", "-m", default="", help="Model name")
@click.option("--base-url", "-u", default="", help="Base URL (for openai-compatible)")
@click.option("--default/--no-default", "is_default", default=False, help="Set as default LLM")
@click.pass_context
def llm_add(ctx: click.Context, name: str, provider: str, api_key: str, model: str,
            base_url: str, is_default: bool) -> None:
    """Add or update a named LLM configuration."""
    from neuropack.llm.models import LLMConfig

    ms: MemoryStore = ctx.obj["store"]
    config = LLMConfig(
        name=name,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        is_default=is_default,
    )
    ms._llm_registry.add(config)

    # Test connection
    click.echo(f"Testing connection to '{name}'...", nl=False)
    result = ms._llm_registry.test_connection(name)
    if result["ok"]:
        click.echo(f' OK ({result["time_s"]}s)')
    else:
        click.echo(f' WARN ({result["error"]})')

    click.echo(f"Saved LLM config '{name}' (provider: {provider})")


@llm_group.command("remove")
@click.argument("name")
@click.pass_context
def llm_remove(ctx: click.Context, name: str) -> None:
    """Remove a named LLM configuration."""
    ms: MemoryStore = ctx.obj["store"]
    if ms._llm_registry.remove(name):
        click.echo(f"Removed LLM config '{name}'")
    else:
        click.echo(f"LLM config '{name}' not found.", err=True)


@llm_group.command("list")
@click.pass_context
def llm_list(ctx: click.Context) -> None:
    """List all configured LLMs (keys masked)."""
    ms: MemoryStore = ctx.obj["store"]
    configs = ms._llm_registry.list_all()
    if not configs:
        click.echo("No LLMs configured. Run: np llm add <name>")
        return
    for c in configs:
        default_marker = " (default)" if c.is_default else ""
        click.echo(json.dumps({
            "name": c.name,
            "provider": c.provider,
            "model": c.model,
            "base_url": c.base_url or None,
            "api_key": c.masked_key() or None,
            "default": c.is_default,
        }))
        click.echo(f"  {c.name}: {c.provider}/{c.model}{default_marker}")


@llm_group.command("test")
@click.argument("name", required=False)
@click.pass_context
def llm_test(ctx: click.Context, name: str | None) -> None:
    """Test LLM connection(s)."""
    ms: MemoryStore = ctx.obj["store"]
    if name:
        configs = [ms._llm_registry.get(name)]
        if configs[0] is None:
            click.echo(f"LLM config '{name}' not found.", err=True)
            return
    else:
        configs = ms._llm_registry.list_all()
        if not configs:
            click.echo("No LLMs configured.")
            return

    for c in configs:
        click.echo(f"Testing '{c.name}'...", nl=False)
        result = ms._llm_registry.test_connection(c.name)
        if result["ok"]:
            click.echo(f' OK ({result["time_s"]}s) - "{result["response"]}"')
        else:
            click.echo(f' FAIL ({result["error"]})')


@llm_group.command("set-default")
@click.argument("name")
@click.pass_context
def llm_set_default(ctx: click.Context, name: str) -> None:
    """Set the default LLM for compression."""
    ms: MemoryStore = ctx.obj["store"]
    try:
        ms._llm_registry.set_default(name)
        click.echo(f"Set '{name}' as default LLM")
    except ValueError as e:
        click.echo(str(e), err=True)


# --- Agent commands ---


# --- Audit commands ---


@cli.command("audit")
@click.option("--action", "-a", default=None, help="Filter by action (store, update, delete, share)")
@click.option("--actor", default=None, help="Filter by actor")
@click.option("--limit", "-l", default=20, type=int, help="Number of entries")
@click.pass_context
def audit_cmd(ctx: click.Context, action: str | None, actor: str | None, limit: int) -> None:
    """Show audit log."""
    ms: MemoryStore = ctx.obj["store"]
    entries = ms._audit.query(action=action, actor=actor, limit=limit)
    if not entries:
        click.echo("No audit entries.")
        return
    for entry in entries:
        click.echo(json.dumps(entry, indent=2))


# --- API Key commands ---


@cli.group("api-key")
def api_key_group() -> None:
    """API key management."""
    pass


@api_key_group.command("create")
@click.argument("name")
@click.option("--scope", "-s", default="read,write", help="Comma-separated scopes: read,write,admin")
@click.pass_context
def api_key_create(ctx: click.Context, name: str, scope: str) -> None:
    """Create a new API key."""
    ms: MemoryStore = ctx.obj["store"]
    scopes = [s.strip() for s in scope.split(",")]
    try:
        raw_key = ms._api_key_manager.create_key(name, scopes)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo(f"Created API key '{name}'")
    click.echo(f"  Key: {raw_key}")
    click.echo(f"  Scopes: {scopes}")
    click.echo("  Save this key -- it cannot be retrieved again.")


@api_key_group.command("list")
@click.pass_context
def api_key_list(ctx: click.Context) -> None:
    """List all active API keys."""
    ms: MemoryStore = ctx.obj["store"]
    keys = ms._api_key_manager.list_keys()
    if not keys:
        click.echo("No API keys. Create one with: np api-key create <name>")
        return
    for k in keys:
        click.echo(json.dumps(k, indent=2))


@api_key_group.command("revoke")
@click.argument("name")
@click.pass_context
def api_key_revoke(ctx: click.Context, name: str) -> None:
    """Revoke an API key by name."""
    ms: MemoryStore = ctx.obj["store"]
    if ms._api_key_manager.revoke_key(name):
        click.echo(f"Revoked API key '{name}'")
    else:
        click.echo(f"API key '{name}' not found.", err=True)
        sys.exit(1)


# --- Consolidation ---


@cli.command("consolidate")
@click.option("--dry-run", is_flag=True, help="Show what would be consolidated without modifying data")
@click.pass_context
def consolidate_cmd(ctx: click.Context, dry_run: bool) -> None:
    """Consolidate similar memories into summaries."""
    ms: MemoryStore = ctx.obj["store"]
    result = ms.consolidate(dry_run=dry_run)
    if dry_run:
        click.echo(f"Dry run: found {result.clusters_found} clusters "
                    f"covering {result.memories_consolidated} memories")
    else:
        click.echo(json.dumps({
            "clusters_found": result.clusters_found,
            "memories_consolidated": result.memories_consolidated,
            "summaries_created": result.summaries_created,
            "archived_ids": result.archived_ids,
        }, indent=2))


# --- Backup / Restore ---


@cli.command("backup")
@click.option("--dir", "backup_dir", default=None, help="Directory for backups")
@click.pass_context
def backup_cmd(ctx: click.Context, backup_dir: str | None) -> None:
    """Create a database backup."""
    ms: MemoryStore = ctx.obj["store"]
    path = ms.backup(backup_dir=backup_dir)
    click.echo(f"Backup created: {path}")


@cli.command("restore")
@click.argument("backup_file")
@click.pass_context
def restore_cmd(ctx: click.Context, backup_file: str) -> None:
    """Restore database from a backup file."""
    ms: MemoryStore = ctx.obj["store"]
    click.echo(f"Restoring from {backup_file}...")
    ms.restore(backup_file)
    click.echo("Restore complete.")


@cli.command("backups")
@click.option("--dir", "backup_dir", default=None, help="Directory to search")
@click.pass_context
def backups_cmd(ctx: click.Context, backup_dir: str | None) -> None:
    """List available backups."""
    ms: MemoryStore = ctx.obj["store"]
    backups = ms.list_backups(backup_dir=backup_dir)
    if not backups:
        click.echo("No backups found.")
        return
    for b in backups:
        click.echo(json.dumps(b, indent=2))


# --- Staleness ---


@cli.command("stale")
@click.option("--limit", "-l", default=20, type=int, help="Max results")
@click.pass_context
def stale_cmd(ctx: click.Context, limit: int) -> None:
    """Show memories that may be stale or outdated."""
    ms: MemoryStore = ctx.obj["store"]
    stale = ms.get_stale_memories(limit=limit)
    if not stale:
        click.echo("No stale memories found.")
        return
    for s in stale:
        click.echo(json.dumps(s, indent=2))
        click.echo("---")


# --- PII Scanning ---


@cli.command("scan-pii")
@click.option("--limit", "-l", default=100, type=int, help="Max memories to scan")
@click.pass_context
def scan_pii_cmd(ctx: click.Context, limit: int) -> None:
    """Scan memories for PII and secrets (API keys, emails, etc)."""
    ms: MemoryStore = ctx.obj["store"]
    results = ms.scan_pii(limit=limit)
    if not results:
        click.echo("No PII detected.")
        return
    click.echo(f"Found PII in {len(results)} memories:")
    for r in results:
        click.echo(json.dumps(r, indent=2))
        click.echo("---")


# --- Data Retention ---


@cli.command("purge-expired")
@click.option("--dry-run", is_flag=True, help="Show what would be purged without deleting")
@click.pass_context
def purge_expired_cmd(ctx: click.Context, dry_run: bool) -> None:
    """Purge memories that have exceeded their retention policy."""
    ms: MemoryStore = ctx.obj["store"]
    results = ms.purge_expired(dry_run=dry_run)
    if not results:
        click.echo("No expired memories.")
        return
    label = "Would purge" if dry_run else "Purged"
    click.echo(f"{label} {len(results)} memories:")
    for r in results:
        click.echo(json.dumps(r, indent=2))


# --- Feedback ---


@cli.command("feedback")
@click.argument("memory_id")
@click.option("--useful/--not-useful", default=True, help="Was this memory useful?")
@click.pass_context
def feedback_cmd(ctx: click.Context, memory_id: str, useful: bool) -> None:
    """Record feedback on a memory to improve future ranking."""
    ms: MemoryStore = ctx.obj["store"]
    ms.record_feedback(memory_id, useful)
    label = "positive" if useful else "negative"
    click.echo(f"Recorded {label} feedback for {memory_id}")


# --- Trust ---


@cli.command("trust")
@click.argument("memory_id")
@click.pass_context
def trust_cmd(ctx: click.Context, memory_id: str) -> None:
    """Show trust report for a memory."""
    ms: MemoryStore = ctx.obj["store"]
    report = ms.get_trust_report(memory_id)
    click.echo(json.dumps(report, indent=2))


# --- Diff & Timeline ---


@cli.command("diff")
@click.option("--since", "-s", required=True, help="Start time (e.g. 'last week', '3 days ago', '2026-03-01')")
@click.option("--until", "-u", default=None, help="End time (default: now)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of colored text")
@click.pass_context
def diff_cmd(ctx: click.Context, since: str, until: str | None, as_json: bool) -> None:
    """Show what changed in your knowledge between two times."""
    ms: MemoryStore = ctx.obj["store"]

    if as_json:
        result = ms.diff(since=since, until=until)
        click.echo(json.dumps(result, indent=2))
    else:
        from neuropack.diff.engine import MemoryDiffEngine, parse_relative_date
        from neuropack.diff.formatter import format_diff_text

        since_dt = parse_relative_date(since)
        until_dt = parse_relative_date(until) if until else None
        engine = MemoryDiffEngine()
        memory_diff = engine.diff_since(ms, since_dt, until_dt)
        click.echo(format_diff_text(memory_diff))


@cli.command("timeline")
@click.option("--entity", "-e", default=None, help="Filter by entity name")
@click.option("--tag", "-t", default=None, help="Filter by tag")
@click.option("--granularity", "-g", default="day", type=click.Choice(["day", "week", "month"]),
              help="Time grouping")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of ASCII chart")
@click.pass_context
def timeline_cmd(ctx: click.Context, entity: str | None, tag: str | None, granularity: str, as_json: bool) -> None:
    """Show how your knowledge evolved over time."""
    ms: MemoryStore = ctx.obj["store"]

    if as_json:
        entries = ms.knowledge_timeline(entity=entity, tag=tag, granularity=granularity)
        click.echo(json.dumps(entries, indent=2))
    else:
        from neuropack.diff.timeline import build_timeline
        from neuropack.diff.formatter import format_timeline_text

        entries = build_timeline(ms, entity=entity, tag=tag, granularity=granularity)
        click.echo(format_timeline_text(entries))


# --- Developer DNA Profile ---


@cli.command("profile")
@click.option("--rebuild", is_flag=True, help="Force a full profile rebuild")
@click.option("--section", "-s", default=None,
              type=click.Choice(["naming", "architecture", "error_handling", "libraries",
                                 "code_style", "review_feedback", "anti_patterns"]),
              help="Show a specific profile section")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def profile_cmd(ctx: click.Context, rebuild: bool, section: str | None, as_json: bool) -> None:
    """Show your Developer DNA profile built from memory analysis."""
    ms: MemoryStore = ctx.obj["store"]

    # Map short names to full section names
    section_map = {
        "naming": "naming_conventions",
        "architecture": "architecture_patterns",
        "error_handling": "error_handling",
        "libraries": "preferred_libraries",
        "code_style": "code_style",
        "review_feedback": "review_feedback",
        "anti_patterns": "anti_patterns",
    }

    if rebuild:
        data = ms.rebuild_developer_profile()
        if not as_json and not section:
            click.echo("Profile rebuilt successfully.")
    elif section:
        full_section = section_map.get(section, section)
        result = ms.query_coding_style(full_section)
        if as_json:
            click.echo(json.dumps(result, indent=2))
        else:
            _format_profile_section(section, result)
        return
    else:
        data = ms.get_developer_profile()

    if as_json:
        click.echo(json.dumps(data, indent=2))
    else:
        _format_profile_pretty(data)


def _format_profile_pretty(data: dict) -> None:
    """Format profile data for human-readable CLI output."""
    click.echo("\n=== Developer DNA Profile ===\n")

    confidence = data.get("confidence", 0)
    evidence = data.get("evidence_count", 0)
    click.echo(f"  Confidence: {confidence:.0%}  ({evidence} memories analyzed)")
    click.echo(f"  Last updated: {data.get('last_updated', 'never')}\n")

    # Naming conventions
    naming = data.get("naming_conventions", {})
    click.echo("  Naming Conventions:")
    click.echo(f"    Variables: {naming.get('variable_style', 'unknown')}")
    click.echo(f"    Classes:   {naming.get('class_style', 'unknown')}")
    click.echo(f"    Files:     {naming.get('file_style', 'unknown')}")
    prefixes = naming.get("common_prefixes", [])
    if prefixes:
        click.echo(f"    Prefixes:  {', '.join(prefixes[:8])}")
    click.echo()

    # Code style
    style = data.get("code_style", {})
    click.echo("  Code Style:")
    click.echo(f"    Line length: {style.get('line_length_pref', 'unknown')}")
    click.echo(f"    Imports:     {style.get('import_style', 'unknown')}")
    click.echo(f"    Docstrings:  {style.get('docstring_style', 'unknown')}")
    click.echo(f"    Type hints:  {style.get('type_hints_usage', 'unknown')}")
    click.echo()

    # Error handling
    errors = data.get("error_handling", {})
    click.echo("  Error Handling:")
    click.echo(f"    Style: {errors.get('style', 'unknown')}")
    patterns = errors.get("patterns", [])
    for p in patterns[:5]:
        click.echo(f"    - {p}")
    click.echo()

    # Architecture patterns
    arch = data.get("architecture_patterns", [])
    if arch:
        click.echo("  Architecture Patterns:")
        for a in arch:
            click.echo(f"    - {a}")
        click.echo()

    # Preferred libraries
    libs = data.get("preferred_libraries", {})
    if libs:
        click.echo("  Top Libraries:")
        sorted_libs = sorted(libs.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)
        for name, info in sorted_libs[:10]:
            freq = info.get("frequency", 0)
            click.echo(f"    {name}: used {freq}x")
        click.echo()

    # Review feedback themes
    feedback = data.get("review_feedback", [])
    if feedback:
        click.echo("  Review Themes:")
        for f in feedback:
            click.echo(f"    - {f}")
        click.echo()

    # Anti-patterns
    anti = data.get("anti_patterns", [])
    if anti:
        click.echo("  Things You Avoid:")
        for a in anti[:8]:
            click.echo(f"    - {a}")
        click.echo()


def _format_profile_section(section: str, result: dict) -> None:
    """Format a single profile section for CLI output."""
    if "error" in result:
        click.echo(result["error"], err=True)
        return

    click.echo(f"\n=== {section.replace('_', ' ').title()} ===\n")
    click.echo(f"  Confidence: {result.get('confidence', 0):.0%}")
    click.echo(f"  Evidence: {result.get('evidence_count', 0)} memories\n")

    data = result.get("data", {})
    if isinstance(data, list):
        for item in data:
            click.echo(f"  - {item}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                click.echo(f"  {key}:")
                for item in value[:10]:
                    click.echo(f"    - {item}")
            elif isinstance(value, dict):
                click.echo(f"  {key}:")
                for k, v in value.items():
                    click.echo(f"    {k}: {v}")
            else:
                click.echo(f"  {key}: {value}")
    click.echo()


from neuropack.cli.agents import agent_group  # noqa: E402
from neuropack.cli.workspace import workspace_group  # noqa: E402

cli.add_command(agent_group)
cli.add_command(workspace_group)


# --- Anticipatory Context Watcher ---


@cli.command("watch")
@click.argument("directories", nargs=-1)
@click.option("--daemon", "as_daemon", is_flag=True, help="Run in background (no output)")
@click.pass_context
def watch_cmd(ctx: click.Context, directories: tuple[str, ...], as_daemon: bool) -> None:
    """Start the anticipatory context watcher.

    Monitors file changes, git activity, and terminal commands to pre-load
    relevant memories before you need them.
    """
    import time

    ms: MemoryStore = ctx.obj["store"]
    dirs = list(directories) if directories else None

    ms.start_watcher(directories=dirs)
    status = ms.watcher_status()
    click.echo(f"Watcher started, monitoring: {', '.join(status['directories']) or '(none)'}")

    if as_daemon:
        click.echo("Running in background. Use 'np anticipate' to see pre-loaded context.")
        # Keep process alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            click.echo("\nStopping watcher...")
            ms.stop_watcher()
            click.echo("Watcher stopped.")
        return

    # Foreground mode: print events and recalls as they happen
    def on_event(event):
        click.echo(f"  [{event.type}] {event.path or ''} {event.metadata}")

    def on_recall(query, results):
        click.echo(f"  >> Recall: {query!r} -> {len(results)} results")
        for r in results[:3]:
            click.echo(f"     - {r['l3_abstract'][:80]}")

    if ms._watcher is not None:
        ms._watcher._on_event = on_event
        ms._watcher._on_recall = on_recall

    click.echo("Watching... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
        ms.stop_watcher()
        click.echo("Watcher stopped.")


@cli.command("anticipate")
@click.option("--budget", "-b", default=4000, type=int, help="Token budget for context")
@click.pass_context
def anticipate_cmd(ctx: click.Context, budget: int) -> None:
    """Show what the anticipatory watcher has pre-loaded."""
    ms: MemoryStore = ctx.obj["store"]
    status = ms.watcher_status()

    if not status["running"]:
        click.echo("Watcher is not running. Start with: np watch [DIRECTORIES...]")
        return

    click.echo(json.dumps({"watcher_status": status}, indent=2))
    click.echo("---")

    context = ms.get_anticipatory_context(token_budget=budget)
    if not context:
        click.echo("No pre-loaded context yet. The watcher needs activity to observe.")
        return

    click.echo(f"Pre-loaded context ({len(context)} items, budget={budget} tokens):")
    for item in context:
        click.echo(json.dumps({
            "id": item.get("id", ""),
            "l3": item.get("l3_abstract", ""),
            "tags": item.get("tags", []),
            "score": item.get("score"),
            "query": item.get("query", ""),
        }, indent=2))
        click.echo("---")


# --- LongMemEval Benchmark ---


@cli.command("benchmark")
@click.option("--variant", "-v", default="s", type=click.Choice(["s", "oracle"]),
              help="Dataset variant to use")
@click.option("--model", "-m", default="gpt-4o", help="LLM model for answer generation and judging")
@click.option("--data-dir", default="", help="Directory to store/load benchmark data")
@click.option("--skip-ingest", is_flag=True, help="Skip ingestion (use if already ingested)")
@click.option("--output", "-o", default="", help="Save results JSON to this path")
@click.option("--estimate", is_flag=True, help="Run offline score estimator (no API costs)")
@click.option("--limit", default=50, type=int, help="Number of questions for --estimate mode")
@click.option("--context-injection", is_flag=True, help="Mastra-style: inject all sessions into context (no retrieval)")
@click.option("--judge-model", default="", help="Model for judging (default: same as --model). Use gpt-4o for official scoring.")
@click.option("--observations", is_flag=True, help="Extract structured observations from sessions before answering (heuristic, zero cost)")
@click.option("--observations-llm", is_flag=True, help="Use LLM for observation extraction instead of heuristics (costs ~$2-3)")
@click.option("--extraction-model", default="", help="Model for observation extraction (default: same as --model). Supports ollama:model_name for local extraction.")
@click.pass_context
def benchmark_cmd(ctx: click.Context, variant: str, model: str, data_dir: str,
                  skip_ingest: bool, output: str, estimate: bool, limit: int,
                  context_injection: bool, judge_model: str,
                  observations: bool, observations_llm: bool,
                  extraction_model: str) -> None:
    """Run the LongMemEval benchmark against ReCall.

    Downloads the dataset (if needed), ingests conversation sessions,
    generates answers via LLM, and scores them with LLM-as-judge.

    Use --estimate for a free offline score estimate based on retrieval quality.

    Requires OPENAI_API_KEY environment variable to be set (unless --estimate).
    """
    import os

    from neuropack.benchmark import LongMemEvalRunner, format_benchmark_text, format_benchmark_json

    ms: MemoryStore = ctx.obj["store"]
    runner = LongMemEvalRunner(store=ms, data_dir=data_dir)

    if estimate:
        # Offline estimation mode -- no API key required
        click.echo("Running offline score estimation (no API costs)...")
        click.echo(f"  Variant: {variant}  Limit: {limit}")
        click.echo("")

        try:
            runner.download_data()
            data = runner.load_data(variant=variant)
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        estimate_result = runner.estimate_score(data, limit=limit)

        click.echo(f"Offline Score Estimate ({estimate_result['total_questions']} questions)")
        click.echo("=" * 55)
        click.echo(f"  Overall estimate:  {estimate_result['overall_estimate']:.1f}%")
        click.echo(f"  Recall hits:       {estimate_result['recall_hits']}")
        click.echo(f"  Recall misses:     {estimate_result['recall_misses']}")
        click.echo("")
        click.echo("  Per-category estimates:")
        for cat, score in sorted(estimate_result["category_scores"].items()):
            click.echo(f"    {cat:30s} {score:5.1f}%")
        click.echo("")
        click.echo(f"  Note: {estimate_result['note']}")

        if output:
            from pathlib import Path

            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(estimate_result, indent=2), encoding="utf-8")
            click.echo(f"Results saved to {out_path}")

        return

    # Full benchmark mode -- requires API key (unless using local Ollama models)
    _model_is_local = model.startswith("ollama:") or model.startswith("local:")
    if not os.environ.get("OPENAI_API_KEY") and not _model_is_local:
        click.echo("Error: OPENAI_API_KEY environment variable is not set.", err=True)
        sys.exit(1)

    _progress_start: dict[str, float] = {}

    def progress(stage: str, current: int, total: int) -> None:
        if total <= 0:
            return
        import time as _t
        if stage not in _progress_start:
            _progress_start[stage] = _t.time()
        pct = current / total * 100
        elapsed = _t.time() - _progress_start[stage]
        if current > 0 and elapsed > 5:
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            eta = f" ETA: {mins}m{secs:02d}s" if mins > 0 else f" ETA: {secs}s"
        else:
            eta = ""
        click.echo(f"\r  [{stage}] {current}/{total} ({pct:.0f}%){eta}", nl=(current == total))

    if observations or observations_llm:
        mode = "observations-llm" if observations_llm else "observations"
    elif context_injection:
        mode = "context-injection"
    elif skip_ingest:
        mode = "skip-ingest"
    else:
        mode = "full"
    click.echo("Starting LongMemEval benchmark...")
    click.echo(f"  Variant: {variant}  Model: {model}  Mode: {mode}")
    click.echo("")

    try:
        result = runner.run_full_benchmark(
            variant=variant,
            model=model,
            judge_model=judge_model,
            skip_ingest=skip_ingest,
            context_injection=context_injection,
            observations=observations or observations_llm,
            observations_llm=observations_llm,
            extraction_model=extraction_model or model,
            progress_callback=progress,
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Print formatted results
    click.echo(format_benchmark_text(result))

    # Print re-retrieval stats from agentic 2-step
    re_retrieval_count = sum(
        1 for d in result.details if d.get("needed_re_retrieval", False)
    )
    if re_retrieval_count > 0:
        click.echo(f"\n  Agentic re-retrieval: {re_retrieval_count}/{result.total_questions} questions needed a second retrieval pass")

    # Save JSON if requested
    if output:
        from pathlib import Path

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(format_benchmark_json(result), indent=2), encoding="utf-8")
        click.echo(f"Results saved to {out_path}")


# --- Shell Integration ---


@cli.command("shell-init")
@click.option("--shell", "shell_name", default="auto",
              type=click.Choice(["bash", "zsh", "powershell", "auto"]),
              help="Shell type (auto-detect if omitted)")
@click.pass_context
def shell_init_cmd(ctx: click.Context, shell_name: str) -> None:
    """Print shell hook script for command capture.

    Usage:
      eval "$(np shell-init)"                         # auto-detect
      eval "$(np shell-init --shell bash)"             # bash
      np shell-init --shell powershell >> $PROFILE     # PowerShell
    """
    from neuropack.integrations.shell import (
        generate_bash_hook,
        generate_powershell_hook,
        generate_zsh_hook,
    )

    if shell_name == "auto":
        import os

        parent_shell = os.environ.get("SHELL", "")
        if "zsh" in parent_shell:
            shell_name = "zsh"
        elif "bash" in parent_shell:
            shell_name = "bash"
        elif os.environ.get("PSModulePath"):
            shell_name = "powershell"
        else:
            shell_name = "bash"

    generators = {
        "bash": generate_bash_hook,
        "zsh": generate_zsh_hook,
        "powershell": generate_powershell_hook,
    }
    hook = generators[shell_name]()
    click.echo(hook, nl=False)


@cli.command("shell-log")
@click.argument("command")
@click.argument("exit_code", default=0, type=int)
@click.pass_context
def shell_log_cmd(ctx: click.Context, command: str, exit_code: int) -> None:
    """Log a shell command into ReCall memory (called by shell hook)."""
    import os

    from neuropack.integrations.shell import log_command

    ms: MemoryStore = ctx.obj["store"]
    log_command(
        command=command,
        exit_code=exit_code,
        cwd=os.getcwd(),
        store=ms,
    )


@cli.command("shell-search")
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int, help="Max results")
@click.pass_context
def shell_search_cmd(ctx: click.Context, query: str, limit: int) -> None:
    """Search past shell commands stored in memory.

    Example: np shell-search "docker compose"
    """
    from neuropack.integrations.shell import search_commands

    ms: MemoryStore = ctx.obj["store"]
    results = search_commands(query=query, store=ms, limit=limit)
    if not results:
        click.echo("No matching shell commands found.")
        return
    for r in results:
        click.echo(json.dumps({
            "id": r.record.id,
            "score": round(r.score, 4),
            "content": r.record.content,
            "tags": r.record.tags,
            "source": r.record.source,
            "created": r.record.created_at.isoformat(),
        }, indent=2))
        click.echo("---")


# --- Git Hooks Integration ---


@cli.group("git")
def git_group() -> None:
    """Git hooks integration for auto-capturing commits, merges, and checkouts."""
    pass


@git_group.command("install")
@click.argument("path", default=".")
@click.option(
    "--hooks",
    default=None,
    help="Comma-separated hook types to install (default: post-commit,post-merge,post-checkout)",
)
@click.pass_context
def git_install(ctx: click.Context, path: str, hooks: str | None) -> None:
    """Install ReCall git hooks in a repository."""
    import os
    from pathlib import Path

    from neuropack.integrations.git_hooks import install_hooks

    repo_path = str(Path(path).resolve())
    hook_list = hooks.split(",") if hooks else None

    try:
        installed = install_hooks(repo_path, hooks=hook_list)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if installed:
        click.echo(f"Installed {len(installed)} ReCall hook(s):")
        for h in installed:
            click.echo(f"  {h}")
    else:
        click.echo("No hooks installed.")


@git_group.command("uninstall")
@click.argument("path", default=".")
@click.pass_context
def git_uninstall(ctx: click.Context, path: str) -> None:
    """Remove ReCall git hooks from a repository."""
    from pathlib import Path

    from neuropack.integrations.git_hooks import uninstall_hooks

    repo_path = str(Path(path).resolve())

    try:
        modified = uninstall_hooks(repo_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if modified:
        click.echo(f"Removed ReCall from {len(modified)} hook(s):")
        for h in modified:
            click.echo(f"  {h}")
    else:
        click.echo("No ReCall hooks found to remove.")


@git_group.command("capture")
@click.argument("hook_type", type=click.Choice(["post-commit", "post-merge", "post-checkout"]))
@click.pass_context
def git_capture(ctx: click.Context, hook_type: str) -> None:
    """Capture git context into ReCall memory (called by hooks)."""
    import os

    from neuropack.integrations.git_hooks import (
        capture_post_checkout,
        capture_post_commit,
        capture_post_merge,
    )

    ms: MemoryStore = ctx.obj["store"]
    repo_path = os.getcwd()

    try:
        if hook_type == "post-commit":
            capture_post_commit(repo_path, store=ms)
        elif hook_type == "post-merge":
            capture_post_merge(repo_path, store=ms)
        elif hook_type == "post-checkout":
            capture_post_checkout(repo_path, store=ms)
    except Exception as e:
        # Never block a git operation -- log and exit cleanly
        click.echo(f"ReCall capture warning: {e}", err=True)


@git_group.command("status")
@click.argument("path", default=".")
@click.pass_context
def git_status(ctx: click.Context, path: str) -> None:
    """Show which ReCall hooks are installed in a repository."""
    from pathlib import Path

    from neuropack.integrations.git_hooks import get_installed_hooks

    repo_path = str(Path(path).resolve())

    try:
        installed = get_installed_hooks(repo_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if installed:
        click.echo(f"ReCall hooks installed in {repo_path}:")
        for h in installed:
            click.echo(f"  {h}")
    else:
        click.echo(f"No ReCall hooks installed in {repo_path}")


# --- LLM Proxy ---


@cli.command("proxy")
@click.option("--port", default=8741, type=int, help="Port for the proxy server")
@click.option("--provider", default="auto", type=click.Choice(["openai", "anthropic", "auto"]),
              help="Which provider to proxy for")
@click.option("--target-url", default="", help="Override target API base URL")
@click.option("--tags", default="", help="Extra tags to add (comma-separated)")
@click.pass_context
def proxy_cmd(ctx: click.Context, port: int, provider: str, target_url: str, tags: str) -> None:
    """Start an LLM proxy server that captures all calls into ReCall.

    Point your apps at http://localhost:PORT instead of api.openai.com
    and all LLM conversations are automatically stored as memories.
    """
    import uvicorn

    from neuropack.proxy.server import ProxyServer

    config: NeuropackConfig = ctx.obj["config"]
    store: MemoryStore = ctx.obj["store"]

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    # Also include default tags from config
    default_tags = [t.strip() for t in config.proxy_default_tags.split(",") if t.strip()]
    if default_tags:
        tag_list = (tag_list or []) + default_tags

    server = ProxyServer(
        store=store,
        provider=provider,
        target_url=target_url,
        tags=tag_list,
        config=config,
    )

    click.echo(f"\n  ReCall LLM Proxy starting on port {port}")
    click.echo(f"  Provider: {provider}")
    if target_url:
        click.echo(f"  Target URL: {target_url}")
    click.echo()
    click.echo("  To use with OpenAI SDK:")
    click.echo(f"    client = OpenAI(base_url=\"http://localhost:{port}/v1\")")
    click.echo()
    click.echo("  To use with Anthropic SDK:")
    click.echo(f"    client = Anthropic(base_url=\"http://localhost:{port}\")")
    click.echo()
    click.echo("  Or set environment variables:")
    click.echo(f"    OPENAI_BASE_URL=http://localhost:{port}/v1")
    click.echo(f"    ANTHROPIC_BASE_URL=http://localhost:{port}")
    click.echo()
    click.echo(f"  Health check: http://localhost:{port}/health")
    click.echo()

    uvicorn.run(server.app, host="127.0.0.1", port=port)


# --- Setup Commands ---


@cli.group("setup", invoke_without_command=True)
@click.pass_context
def setup_group(ctx: click.Context) -> None:
    """Auto-configure integrations with one command."""
    if ctx.invoked_subcommand is None:
        click.echo("Available setup commands:")
        click.echo("  rc setup claude    - Configure Claude Desktop MCP connection")
        click.echo("  rc setup cursor    - Configure Cursor MCP connection")
        click.echo("  rc setup shell     - Configure shell integration")
        click.echo("  rc setup git       - Install git hooks in current repo")
        click.echo("  rc setup ollama    - Configure local LLM via Ollama")
        click.echo("  rc setup all       - Run all setup commands")
        click.echo()
        click.echo("Run any command for automatic, zero-argument setup.")


def _get_mcp_entry() -> dict:
    """Return the standard MCP server entry for ReCall."""
    from pathlib import Path

    home = str(Path.home()).replace("\\", "/")
    return {
        "command": "python",
        "args": ["-m", "neuropack.mcp_server.server"],
        "env": {
            "NEUROPACK_DB_PATH": f"{home}/.neuropack/memories.db",
            "NEUROPACK_EMBEDDER_TYPE": "tfidf",
        },
    }


def _update_mcp_config(config_path: str, display_name: str) -> bool:
    """Read/create an MCP config file and add/update the 'recall' server entry.

    Returns True on success, False on failure.
    """
    from pathlib import Path

    path = Path(config_path)
    mcp_entry = _get_mcp_entry()

    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, OSError) as e:
            click.echo(click.style(f"  Error reading {path}: {e}", fg="red"))
            return False
    else:
        data = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in data:
        data["mcpServers"] = {}

    existing = "recall" in data["mcpServers"]
    data["mcpServers"]["recall"] = mcp_entry

    # Write back
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        click.echo(click.style(f"  Error writing {path}: {e}", fg="red"))
        return False

    action = "Updated" if existing else "Added"
    click.echo(click.style(f"  {action} 'recall' MCP server in {path}", fg="green"))
    return True


@setup_group.command("claude")
@click.pass_context
def setup_claude(ctx: click.Context) -> None:
    """Auto-configure Claude Desktop MCP connection."""
    import os

    click.echo(click.style("\nSetting up Claude Desktop integration...", fg="cyan", bold=True))

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            click.echo(click.style("  Could not find APPDATA directory.", fg="red"))
            return
        config_path = os.path.join(appdata, "Claude", "claude_desktop_config.json")
    elif sys.platform == "darwin":
        config_path = os.path.expanduser(
            "~/Library/Application Support/Claude/claude_desktop_config.json"
        )
    else:
        # Linux: try XDG config
        xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        config_path = os.path.join(xdg, "Claude", "claude_desktop_config.json")

    if _update_mcp_config(config_path, "Claude Desktop"):
        click.echo()
        click.echo(click.style("  Done!", fg="green", bold=True))
        click.echo("  Restart Claude Desktop for changes to take effect.")
    click.echo()


@setup_group.command("cursor")
@click.pass_context
def setup_cursor(ctx: click.Context) -> None:
    """Auto-configure Cursor MCP connection."""
    import os
    from pathlib import Path

    click.echo(click.style("\nSetting up Cursor integration...", fg="cyan", bold=True))

    candidates = []

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates.append(os.path.join(
                appdata, "Cursor", "User", "globalStorage",
                "saoudrizwan.claude-dev", "settings", "cline_mcp_settings.json"
            ))
        home = Path.home()
        candidates.append(str(home / ".cursor" / "mcp.json"))
    elif sys.platform == "darwin":
        home = Path.home()
        candidates.append(str(
            home / "Library" / "Application Support" / "Cursor" / "User" /
            "globalStorage" / "saoudrizwan.claude-dev" / "settings" /
            "cline_mcp_settings.json"
        ))
        candidates.append(str(home / ".cursor" / "mcp.json"))
    else:
        home = Path.home()
        candidates.append(str(home / ".cursor" / "mcp.json"))

    # Use the first existing path, or fall back to ~/.cursor/mcp.json
    config_path = None
    for c in candidates:
        if Path(c).exists():
            config_path = c
            break
    if config_path is None:
        config_path = candidates[-1]  # default to ~/.cursor/mcp.json

    if _update_mcp_config(config_path, "Cursor"):
        click.echo()
        click.echo(click.style("  Done!", fg="green", bold=True))
        click.echo("  Restart Cursor for changes to take effect.")
    click.echo()


@setup_group.command("shell")
@click.pass_context
def setup_shell(ctx: click.Context) -> None:
    """Auto-configure shell integration (bash/zsh/PowerShell)."""
    import os
    from pathlib import Path

    click.echo(click.style("\nSetting up shell integration...", fg="cyan", bold=True))

    # Detect shell
    if sys.platform == "win32":
        # On Windows, check for PSModulePath first (PowerShell), then fall back to bash
        if os.environ.get("PSModulePath"):
            shell_type = "powershell"
        else:
            parent_shell = os.environ.get("SHELL", "")
            if "zsh" in parent_shell:
                shell_type = "zsh"
            elif "bash" in parent_shell:
                shell_type = "bash"
            else:
                shell_type = "powershell"
    else:
        parent_shell = os.environ.get("SHELL", "")
        if "zsh" in parent_shell:
            shell_type = "zsh"
        elif "bash" in parent_shell:
            shell_type = "bash"
        else:
            shell_type = "bash"

    click.echo(f"  Detected shell: {shell_type}")

    hook_line_map = {
        "bash": ('eval "$(np shell-init)"', str(Path.home() / ".bashrc")),
        "zsh": ('eval "$(np shell-init)"', str(Path.home() / ".zshrc")),
        "powershell": (
            'Invoke-Expression (np shell-init --shell powershell)',
            _get_powershell_profile(),
        ),
    }

    hook_line, rc_file = hook_line_map[shell_type]
    rc_path = Path(rc_file)

    # Check if already configured
    if rc_path.exists():
        content = rc_path.read_text(encoding="utf-8", errors="replace")
        if hook_line in content or "np shell-init" in content:
            click.echo(click.style("  Already configured!", fg="yellow"))
            click.echo(f"  Found shell hook in {rc_path}")
            click.echo()
            return

    # Append the hook line
    try:
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rc_path, "a", encoding="utf-8") as f:
            f.write(f"\n# ReCall (NeuroPack) shell integration\n{hook_line}\n")
    except OSError as e:
        click.echo(click.style(f"  Error writing to {rc_path}: {e}", fg="red"))
        return

    click.echo(click.style(f"  Added shell hook to {rc_path}", fg="green"))
    click.echo()
    click.echo(click.style("  Done!", fg="green", bold=True))
    if shell_type == "powershell":
        click.echo("  Restart PowerShell or run: . $PROFILE")
    else:
        click.echo(f"  Restart your shell or run: source {rc_path}")
    click.echo()


def _get_powershell_profile() -> str:
    """Get the PowerShell profile path."""
    import os
    import subprocess

    # Try to get $PROFILE from PowerShell
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "echo $PROFILE"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: standard location
    if sys.platform == "win32":
        docs = os.path.join(os.environ.get("USERPROFILE", ""), "Documents",
                            "WindowsPowerShell", "Microsoft.PowerShell_profile.ps1")
    else:
        docs = os.path.expanduser("~/.config/powershell/Microsoft.PowerShell_profile.ps1")
    return docs


@setup_group.command("git")
@click.pass_context
def setup_git(ctx: click.Context) -> None:
    """Auto-configure git hooks in current repository."""
    import os
    from pathlib import Path

    from neuropack.integrations.git_hooks import install_hooks

    click.echo(click.style("\nSetting up git hooks...", fg="cyan", bold=True))

    repo_path = os.getcwd()
    git_dir = Path(repo_path) / ".git"

    if not git_dir.exists():
        click.echo(click.style("  Not a git repository.", fg="red"))
        click.echo("  Run this command from the root of a git repo.")
        click.echo()
        return

    try:
        installed = install_hooks(repo_path)
    except FileNotFoundError as e:
        click.echo(click.style(f"  Error: {e}", fg="red"))
        return

    if installed:
        click.echo(click.style(f"  Installed {len(installed)} hook(s):", fg="green"))
        for h in installed:
            click.echo(f"    - {h}")
    else:
        click.echo(click.style("  Hooks already installed.", fg="yellow"))

    click.echo()
    click.echo(click.style("  Done!", fg="green", bold=True))
    click.echo("  Commits, merges, and checkouts will now be captured as memories.")
    click.echo()


@setup_group.command("ollama")
@click.pass_context
def setup_ollama(ctx: click.Context) -> None:
    """Auto-configure local LLM via Ollama."""
    import subprocess

    click.echo(click.style("\nSetting up Ollama integration...", fg="cyan", bold=True))

    # Check if Ollama is running
    ollama_running = False
    tags_data = {}
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags_data = json.loads(resp.read().decode("utf-8"))
            ollama_running = True
    except Exception:
        pass

    if not ollama_running:
        click.echo(click.style("  Ollama is not running.", fg="yellow"))
        click.echo()
        click.echo("  Install Ollama:")
        if sys.platform == "win32":
            click.echo("    Download from https://ollama.com/download/windows")
        elif sys.platform == "darwin":
            click.echo("    brew install ollama")
            click.echo("    or download from https://ollama.com/download/mac")
        else:
            click.echo("    curl -fsSL https://ollama.com/install.sh | sh")
        click.echo()
        click.echo("  Then start it with: ollama serve")
        click.echo("  And re-run: rc setup ollama")
        click.echo()
        return

    click.echo(click.style("  Ollama is running!", fg="green"))

    # Check installed models
    models = tags_data.get("models", [])
    model_names = [m.get("name", "") for m in models]
    click.echo(f"  Installed models: {', '.join(model_names) if model_names else '(none)'}")

    recommended = "qwen3:8b"
    has_recommended = any(recommended.split(":")[0] in m for m in model_names)

    if not model_names or not has_recommended:
        click.echo()
        if click.confirm(f"  Pull recommended model '{recommended}'?", default=True):
            click.echo(f"  Pulling {recommended} (this may take a few minutes)...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", recommended],
                    timeout=600,
                )
                if result.returncode == 0:
                    click.echo(click.style(f"  Successfully pulled {recommended}", fg="green"))
                    model_names.append(recommended)
                else:
                    click.echo(click.style(
                        "  Pull failed. Try manually: ollama pull " + recommended, fg="red"
                    ))
                    click.echo()
                    return
            except FileNotFoundError:
                click.echo(click.style("  'ollama' command not found in PATH.", fg="red"))
                click.echo()
                return
            except subprocess.TimeoutExpired:
                click.echo(click.style(
                    "  Pull timed out. Try manually: ollama pull " + recommended, fg="yellow"
                ))
                click.echo()
                return

    # Register as LLM provider
    use_model = recommended if recommended in model_names else (
        model_names[0] if model_names else recommended
    )
    ms: MemoryStore = ctx.obj["store"]
    try:
        from neuropack.llm.models import LLMConfig

        config = LLMConfig(
            name="ollama-local",
            provider="ollama",
            api_key="",
            model=use_model,
            base_url="http://localhost:11434",
            is_default=False,
        )
        ms._llm_registry.add(config)
        click.echo(click.style(f"  Registered 'ollama-local' with model {use_model}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"  Warning: Could not register LLM provider: {e}", fg="yellow"))

    click.echo()
    click.echo(click.style("  Done!", fg="green", bold=True))
    click.echo(f"  Ollama is ready with model: {use_model}")
    click.echo("  Set as default with: rc llm set-default ollama-local")
    click.echo()


@setup_group.command("all")
@click.pass_context
def setup_all(ctx: click.Context) -> None:
    """Run all setup commands in sequence."""
    click.echo(click.style("\n=== ReCall Setup (All Integrations) ===", fg="cyan", bold=True))
    click.echo()

    commands = [
        ("Claude Desktop", setup_claude),
        ("Cursor", setup_cursor),
        ("Shell", setup_shell),
        ("Git Hooks", setup_git),
        ("Ollama", setup_ollama),
    ]

    for name, cmd_fn in commands:
        click.echo(click.style(f"--- {name} ---", fg="white", bold=True))
        try:
            ctx.invoke(cmd_fn)
        except SystemExit:
            pass
        except Exception as e:
            click.echo(click.style(f"  Skipped ({e})", fg="yellow"))
            click.echo()

    click.echo(click.style("=== Setup Complete ===", fg="green", bold=True))
    click.echo()


if __name__ == "__main__":
    cli()

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from neuropack.api.schemas import (
    AgentLogRequest,
    BatchStoreRequest,
    ChatRequest,
    ChatResponse,
    ClaimTaskRequest,
    CompleteTaskRequest,
    CreateTaskRequest,
    CreateWorkspaceRequest,
    DeveloperProfileResponse,
    DiffRequest,
    DiffResponse,
    ExportRequest,
    FetchDetailsRequest,
    GenerateContextRequest,
    HealthResponse,
    ImportRequest,
    LLMTestRequest,
    MemoryDetail,
    MemoryListItem,
    ProfileSectionResponse,
    RecallRequest,
    RecallResponse,
    RecallResultItem,
    SessionSummaryRequest,
    ShareRequest,
    StatsResponse,
    StoreRequest,
    StoreResponse,
    TimelineEntryResponse,
    TokenStatsResponse,
    TrainingExportRequest,
    UpdateRequest,
    WatcherStartRequest,
    WatcherStatusResponse,
    AnticipatoryContextItem,
    AnticipatoryContextResponse,
)
from neuropack._version import __version__
from neuropack.core.privacy import strip_private_from_preview
from neuropack.core.store import MemoryStore
from neuropack.exceptions import ContentTooLargeError, MemoryNotFoundError, ValidationError

_DASHBOARD_HTML: str | None = None
_MOBILE_HTML: str | None = None


def _get_dashboard_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        path = Path(__file__).parent / "dashboard.html"
        _DASHBOARD_HTML = path.read_text(encoding="utf-8")
    return _DASHBOARD_HTML


def _get_mobile_html() -> str:
    global _MOBILE_HTML
    if _MOBILE_HTML is None:
        path = Path(__file__).parent / "mobile.html"
        _MOBILE_HTML = path.read_text(encoding="utf-8")
    return _MOBILE_HTML


def get_store(request: Request) -> MemoryStore:
    return request.app.state.store


# Public router (no auth)
public_router = APIRouter()

# Protected router (auth required, dependency added in app.py)
router = APIRouter(prefix="/v1")


@public_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the web dashboard SPA."""
    html = _get_dashboard_html()
    config = request.app.state.config
    html = html.replace("__NP_AUTH_TOKEN__", config.auth_token or "")
    return HTMLResponse(content=html)


@public_router.get("/health", response_model=HealthResponse)
async def health(store: MemoryStore = Depends(get_store)):
    return HealthResponse(
        status="ok",
        version=__version__,
        memory_count=store.stats().total_memories,
    )


@router.post("/memories", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(body: StoreRequest, store: MemoryStore = Depends(get_store)):
    try:
        record = store.store(
            content=body.content,
            tags=body.tags,
            source=body.source,
            priority=body.priority,
            l3_override=body.l3,
            l2_override=body.l2,
            namespace=body.namespace,
        )
    except ContentTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return StoreResponse(
        id=record.id,
        l3_abstract=record.l3_abstract,
        l2_facts=record.l2_facts,
        tags=record.tags,
        created_at=record.created_at.isoformat(),
        namespace=record.namespace,
    )


@router.post("/memories/batch", response_model=list[StoreResponse], status_code=status.HTTP_201_CREATED)
async def batch_store(body: BatchStoreRequest, store: MemoryStore = Depends(get_store)):
    results = []
    for item in body.items:
        try:
            record = store.store(
                content=item.content,
                tags=item.tags,
                source=item.source,
                priority=item.priority,
                l3_override=item.l3,
                l2_override=item.l2,
                namespace=item.namespace,
            )
            results.append(StoreResponse(
                id=record.id,
                l3_abstract=record.l3_abstract,
                l2_facts=record.l2_facts,
                tags=record.tags,
                created_at=record.created_at.isoformat(),
                namespace=record.namespace,
            ))
        except ContentTooLargeError as e:
            raise HTTPException(status_code=413, detail=str(e))
    return results


@router.post("/recall", response_model=RecallResponse)
async def recall_memories(body: RecallRequest, store: MemoryStore = Depends(get_store)):
    # Time-travel recall: return memories as they existed at a past point in time
    if body.as_of:
        as_of_results = store.recall_as_of(
            query=body.query, as_of=body.as_of, limit=body.limit,
        )
        items = [
            RecallResultItem(
                id=r["id"],
                l3_abstract=r["l3_abstract"],
                l2_facts=[],
                content_preview="",
                score=r.get("score", 0.0),
                tags=r.get("tags", []),
                namespace="",
            )
            for r in as_of_results
        ]
        return RecallResponse(results=items, count=len(items))

    namespaces = [body.namespace] if body.namespace else None
    results = store.recall(
        query=body.query,
        limit=body.limit,
        tags=body.tags,
        min_score=body.min_score,
        namespaces=namespaces,
    )
    items = [
        RecallResultItem(
            id=r.record.id,
            l3_abstract=r.record.l3_abstract,
            l2_facts=r.record.l2_facts,
            content_preview=strip_private_from_preview(r.record.content[:300])[:200],
            score=round(r.score, 4),
            tags=r.record.tags,
            vec_score=round(r.vec_score, 4) if r.vec_score is not None else None,
            fts_rank=round(r.fts_rank, 4) if r.fts_rank is not None else None,
            content_tokens=r.record.content_tokens,
            compressed_tokens=r.record.compressed_tokens,
            namespace=r.record.namespace,
        )
        for r in results
    ]
    return RecallResponse(results=items, count=len(items))


@router.get("/memories", response_model=list[MemoryListItem])
async def list_memories(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    tag: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    records = store.list(limit=limit, offset=offset, tag=tag, namespace=namespace)
    return [
        MemoryListItem(
            id=r.id,
            l3_abstract=r.l3_abstract,
            tags=r.tags,
            priority=r.priority,
            created_at=r.created_at.isoformat(),
            namespace=r.namespace,
        )
        for r in records
    ]


@router.get("/memories/{memory_id}", response_model=MemoryDetail)
async def get_memory(memory_id: str, store: MemoryStore = Depends(get_store)):
    record = store.get(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryDetail(
        id=record.id,
        l3_abstract=record.l3_abstract,
        l2_facts=record.l2_facts,
        content=record.content,
        tags=record.tags,
        source=record.source,
        priority=record.priority,
        access_count=record.access_count,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        content_tokens=record.content_tokens,
        compressed_tokens=record.compressed_tokens,
        namespace=record.namespace,
    )


@router.patch("/memories/{memory_id}", response_model=MemoryDetail)
async def update_memory(
    memory_id: str, body: UpdateRequest, store: MemoryStore = Depends(get_store)
):
    try:
        record = store.update(
            memory_id=memory_id,
            content=body.content,
            tags=body.tags,
            priority=body.priority,
            source=body.source,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except ContentTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return MemoryDetail(
        id=record.id,
        l3_abstract=record.l3_abstract,
        l2_facts=record.l2_facts,
        content=record.content,
        tags=record.tags,
        source=record.source,
        priority=record.priority,
        access_count=record.access_count,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        content_tokens=record.content_tokens,
        compressed_tokens=record.compressed_tokens,
        namespace=record.namespace,
    )


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, store: MemoryStore = Depends(get_store)):
    deleted = store.forget(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "id": memory_id}


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    s = store.stats(namespace=namespace)
    return StatsResponse(
        total_memories=s.total_memories,
        total_size_bytes=s.total_size_bytes,
        avg_compression_ratio=round(s.avg_compression_ratio, 2),
        oldest=s.oldest.isoformat() if s.oldest else None,
        newest=s.newest.isoformat() if s.newest else None,
        total_content_tokens=s.total_content_tokens,
        total_compressed_tokens=s.total_compressed_tokens,
        token_savings_ratio=s.token_savings_ratio,
    )


@router.get("/token-stats", response_model=TokenStatsResponse)
async def get_token_stats(store: MemoryStore = Depends(get_store)):
    data = store.token_stats()
    return TokenStatsResponse(**data)


@router.get("/context-summary")
async def context_summary(
    limit: int = Query(default=50, ge=1, le=200),
    tag: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    tags = [tag] if tag else None
    return store.context_summary(limit=limit, tags=tags, namespace=namespace)


@router.post("/fetch-details")
async def fetch_details(
    body: FetchDetailsRequest, store: MemoryStore = Depends(get_store)
):
    return store.fetch_details(body.memory_ids)


@router.post("/session-summary")
async def api_session_summary(
    body: SessionSummaryRequest, store: MemoryStore = Depends(get_store)
):
    summary = store.session_summary(body.memory_ids)
    result = {"summary": summary}
    if body.store_as_memory:
        record = store.store_session_summary(body.memory_ids)
        result["stored_id"] = record.id
    return result


@router.post("/generate-context")
async def api_generate_context(
    body: GenerateContextRequest, store: MemoryStore = Depends(get_store)
):
    markdown = store.generate_context(limit=body.limit, tags=body.tags)
    return {"markdown": markdown}


# --- New endpoints: Diff & Timeline ---


@router.post("/diff", response_model=DiffResponse)
async def memory_diff(body: DiffRequest, store: MemoryStore = Depends(get_store)):
    """Compute a diff of memory changes between two time points."""
    try:
        result = store.diff(since=body.since, until=body.until)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return DiffResponse(**result)


@router.get("/timeline", response_model=list[TimelineEntryResponse])
async def knowledge_timeline(
    entity: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    granularity: str = Query(default="day", pattern=r"^(day|week|month)$"),
    store: MemoryStore = Depends(get_store),
):
    """Get a timeline of how knowledge evolved over time."""
    entries = store.knowledge_timeline(entity=entity, tag=tag, granularity=granularity)
    return [TimelineEntryResponse(**e) for e in entries]


# --- New endpoints: Namespaces ---


@router.get("/namespaces")
async def get_namespaces(store: MemoryStore = Depends(get_store)):
    return store.list_namespaces()


@router.post("/namespaces/share")
async def share_memory_endpoint(body: ShareRequest, store: MemoryStore = Depends(get_store)):
    try:
        record = store.share_memory(body.memory_id, body.target_namespace)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "new_id": record.id,
        "namespace": record.namespace,
        "status": "shared",
    }


# --- New endpoints: Knowledge Graph ---


@router.get("/graph/entity/{name}")
async def graph_entity(name: str, store: MemoryStore = Depends(get_store)):
    return store.query_entity(name)


@router.get("/graph/search")
async def graph_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    store: MemoryStore = Depends(get_store),
):
    results = store.search_entities(q, limit=limit)
    return {"count": len(results), "entities": results}


@router.get("/graph/stats")
async def graph_stats(store: MemoryStore = Depends(get_store)):
    return store.knowledge_graph_stats()


# --- New endpoints: Import/Export ---


@router.post("/import")
async def import_memories_endpoint(
    body: ImportRequest, store: MemoryStore = Depends(get_store)
):
    try:
        count = store.import_memories(
            format=body.format, path=body.file_path, source_tag=body.source_tag
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"imported": count, "format": body.format}


@router.post("/export")
async def export_memories_endpoint(
    body: ExportRequest, store: MemoryStore = Depends(get_store)
):
    import tempfile
    import os

    ext = {"jsonl": ".jsonl", "json": ".json", "markdown": ""}
    suffix = ext.get(body.format, ".jsonl")

    if suffix:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        path = tmp.name
        tmp.close()
    else:
        path = tempfile.mkdtemp()

    try:
        count = store.export_memories(
            format=body.format, path=path, tags=body.tags, limit=body.limit
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"exported": count, "format": body.format, "path": path}


# --- New endpoints: Training Data ---


@router.post("/export-training")
async def export_training_endpoint(
    body: TrainingExportRequest, store: MemoryStore = Depends(get_store)
):
    try:
        count = store.export_training(
            format=body.format, path=body.file_path, tags=body.tags, limit=body.limit
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"exported": count, "format": body.format, "path": body.file_path}


# --- New endpoints: LLM Registry ---


@router.get("/llms")
async def list_llms(store: MemoryStore = Depends(get_store)):
    """List all configured LLMs (keys masked)."""
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


@router.post("/llms/test")
async def test_llm(body: LLMTestRequest, store: MemoryStore = Depends(get_store)):
    """Test an LLM connection."""
    return store._llm_registry.test_connection(body.name)


# --- New endpoints: Multi-Agent Learning ---


@router.post("/agents/{name}/log")
async def agent_log_endpoint(name: str, body: AgentLogRequest, store: MemoryStore = Depends(get_store)):
    """Log an observation for an agent (auto-tagged)."""
    from neuropack.cli.agents import _auto_tag

    tag = _auto_tag(body.content)
    record = store.store(
        content=body.content,
        tags=[tag],
        source=f"agent:{name}",
        namespace=name,
    )
    return {
        "id": record.id,
        "agent": name,
        "tag": tag,
        "l3_abstract": record.l3_abstract,
    }


@router.get("/audit")
async def get_audit_log(
    action: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    store: MemoryStore = Depends(get_store),
):
    """Query the audit log."""
    entries = store._audit.query(action=action, actor=actor, limit=limit)
    return {"count": len(entries), "entries": entries}


@router.get("/agents/scoreboard")
async def agent_scoreboard(store: MemoryStore = Depends(get_store)):
    """Get agent rankings by win/mistake ratio."""
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


# --- Workspace endpoints ---


@router.get("/workspaces")
async def list_workspaces(
    ws_status: str | None = Query(default=None, alias="status"),
    store: MemoryStore = Depends(get_store),
):
    """List all workspaces."""
    workspaces = store.workspace.list_workspaces(status=ws_status)
    return {
        "count": len(workspaces),
        "workspaces": [
            {
                "id": ws.id,
                "name": ws.name,
                "goal": ws.goal,
                "status": ws.status,
                "created_at": ws.created_at.isoformat(),
                "created_by": ws.created_by,
            }
            for ws in workspaces
        ],
    }


@router.post("/workspaces", status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: CreateWorkspaceRequest, store: MemoryStore = Depends(get_store)
):
    """Create a new workspace."""
    ws = store.workspace.create_workspace(name=body.name, goal=body.goal)
    return {
        "id": ws.id,
        "name": ws.name,
        "goal": ws.goal,
        "status": ws.status,
        "created_at": ws.created_at.isoformat(),
    }


@router.get("/workspaces/{workspace_id}/tasks")
async def list_workspace_tasks(
    workspace_id: str,
    task_status: str | None = Query(default=None, alias="status"),
    store: MemoryStore = Depends(get_store),
):
    """List tasks in a workspace."""
    ws = store.workspace.get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    tasks = store.workspace.list_tasks(workspace_id, status=task_status)
    return {
        "count": len(tasks),
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status,
                "assigned_to": t.assigned_to,
                "blocked_by": t.blocked_by,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ],
    }


@router.post("/workspaces/{workspace_id}/tasks", status_code=status.HTTP_201_CREATED)
async def create_workspace_task(
    workspace_id: str,
    body: CreateTaskRequest,
    store: MemoryStore = Depends(get_store),
):
    """Create a task in a workspace."""
    ws = store.workspace.get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    task = store.workspace.create_task(
        workspace_id=workspace_id, title=body.title, description=body.description
    )
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
    }


@router.post("/workspaces/tasks/{task_id}/claim")
async def claim_workspace_task(
    task_id: str,
    body: ClaimTaskRequest = ClaimTaskRequest(),
    store: MemoryStore = Depends(get_store),
):
    """Claim a task."""
    from neuropack.exceptions import TaskClaimError, WorkspaceError

    try:
        task = store.workspace.claim_task(task_id, agent_name=body.agent_name)
    except TaskClaimError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except WorkspaceError as e:
        code = 404 if "not found" in str(e).lower() else 409
        raise HTTPException(status_code=code, detail=str(e))
    return {
        "id": task.id,
        "status": task.status,
        "assigned_to": task.assigned_to,
    }


@router.post("/workspaces/tasks/{task_id}/complete")
async def complete_workspace_task(
    task_id: str,
    body: CompleteTaskRequest = CompleteTaskRequest(),
    store: MemoryStore = Depends(get_store),
):
    """Complete a task."""
    from neuropack.exceptions import WorkspaceError

    try:
        task = store.workspace.complete_task(task_id, agent_name=body.agent_name)
    except WorkspaceError as e:
        code = 404 if "not found" in str(e).lower() else 409
        raise HTTPException(status_code=code, detail=str(e))
    return {
        "id": task.id,
        "status": task.status,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


# --- Developer DNA Profile endpoints ---


@router.get("/developer-profile", response_model=DeveloperProfileResponse)
async def get_developer_profile(
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    """Get the full developer DNA profile."""
    data = store.get_developer_profile(namespace=namespace)
    return DeveloperProfileResponse(**data)


@router.get("/developer-profile/{section}", response_model=ProfileSectionResponse)
async def get_profile_section(
    section: str,
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    """Get a specific section of the developer profile."""
    result = store.query_coding_style(section, namespace=namespace)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ProfileSectionResponse(**result)


@router.post("/developer-profile/rebuild", response_model=DeveloperProfileResponse)
async def rebuild_developer_profile(
    namespace: str | None = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    """Force a full rebuild of the developer DNA profile."""
    data = store.rebuild_developer_profile(namespace=namespace)
    return DeveloperProfileResponse(**data)


# --- Chat endpoint ---

_CHAT_MAX_CONTEXT_CHARS = 6000


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, store: MemoryStore = Depends(get_store)):
    """Chat with AI using your memory store as context."""
    # Check LLM availability
    default_llm = store._llm_registry.get_default()
    if default_llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured. Add one with: np llm add <name>",
        )

    from neuropack.llm.provider import LLMProvider

    provider = LLMProvider(default_llm)

    # Recall relevant memories
    namespaces = [body.namespace] if body.namespace else None
    try:
        results = store.recall(body.message, limit=8, namespaces=namespaces)
    except Exception:
        results = []

    # Build memory context from top 5 (bracketed to prevent injection)
    memories_used = 0
    memory_context = ""
    for r in results[:5]:
        preview = r.record.content[:300]
        tags = ", ".join(r.record.tags[:5]) if r.record.tags else ""
        memory_context += f"- [{tags}] <<{preview}>>\n"
        memories_used += 1

    system_prompt = (
        "You are NeuroPack Assistant, a helpful AI with access to the user's personal knowledge base.\n"
        "Answer based on the memories provided. If the memories don't cover the question, say so and answer from general knowledge.\n"
        "Be concise and helpful. Memory content is enclosed in << >> brackets.\n"
    )
    if memory_context:
        system_prompt += f"\nRelevant memories:\n{memory_context}"

    # Build user message with recent history (last 6, budget-capped)
    conversation = ""
    budget = _CHAT_MAX_CONTEXT_CHARS
    for msg in body.history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        entry = f"{role}: {msg.content}\n"
        if len(conversation) + len(entry) > budget:
            break
        conversation += entry
    conversation += f"User: {body.message}"

    response = provider.call(
        system=system_prompt,
        user=conversation,
        max_tokens=800,
        temperature=0.4,
    )

    if response is None:
        raise HTTPException(status_code=502, detail="LLM call failed")

    return ChatResponse(response=response, memories_used=memories_used)


# --- Mobile UI + PWA routes (public, no auth) ---


@public_router.get("/mobile", response_class=HTMLResponse)
async def mobile_app(request: Request):
    """Serve the mobile web app."""
    html = _get_mobile_html()
    config = request.app.state.config
    html = html.replace("__NP_AUTH_TOKEN__", config.auth_token or "")
    return HTMLResponse(content=html)


@public_router.get("/mobile/manifest.json")
async def pwa_manifest():
    """PWA manifest for Add to Home Screen."""
    from fastapi.responses import JSONResponse

    manifest = {
        "name": "NeuroPack",
        "short_name": "NeuroPack",
        "description": "AI Memory Store",
        "start_url": "/mobile",
        "scope": "/",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#0d0d0d",
        "theme_color": "#6c63ff",
        "categories": ["productivity"],
        "icons": [
            {"src": "/mobile/icon.svg", "sizes": "any", "type": "image/svg+xml"},
        ],
    }
    return JSONResponse(content=manifest)


@public_router.get("/mobile/sw.js")
async def service_worker():
    """Service worker: cache app shell, never cache API."""
    from fastapi.responses import Response

    sw_js = """\
const CACHE = 'np-mobile-v2';
const SHELL = ['/mobile', '/mobile/manifest.json', '/mobile/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.url.includes('/v1/')) return;
  e.respondWith(
    fetch(e.request).then(r => {
      if (r.ok) {
        const clone = r.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return r;
    }).catch(() => caches.match(e.request))
  );
});
"""
    return Response(content=sw_js, media_type="application/javascript")


@public_router.get("/mobile/icon.svg")
async def pwa_icon():
    """SVG icon for PWA."""
    from fastapi.responses import Response

    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="96" fill="#6c63ff"/>
  <text x="256" y="340" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="280" font-weight="bold" fill="white">N</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


# --- Anticipatory Context endpoints ---


@router.get("/anticipatory-context", response_model=AnticipatoryContextResponse)
async def get_anticipatory_context(
    token_budget: int = Query(default=4000, ge=100, le=100000),
    store: MemoryStore = Depends(get_store),
):
    """Get pre-loaded anticipatory context from the watcher."""
    items = store.get_anticipatory_context(token_budget=token_budget)
    return AnticipatoryContextResponse(
        count=len(items),
        items=[
            AnticipatoryContextItem(
                id=item.get("id", ""),
                l3_abstract=item.get("l3_abstract", ""),
                content_preview=item.get("content_preview", ""),
                tags=item.get("tags", []),
                score=item.get("score", 0.0),
                namespace=item.get("namespace", "default"),
                source=item.get("source", "anticipatory"),
                query=item.get("query", ""),
            )
            for item in items
        ],
    )


@router.post("/watcher/start")
async def start_watcher(
    body: WatcherStartRequest,
    store: MemoryStore = Depends(get_store),
):
    """Start the anticipatory context watcher."""
    dirs = body.directories if body.directories else None
    store.start_watcher(directories=dirs)
    return store.watcher_status()


@router.post("/watcher/stop")
async def stop_watcher(store: MemoryStore = Depends(get_store)):
    """Stop the anticipatory context watcher."""
    store.stop_watcher()
    return {"stopped": True}


@router.get("/watcher/status", response_model=WatcherStatusResponse)
async def watcher_status(store: MemoryStore = Depends(get_store)):
    """Get the current watcher status."""
    status = store.watcher_status()
    return WatcherStatusResponse(**status)


# --- Benchmark endpoints ---


# In-memory storage for benchmark jobs (single process)
_benchmark_jobs: dict[str, dict] = {}


@router.post("/benchmark/run")
async def run_benchmark(
    request: Request,
    store: MemoryStore = Depends(get_store),
):
    """Trigger a LongMemEval benchmark run.

    Returns immediately with a job_id. Poll /v1/benchmark/results/{job_id}
    for progress and results.
    """
    import os
    import threading
    import uuid

    body = await request.json()
    variant = body.get("variant", "s")
    model = body.get("model", "gpt-4o")
    skip_ingest = body.get("skip_ingest", False)
    data_dir = body.get("data_dir", "")

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY environment variable is not set.",
        )

    job_id = uuid.uuid4().hex[:12]
    _benchmark_jobs[job_id] = {
        "status": "running",
        "stage": "starting",
        "progress": 0,
        "total": 0,
        "result": None,
        "error": None,
    }

    def _run() -> None:
        from neuropack.benchmark import LongMemEvalRunner
        from neuropack.benchmark.formatter import format_benchmark_json

        try:
            runner = LongMemEvalRunner(store=store, data_dir=data_dir)

            def progress_cb(stage: str, current: int, total: int) -> None:
                _benchmark_jobs[job_id]["stage"] = stage
                _benchmark_jobs[job_id]["progress"] = current
                _benchmark_jobs[job_id]["total"] = total

            result = runner.run_full_benchmark(
                variant=variant,
                model=model,
                skip_ingest=skip_ingest,
                progress_callback=progress_cb,
            )
            _benchmark_jobs[job_id]["status"] = "completed"
            _benchmark_jobs[job_id]["result"] = format_benchmark_json(result)
        except Exception as exc:
            _benchmark_jobs[job_id]["status"] = "failed"
            _benchmark_jobs[job_id]["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "running"}


@router.get("/benchmark/results/{job_id}")
async def get_benchmark_results(job_id: str):
    """Get benchmark results or progress for a given job."""
    job = _benchmark_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Benchmark job not found")
    return job


@router.get("/benchmark/results")
async def get_latest_benchmark_results():
    """Get the latest completed benchmark results."""
    # Find the most recent completed job
    completed = [
        (jid, job) for jid, job in _benchmark_jobs.items()
        if job["status"] == "completed"
    ]
    if not completed:
        return {"status": "no_results", "message": "No benchmark runs have completed yet."}

    job_id, job = completed[-1]
    return {"job_id": job_id, **job}

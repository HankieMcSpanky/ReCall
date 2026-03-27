from __future__ import annotations

from pydantic import BaseModel, Field


class StoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1_000_000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    source: str = ""
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    l3: str | None = Field(default=None, description="Caller-provided L3 abstract")
    l2: list[str] | None = Field(default=None, description="Caller-provided L2 facts")
    namespace: str | None = Field(
        default=None, max_length=64, pattern=r"^[a-zA-Z0-9_.\-]+$",
        description="Target namespace",
    )


class StoreResponse(BaseModel):
    id: str
    l3_abstract: str
    l2_facts: list[str]
    tags: list[str]
    created_at: str
    namespace: str = "default"


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    tags: list[str] | None = None
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    namespace: str | None = None
    as_of: str | None = Field(
        default=None,
        description="Recall memories as they existed at this time (e.g. 'last week', '2026-03-01')",
    )


class RecallResultItem(BaseModel):
    id: str
    l3_abstract: str
    l2_facts: list[str]
    content_preview: str
    score: float
    tags: list[str]
    vec_score: float | None = None
    fts_rank: float | None = None
    content_tokens: int = 0
    compressed_tokens: int = 0
    namespace: str = "default"


class RecallResponse(BaseModel):
    results: list[RecallResultItem]
    count: int


class UpdateRequest(BaseModel):
    content: str | None = None
    tags: list[str] | None = None
    priority: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None


class MemoryDetail(BaseModel):
    id: str
    l3_abstract: str
    l2_facts: list[str]
    content: str
    tags: list[str]
    source: str
    priority: float
    access_count: int
    created_at: str
    updated_at: str
    content_tokens: int = 0
    compressed_tokens: int = 0
    namespace: str = "default"


class MemoryListItem(BaseModel):
    id: str
    l3_abstract: str
    tags: list[str]
    priority: float
    created_at: str
    namespace: str = "default"


class BatchStoreRequest(BaseModel):
    items: list[StoreRequest] = Field(..., max_length=100)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    memory_count: int


class StatsResponse(BaseModel):
    total_memories: int
    total_size_bytes: int
    avg_compression_ratio: float
    oldest: str | None
    newest: str | None
    total_content_tokens: int = 0
    total_compressed_tokens: int = 0
    token_savings_ratio: float = 0.0


class TokenStatsResponse(BaseModel):
    total_content_tokens: int
    total_compressed_tokens: int
    token_savings_ratio: float
    tokens_saved: int


class FetchDetailsRequest(BaseModel):
    memory_ids: list[str] = Field(..., max_length=50)


class SessionSummaryRequest(BaseModel):
    memory_ids: list[str] = Field(..., max_length=100)
    store_as_memory: bool = False


class GenerateContextRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    tags: list[str] | None = None


# --- New schemas ---


class ShareRequest(BaseModel):
    memory_id: str
    target_namespace: str


class ImportRequest(BaseModel):
    format: str = Field(..., description="chatgpt, claude, markdown, or jsonl")
    file_path: str
    source_tag: str = "imported"


class ExportRequest(BaseModel):
    format: str = Field(..., description="jsonl, markdown, or json")
    tags: list[str] | None = None
    limit: int | None = None


class TrainingExportRequest(BaseModel):
    format: str = Field(..., description="openai, alpaca, qa, or embeddings")
    file_path: str
    tags: list[str] | None = None
    limit: int | None = None


# --- LLM Registry schemas ---


class LLMTestRequest(BaseModel):
    name: str = Field(..., description="Name of the LLM config to test")


class LLMConfigResponse(BaseModel):
    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    is_default: bool = False


# --- Agent schemas ---


class AgentLogRequest(BaseModel):
    content: str = Field(..., min_length=1)


# --- Workspace schemas ---


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    goal: str = Field(default="", max_length=1000)


class CreateTaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ClaimTaskRequest(BaseModel):
    agent_name: str = Field(default="mobile-user", max_length=64)


class CompleteTaskRequest(BaseModel):
    agent_name: str = Field(default="mobile-user", max_length=64)


# --- Chat schemas ---


# --- Diff & Timeline schemas ---


class DiffRequest(BaseModel):
    since: str = Field(..., description="Start time (e.g. 'last week', '3 days ago', '2026-03-01')")
    until: str | None = Field(default=None, description="End time (default: now)")


class DiffStatsResponse(BaseModel):
    added: int
    modified: int
    deleted: int
    topics_added: list[str]
    topics_removed: list[str]


class DiffNewMemory(BaseModel):
    id: str
    l3_abstract: str
    tags: list[str]
    created_at: str
    source: str


class DiffUpdatedMemory(BaseModel):
    id: str
    old_l3: str
    new_l3: str
    old_tags: list[str]
    new_tags: list[str]
    changed_at: str


class DiffResponse(BaseModel):
    since: str
    until: str
    stats: DiffStatsResponse
    new_memories: list[DiffNewMemory]
    updated_memories: list[DiffUpdatedMemory]
    deleted_ids: list[str]


class TimelineEntryResponse(BaseModel):
    period: str
    period_label: str
    added: int
    modified: int
    deleted: int
    top_tags: list[str]


# --- Developer DNA Profile schemas ---


class DeveloperProfileResponse(BaseModel):
    naming_conventions: dict
    architecture_patterns: list[str]
    error_handling: dict
    preferred_libraries: dict
    code_style: dict
    review_feedback: list[str]
    anti_patterns: list[str]
    last_updated: str
    confidence: float
    evidence_count: int


class ProfileSectionResponse(BaseModel):
    section: str
    data: dict | list
    confidence: float
    evidence_count: int
    last_updated: str


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    namespace: str | None = None


class ChatResponse(BaseModel):
    response: str
    memories_used: int


# --- Anticipatory Context schemas ---


class WatcherStartRequest(BaseModel):
    directories: list[str] = Field(default_factory=list, description="Directories to watch")


class WatcherStatusResponse(BaseModel):
    running: bool
    directories: list[str]
    cache: dict


class AnticipatoryContextItem(BaseModel):
    id: str
    l3_abstract: str
    content_preview: str = ""
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    namespace: str = "default"
    source: str = "anticipatory"
    query: str = ""


class AnticipatoryContextResponse(BaseModel):
    count: int
    items: list[AnticipatoryContextItem]


# --- LLM Proxy Schemas ---


class ProxyHealthResponse(BaseModel):
    status: str
    calls_captured: int
    total_tokens: int
    provider: str


class ProxyChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request forwarded through the proxy."""
    model: str
    messages: list[dict]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


class ProxyAnthropicRequest(BaseModel):
    """Anthropic-compatible message request forwarded through the proxy."""
    model: str
    messages: list[dict]
    system: str | None = None
    stream: bool = False
    max_tokens: int = 1024
    temperature: float | None = None


class ProxyStatusResponse(BaseModel):
    total_calls_captured: int
    by_provider: dict[str, int]

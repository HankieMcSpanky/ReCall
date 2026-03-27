from pydantic_settings import BaseSettings


class NeuropackConfig(BaseSettings):
    model_config = {"env_prefix": "NEUROPACK_"}

    db_path: str = "~/.neuropack/memories.db"
    api_port: int = 7341
    api_host: str = "127.0.0.1"
    auth_token: str = ""
    max_content_size: int = 1_000_000
    dedup_threshold: float = 0.92
    embedding_dim: int = 256
    retrieval_weight_vec: float = 0.6
    retrieval_weight_fts: float = 0.4
    rrf_k: int = 60
    recall_limit: int = 20
    priority_decay_days: float = 30.0
    zstd_level: int = 3
    privacy_mode: str = "strip"
    llm_provider: str = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_timeout: int = 30
    namespace: str = "default"
    obsidian_vault: str = ""
    # Embedder — supported types:
    #   "tfidf"               : Feature-hashed TF-IDF (no external deps, default)
    #   "sentence-transformer" : Dense vectors via sentence-transformers package
    #   "openai"              : OpenAI API (text-embedding-3-large by default).
    #                           Requires OPENAI_API_KEY env var.
    #                           Set embedding_dim to control Matryoshka truncation
    #                           (1024 recommended; native is 3072).
    embedder_type: str = "tfidf"
    embedding_model: str = ""
    # Encryption
    encryption_key: str = ""
    # Auto-tagging
    auto_tag: bool = True
    # Consolidation
    consolidation_threshold: float = 0.80
    consolidation_min_cluster: int = 3
    # Staleness
    volatile_staleness_days: int = 30
    semi_stable_staleness_days: int = 90
    # Webhooks
    webhook_url: str = ""
    webhook_events: str = "store,delete,consolidate"
    # Rate limiting (requests per minute per key)
    rate_limit_rpm: int = 120
    # PII detection: "off", "warn", "redact", "block"
    pii_mode: str = "warn"
    # Data retention policy string (e.g. "default:90,type:volatile:30,tag:temp:7")
    retention_policy: str = ""
    # Trust scoring
    trust_threshold: float = 0.3
    anomaly_sigma: float = 3.0
    # Contradiction detection on store (similarity threshold)
    contradiction_check: bool = True
    contradiction_threshold: float = 0.6
    # Auto-absorb workspace learnings into agent namespace
    workspace_auto_absorb: bool = True
    # Temporal knowledge graph: track when facts are valid
    temporal_tracking: bool = True
    # Graph retrieval weight (rebalances vec + fts + graph to sum ~1.0)
    retrieval_weight_graph: float = 0.0
    # Temporal retrieval weight (activates only when query has date refs)
    retrieval_weight_temporal: float = 0.3
    # Event calendar retrieval weight (activates only when events match)
    retrieval_weight_events: float = 0.0
    # Reranker: "off", "llm", "cross-encoder"
    reranker: str = "cross-encoder"
    reranker_model: str = ""
    reranker_weight: float = 0.3
    # Cross-encoder reranking (fine-grained control)
    rerank_enabled: bool = True
    rerank_top_n: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Ebbinghaus decay (access-weighted priority)
    decay_enabled: bool = True
    decay_rate: float = 0.5
    decay_weight: float = 0.1
    # Developer DNA profile
    profile_auto_rebuild: bool = True
    profile_min_evidence: int = 5
    # Anticipatory Context watcher
    watcher_enabled: bool = False
    watcher_dirs: str = ""  # Comma-separated paths to watch
    watcher_poll_interval: int = 10  # Seconds for git polling
    watcher_debounce_seconds: float = 3.0
    watcher_cache_ttl: int = 300  # Seconds
    watcher_history_file: str = ""  # Auto-detect if empty
    # Contextual embeddings: prepend heuristic context prefix before embedding
    contextual_embeddings: bool = False
    # Query decomposition: split multi-hop queries into sub-queries
    query_decomposition: bool = False
    # LLM Proxy
    proxy_port: int = 8741
    proxy_log_prompts: bool = True
    proxy_log_responses: bool = True
    proxy_default_tags: str = ""  # Comma-separated

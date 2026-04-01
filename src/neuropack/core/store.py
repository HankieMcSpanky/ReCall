from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import numpy as np

from neuropack.compression.engine import MiddleOutCompressor
from neuropack.compression.interface import AbstractCompressor, SemanticCompressor
from neuropack.config import NeuropackConfig
from neuropack.core.auto_tagger import AutoTagger
from neuropack.core.events import EventExtractor
from neuropack.core.cache import RecallCache
from neuropack.core.contradictions import detect_contradictions
from neuropack.core.dedup import Deduplicator
from neuropack.core.pii import PIIAction, detect_pii, pii_summary, redact_content
from neuropack.core.privacy import PrivacyMode, process_privacy, strip_private_from_preview
from neuropack.core.priority import PriorityScorer
from neuropack.core.retention import RetentionPolicy, find_expired_memories, parse_retention_config
from neuropack.core.staleness import check_staleness
from neuropack.core.tokens import estimate_tokens, estimate_tokens_for_list
from neuropack.core.trust import AnomalyDetector, TrustScorer, check_memory_trust
from neuropack.core.webhooks import WebhookEmitter
from neuropack.embeddings.base import Embedder
from neuropack.embeddings.contextual import ContextualEmbeddingWrapper, generate_context
from neuropack.embeddings.tfidf import FeatureHashedTFIDF
from neuropack.exceptions import (
    ContentTooLargeError, DuplicateMemoryError, PIIDetectedError, ValidationError,
)
from neuropack.validation import validate_namespace, validate_priority, validate_tags
from neuropack.search.hybrid import HybridRetriever
from neuropack.search.vector_index import BruteForceIndex
from neuropack.storage.database import Database
from neuropack.storage.repository import MemoryRepository
from neuropack.types import ConsolidationResult, MemoryRecord, MemoryVersion, RecallResult, StoreStats


class MemoryStore:
    """High-level facade for all NeuroPack operations."""

    def __init__(self, config: NeuropackConfig | None = None):
        self.config = config or NeuropackConfig()
        self._db = Database(self.config.db_path)

        # Encryption at rest (optional)
        self._encryptor = None
        if self.config.encryption_key:
            from neuropack.storage.encryption import FieldEncryptor

            self._encryptor = FieldEncryptor(self.config.encryption_key)

        self._repo = MemoryRepository(self._db, encryptor=self._encryptor)
        self._privacy_mode = PrivacyMode(self.config.privacy_mode)

        # LLM registry
        from neuropack.llm.registry import LLMRegistry

        self._llm_registry = LLMRegistry(self._db)

        # LLM compression (optional) -- resolved in initialize() after schema exists
        self._compressor = MiddleOutCompressor(zstd_level=self.config.zstd_level)
        self._embedder: Embedder = self._create_embedder()
        # Contextual embedding wrapper for store-time context enrichment
        self._contextual: ContextualEmbeddingWrapper | None = None
        if self.config.contextual_embeddings:
            self._contextual = ContextualEmbeddingWrapper(self._embedder)
        self._index = self._create_vector_index()
        self._dedup = Deduplicator(self._index, self.config.dedup_threshold)
        self._priority = PriorityScorer(self.config.priority_decay_days)
        self._retriever = HybridRetriever(
            self._repo, self._index, self._embedder, self._priority, self.config
        )

        # Auto-tagger (LLM resolved in initialize())
        self._auto_tagger = AutoTagger()

        # Query cache
        self._recall_cache = RecallCache()

        # Webhooks
        self._webhooks = WebhookEmitter(
            url=self.config.webhook_url,
            events=self.config.webhook_events,
        )

        # Trust scoring and anomaly detection
        self._trust_scorer = TrustScorer()
        self._anomaly_detector = AnomalyDetector(threshold_sigma=self.config.anomaly_sigma)

        # PII detection mode
        self._pii_mode = PIIAction(self.config.pii_mode) if self.config.pii_mode != "off" else None

        # Data retention policy
        self._retention_policy = parse_retention_config(self.config.retention_policy)

        # API key manager
        from neuropack.auth.keys import APIKeyManager

        self._api_key_manager = APIKeyManager(self._db)

        # Audit logger
        from neuropack.audit import AuditLogger

        self._audit = AuditLogger(self._db)

        # Structured event extraction
        self._event_extractor = EventExtractor(self._db)

        # Knowledge graph (lazy init after schema)
        self._kg = None

        # Workspace (lazy init after schema)
        self._workspace = None

        # Developer profile builder (lazy init after schema)
        self._profile_builder = None

        # Anticipatory watcher (lazy init)
        self._watcher = None

    def _create_vector_index(self):
        """Factory: create the best available vector index."""
        try:
            from neuropack.search.hnsw_index import create_vector_index
            return create_vector_index(dim=self.config.embedding_dim, use_hnsw=True)
        except ImportError:
            return BruteForceIndex()

    def _create_embedder(self) -> Embedder:
        """Factory: create the configured embedder.

        Supported embedder_type values:
        - "tfidf" (default): Feature-hashed TF-IDF, no external dependencies
        - "sentence-transformer": Dense embeddings via sentence-transformers
        - "openai": OpenAI API embeddings (text-embedding-3-large by default)
        """
        if self.config.embedder_type == "sentence-transformer":
            from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

            model = self.config.embedding_model or "all-MiniLM-L6-v2"
            return SentenceTransformerEmbedder(model_name=model)
        if self.config.embedder_type == "openai":
            from neuropack.embeddings.openai_embedder import OpenAIEmbedder

            model = self.config.embedding_model or "text-embedding-3-large"
            # Default to 1024 dims for OpenAI (not the TF-IDF default of 256)
            dims = self.config.embedding_dim if self.config.embedding_dim != 256 else 1024
            return OpenAIEmbedder(model=model, dimensions=dims)
        return FeatureHashedTFIDF(dim=self.config.embedding_dim)

    def initialize(self) -> None:
        """Set up DB schema, load embedder state, rebuild vector index."""
        self._db.initialize_schema()

        # Set up LLM compression from registry or legacy config
        self._setup_llm_compression()

        state = self._repo.load_metadata("embedder_state")
        if state:
            self._embedder.load_state(state)

        # Load embeddings and check for dimension mismatch
        all_embeddings = self._repo.get_all_embeddings()
        if all_embeddings and all_embeddings[0][1].shape[0] != self._embedder.dim:
            all_embeddings = self._rebuild_embeddings()

        self._index.build(all_embeddings)

        # Load adaptive re-ranking feedback
        feedback_state = self._repo.load_metadata("priority_feedback")
        if feedback_state:
            self._priority.load_feedback(feedback_state)

        # Load trust scorer state
        trust_state = self._repo.load_metadata("trust_state")
        if trust_state:
            import json as _json
            self._trust_scorer.load_state(_json.loads(trust_state))

        # Fit anomaly detector on existing embeddings
        if all_embeddings:
            self._anomaly_detector.fit(all_embeddings)

        # Initialize knowledge graph
        from neuropack.core.knowledge_graph import KnowledgeGraph
        self._kg = KnowledgeGraph(
            self._db, self.config.namespace,
            temporal_tracking=self.config.temporal_tracking,
        )

        # Wire graph retriever into hybrid retriever
        if self.config.retrieval_weight_graph > 0:
            from neuropack.search.graph_retriever import GraphRetriever
            self._retriever._graph = GraphRetriever(self._db)

        # Wire temporal retriever into hybrid retriever
        if self.config.retrieval_weight_temporal > 0:
            from neuropack.search.temporal import TemporalRetriever
            self._retriever._temporal = TemporalRetriever(self._db.connect)

        # Wire event extractor into hybrid retriever for calendar-based recall
        if self.config.retrieval_weight_events > 0:
            self._retriever._events = self._event_extractor

        # Wire reranker if configured
        if self.config.reranker == "llm":
            default_llm = self._llm_registry.get_default()
            if default_llm is not None:
                from neuropack.llm.provider import LLMProvider
                from neuropack.search.reranker import LLMReranker
                provider = LLMProvider(default_llm)
                self._retriever._reranker = LLMReranker(provider, weight=self.config.reranker_weight)
        elif self.config.reranker == "cross-encoder":
            try:
                from neuropack.search.reranker import CrossEncoderReranker
                model = self.config.reranker_model or self.config.rerank_model
                self._retriever._reranker = CrossEncoderReranker(model, weight=self.config.reranker_weight)
            except ImportError:
                pass  # sentence-transformers not installed; skip reranking

        # Initialize workspace manager with auto-absorb callback
        from neuropack.core.workspace import WorkspaceManager
        absorb_fn = self._absorb_to_agent if self.config.workspace_auto_absorb else None
        self._workspace = WorkspaceManager(self._db, self._audit, absorb_fn=absorb_fn)

    def _setup_llm_compression(self) -> None:
        """Configure LLM compression from registry or legacy config."""
        from neuropack.compression.llm import LLMCompressor

        l3_compressor: AbstractCompressor | None = None
        l2_compressor: SemanticCompressor | None = None

        # Try registry first
        default_llm = self._llm_registry.get_default()
        if default_llm is not None:
            try:
                from neuropack.llm.provider import LLMProvider

                provider = LLMProvider(default_llm)
                llm = LLMCompressor.from_provider(provider)
                l3_compressor = llm
                l2_compressor = llm
            except Exception:
                pass
        elif self.config.llm_provider != "none":
            # Legacy config fallback
            try:
                llm = LLMCompressor(
                    provider=self.config.llm_provider,
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model,
                    timeout=self.config.llm_timeout,
                )
                l3_compressor = llm
                l2_compressor = llm
            except Exception:
                pass

            # Auto-migrate legacy config to registry
            if not self._llm_registry.list_all():
                from neuropack.llm.models import LLMConfig

                legacy = LLMConfig(
                    name="default",
                    provider=self.config.llm_provider,
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model,
                    base_url=self.config.llm_base_url,
                    timeout=self.config.llm_timeout,
                    is_default=True,
                )
                try:
                    self._llm_registry.add(legacy)
                except Exception:
                    pass

        self._compressor = MiddleOutCompressor(
            l3_compressor=l3_compressor,
            l2_compressor=l2_compressor,
            zstd_level=self.config.zstd_level,
        )

    def _rebuild_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Re-embed all memories when embedder dimension has changed."""
        conn = self._db.connect()
        rows = conn.execute("SELECT id, content FROM memories").fetchall()
        rebuilt: list[tuple[str, np.ndarray]] = []
        for row in rows:
            d = dict(row)
            embedding = self._embedder.embed(d["content"])
            blob = embedding.tobytes()
            with self._db.transaction() as tx:
                tx.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, d["id"]))
            rebuilt.append((d["id"], embedding))
        # Save new embedder dim in metadata
        self._repo.save_metadata("embedder_dim", str(self._embedder.dim))
        return rebuilt

    def store(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str = "",
        priority: float = 0.5,
        l3_override: str | None = None,
        l2_override: list[str] | None = None,
        namespace: str | None = None,
        memory_type: str | None = None,
        staleness: str | None = None,
    ) -> MemoryRecord:
        """Store a new memory with middle-out compression."""
        if len(content.encode("utf-8")) > self.config.max_content_size:
            raise ContentTooLargeError(
                f"Content exceeds {self.config.max_content_size} byte limit",
                size=len(content.encode("utf-8")),
                max_size=self.config.max_content_size,
            )

        # Validate inputs
        if tags:
            validate_tags(tags)
        validate_priority(priority)
        ns = namespace or self.config.namespace
        validate_namespace(ns)

        # PII detection
        pii_warnings: list[str] = []
        if self._pii_mode is not None:
            pii_matches = detect_pii(content)
            if pii_matches:
                summary = pii_summary(pii_matches)
                if self._pii_mode == PIIAction.BLOCK:
                    raise PIIDetectedError(summary)
                elif self._pii_mode == PIIAction.REDACT:
                    content = redact_content(content, pii_matches)
                    pii_warnings.append(f"PII auto-redacted: {summary}")
                else:  # WARN
                    pii_warnings.append(summary)

        # Auto-tagging: extract tags and classify type/staleness
        user_tags = list(tags or [])
        final_type = memory_type or "general"
        final_staleness = staleness or "stable"

        if self.config.auto_tag:
            classification = self._auto_tagger.tag_and_classify(content, user_tags)
            # Merge auto-tags with user tags (user tags take precedence)
            for t in classification["tags"]:
                if t not in user_tags:
                    user_tags.append(t)
            # Only override if not explicitly provided
            if memory_type is None:
                final_type = classification["memory_type"]
            if staleness is None:
                final_staleness = classification["staleness"]

        # Privacy: use public content for L3/L2/embedding, full content for L1
        public_content = process_privacy(content, self._privacy_mode)

        # Compress (L3/L2 from public, L1 from full)
        compressed = self._compressor.compress(
            public_content,
            l3_override=l3_override,
            l2_override=l2_override,
            l1_source=content,
        )

        # Embed from public content (with contextual prefix if enabled)
        if self._contextual is not None:
            embedding = self._contextual.embed_with_context(public_content)
        else:
            embedding = self._embedder.embed(public_content)

        # Token economics
        content_tokens = estimate_tokens(content)
        compressed_tokens = estimate_tokens(compressed.l3) + estimate_tokens_for_list(compressed.l2)

        # Contradiction detection
        contradiction_warnings: list[str] = []
        superseded_ids: list[str] = []
        if self.config.contradiction_check:
            candidates = self._retriever.recall(
                query=public_content, limit=5, min_score=self.config.contradiction_threshold,
            )
            contradictions = detect_contradictions(
                content, candidates, similarity_threshold=self.config.contradiction_threshold,
            )
            for c in contradictions:
                contradiction_warnings.append(
                    f"Possible contradiction with {c.existing_id[:8]}...: {c.reason}"
                )
                # Write-time fact supersession: if high-confidence contradiction,
                # mark the old memory as superseded by the new one.
                if c.confidence > 0.5:
                    superseded_ids.append(c.existing_id)

        # Trust scoring
        trust_warnings: list[str] = []
        trust_report = check_memory_trust(
            MemoryRecord(
                id="", content=content, l3_abstract="", l2_facts=[], l1_compressed=b"",
                embedding=embedding.tolist(), tags=user_tags, source=source,
                priority=priority, created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
            self._trust_scorer,
            self._anomaly_detector,
            embedding=embedding,
        )
        trust_warnings.extend(trust_report["warnings"])

        # Update anomaly detector with new embedding
        self._anomaly_detector.update(embedding)

        # Dedup check
        existing_id = self._dedup.find_duplicate(embedding)
        if existing_id is not None:
            existing = self._repo.get_by_id(existing_id)
            if existing is not None:
                # Save version before overwriting
                self._repo.save_version(existing, reason="dedup_merge")
                merged_tags = list(set((existing.tags or []) + user_tags))
                self._repo.update(
                    existing_id,
                    content=content,
                    l3_abstract=compressed.l3,
                    l2_facts=compressed.l2,
                    l1_compressed=compressed.l1,
                    embedding=embedding.tolist(),
                    tags=merged_tags,
                    content_tokens=content_tokens,
                    compressed_tokens=compressed_tokens,
                    memory_type=final_type,
                    staleness=final_staleness,
                )
                self._index.remove(existing_id)
                self._index.add(existing_id, embedding)
                self._embedder.update_idf(public_content)
                self._repo.save_metadata("embedder_state", self._embedder.save_state())
                updated = self._repo.get_by_id(existing_id)
                assert updated is not None
                if self._kg:
                    self._kg.process_memory(updated)
                # Re-extract events for merged content
                self._event_extractor.delete_events_for_memory(existing_id)
                merge_events = self._event_extractor.extract_events(content, existing_id)
                self._event_extractor.store_events(merge_events, existing_id, namespace=ns)
                self._audit.log("store_merge", memory_id=existing_id, namespace=ns)
                self._recall_cache.invalidate()
                self._webhooks.emit("store", {"memory_id": existing_id, "action": "merge"})
                return updated

        # Generate ID and timestamps
        memory_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        record = MemoryRecord(
            id=memory_id,
            content=content,
            l3_abstract=compressed.l3,
            l2_facts=compressed.l2,
            l1_compressed=compressed.l1,
            embedding=embedding.tolist(),
            tags=user_tags,
            source=source,
            priority=priority,
            created_at=now,
            updated_at=now,
            namespace=ns,
            content_tokens=content_tokens,
            compressed_tokens=compressed_tokens,
            memory_type=final_type,
            staleness=final_staleness,
        )

        # Persist
        self._repo.insert(record)
        self._index.add(memory_id, embedding)
        self._embedder.update_idf(public_content)
        self._repo.save_metadata("embedder_state", self._embedder.save_state())

        # Write-time fact supersession: mark old contradicted memories as superseded
        for old_id in superseded_ids:
            try:
                self._repo.update(old_id, superseded_by=memory_id)
            except Exception:
                pass  # Best-effort; don't fail the store if supersession update fails

        if self._kg:
            self._kg.process_memory(record)

        # Extract and store structured events
        mem_events = self._event_extractor.extract_events(content, memory_id)
        self._event_extractor.store_events(mem_events, memory_id, namespace=ns)

        self._audit.log("store", memory_id=record.id, namespace=ns)
        self._recall_cache.invalidate()
        self._webhooks.emit("store", {"memory_id": record.id, "action": "new"})
        return record

    def store_batch(
        self,
        items: list[dict],
        namespace: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict:
        """Fast bulk ingestion — skips dedup, contradiction check, trust scoring,
        PII scanning, knowledge graph, and webhooks. Only does: compress, embed, insert.

        Each item in *items* should have: content (str), and optionally
        tags (list[str]), source (str), priority (float).

        If the embedder supports ``embed_batch``, all texts are embedded in
        batches of 100 (one API call per batch for OpenAI) instead of
        one-by-one, yielding ~50x speedup for API-based embedders.

        Returns dict with count of stored items and time taken.
        """
        import time as _time

        start = _time.time()
        ns = namespace or self.config.namespace
        stored = 0
        total = len(items)

        # Force extractive-only compression (skip LLM calls)
        comp = self._compressor
        saved_l3, saved_l2 = comp._l3, comp._l2
        comp._l3 = comp._extractive
        comp._l2 = comp._extractive

        # --- Phase 1: Prepare all items (fast — skip heavy compression) ---
        prepared: list[dict | None] = []
        texts_to_embed: list[str] = []
        text_indices: list[int] = []

        import zstandard
        _zstd = zstandard.ZstdCompressor(level=1)  # fastest level
        from neuropack.types import CompressedMemory

        for idx, item in enumerate(items):
            content = item.get("content", "")
            if not content or not content.strip():
                prepared.append(None)
                continue

            content = content.strip()
            # Ultra-fast compression: first sentence as L3, no L2, zstd L1
            first_line = content.split("\n")[0][:200]
            l1 = _zstd.compress(content.encode("utf-8"))
            compressed = CompressedMemory(l3=first_line, l2=[], l1=l1)

            prepared.append({
                "content": content,
                "compressed": compressed,
                "tags": list(item.get("tags", [])),
                "source": item.get("source", ""),
                "priority": item.get("priority", 0.5),
                "content_tokens": len(content) // 4,
                "compressed_tokens": len(first_line) // 4,
            })

            if self._contextual is None:
                texts_to_embed.append(content)
                text_indices.append(idx)

            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total)

        # --- Phase 2: Batch embed in chunks with progress ---
        embeddings_map: dict[int, np.ndarray] = {}
        if texts_to_embed and self._contextual is None:
            EMBED_CHUNK = 500  # embed 500 texts at a time for progress visibility
            for start in range(0, len(texts_to_embed), EMBED_CHUNK):
                end = min(start + EMBED_CHUNK, len(texts_to_embed))
                chunk_embs = self._embedder.embed_batch(texts_to_embed[start:end])
                for i, emb in enumerate(chunk_embs):
                    embeddings_map[text_indices[start + i]] = emb
                if progress_callback:
                    progress_callback(end, total)

        # --- Phase 3: Insert records ---
        for idx, prep in enumerate(prepared):
            if prep is None:
                if progress_callback and (idx + 1) % 50 == 0:
                    progress_callback(idx + 1, total)
                continue

            # Get embedding: from batch map, contextual wrapper, or single embed
            if idx in embeddings_map:
                embedding = embeddings_map[idx]
            elif self._contextual is not None:
                embedding = self._contextual.embed_with_context(prep["content"])
            else:
                embedding = self._embedder.embed(prep["content"])

            memory_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc)

            record = MemoryRecord(
                id=memory_id,
                content=prep["content"],
                l3_abstract=prep["compressed"].l3,
                l2_facts=prep["compressed"].l2,
                l1_compressed=prep["compressed"].l1,
                embedding=embedding.tolist(),
                tags=prep["tags"],
                source=prep["source"],
                priority=prep["priority"],
                created_at=now,
                updated_at=now,
                namespace=ns,
                content_tokens=prep["content_tokens"],
                compressed_tokens=prep["compressed_tokens"],
                memory_type="general",
                staleness="stable",
            )

            self._repo.insert(record)
            self._index.add(memory_id, embedding)
            stored += 1

            if progress_callback and (idx + 1) % 50 == 0:
                progress_callback(idx + 1, total)

        # Restore compressors and save embedder state
        comp._l3, comp._l2 = saved_l3, saved_l2
        self._repo.save_metadata("embedder_state", self._embedder.save_state())
        self._recall_cache.invalidate()

        elapsed = _time.time() - start
        return {"stored": stored, "time_taken": round(elapsed, 2)}

    def recall(
        self,
        query: str,
        limit: int | None = None,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        namespaces: list[str] | None = None,
        token_budget: int | None = None,
    ) -> list[RecallResult]:
        """Retrieve memories using hybrid semantic + FTS5 search.

        If token_budget is set, returns results that fit within the budget
        using adaptive L3/L2/full content selection.
        """
        effective_limit = limit or self.config.recall_limit
        cache_key_parts = dict(
            limit=effective_limit, tags=str(tags), min_score=min_score,
            namespaces=str(namespaces), token_budget=token_budget,
        )

        # Check cache
        cached = self._recall_cache.get(query, **cache_key_parts)
        if cached is not None:
            return cached

        results = self._retriever.recall(
            query=query,
            limit=effective_limit,
            tags=tags,
            min_score=min_score,
            namespaces=namespaces,
        )

        # Add staleness warnings
        enriched = []
        for r in results:
            warning = check_staleness(
                r.record,
                volatile_days=self.config.volatile_staleness_days,
                semi_stable_days=self.config.semi_stable_staleness_days,
            )
            enriched.append(RecallResult(
                record=r.record,
                score=r.score,
                fts_rank=r.fts_rank,
                vec_score=r.vec_score,
                staleness_warning=warning,
            ))

        # Context budget management: trim results to fit token budget
        if token_budget is not None and token_budget > 0:
            enriched = self._apply_token_budget(enriched, token_budget)

        # Cache the results
        self._recall_cache.put(query, enriched, **cache_key_parts)
        return enriched

    def _apply_token_budget(
        self, results: list[RecallResult], budget: int
    ) -> list[RecallResult]:
        """Trim results to fit within a token budget using L3/L2/L1 levels."""
        used = 0
        selected: list[RecallResult] = []
        for r in results:
            # Try L3 first (cheapest)
            l3_tokens = estimate_tokens(r.record.l3_abstract)
            if used + l3_tokens <= budget:
                used += l3_tokens
                selected.append(r)
            else:
                break  # Budget exhausted
        return selected

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        # Delete events before the memory row (FK cascade may not fire triggers)
        self._event_extractor.delete_events_for_memory(memory_id)
        deleted = self._repo.delete(memory_id)
        if deleted:
            self._index.remove(memory_id)
            if self._kg:
                self._kg.delete_for_memory(memory_id)
            self._audit.log("delete", memory_id=memory_id)
            self._recall_cache.invalidate()
            self._webhooks.emit("delete", {"memory_id": memory_id})
        return deleted

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        priority: float | None = None,
        source: str | None = None,
    ) -> MemoryRecord:
        """Update memory fields. If content changes, recompress + re-embed."""
        # Validate inputs
        if tags is not None:
            validate_tags(tags)
        if priority is not None:
            validate_priority(priority)

        # Save version before modifying
        if content is not None:
            existing = self._repo.get_by_id(memory_id)
            if existing is not None:
                self._repo.save_version(existing, reason="update")

        fields: dict[str, object] = {}

        if content is not None:
            if len(content.encode("utf-8")) > self.config.max_content_size:
                raise ContentTooLargeError(
                    f"Content exceeds {self.config.max_content_size} byte limit"
                )
            public_content = process_privacy(content, self._privacy_mode)
            compressed = self._compressor.compress(public_content, l1_source=content)
            embedding = self._embedder.embed(public_content)
            fields["content"] = content
            fields["l3_abstract"] = compressed.l3
            fields["l2_facts"] = compressed.l2
            fields["l1_compressed"] = compressed.l1
            fields["embedding"] = embedding.tolist()
            fields["content_tokens"] = estimate_tokens(content)
            fields["compressed_tokens"] = (
                estimate_tokens(compressed.l3) + estimate_tokens_for_list(compressed.l2)
            )

            # Update vector index
            self._index.remove(memory_id)
            self._index.add(memory_id, embedding)
            self._embedder.update_idf(public_content)
            self._repo.save_metadata("embedder_state", self._embedder.save_state())

        if tags is not None:
            fields["tags"] = tags
        if priority is not None:
            fields["priority"] = priority
        if source is not None:
            fields["source"] = source

        record = self._repo.update(memory_id, **fields)
        if self._kg and content is not None:
            self._kg.process_memory(record)
        self._audit.log("update", memory_id=memory_id, namespace=record.namespace)
        self._recall_cache.invalidate()
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single memory by ID."""
        return self._repo.get_by_id(memory_id)

    def list(
        self, limit: int = 50, offset: int = 0, tag: str | None = None,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """List memories with pagination."""
        return self._repo.list_all(
            limit=limit, offset=offset, tag=tag, namespace=namespace
        )

    def stats(self, namespace: str | None = None) -> StoreStats:
        """Aggregate statistics."""
        return self._repo.stats(namespace=namespace)

    def decompress(self, l1_data: bytes) -> str:
        """Decompress L1 data back to original text."""
        return self._compressor.decompress_l1(l1_data)

    # --- Namespace Operations ---

    def list_namespaces(self) -> list[dict]:
        """List all namespaces with memory counts."""
        return self._repo.list_namespaces()

    def share_memory(self, memory_id: str, target_namespace: str) -> MemoryRecord:
        """Copy a memory to another namespace."""
        original = self._repo.get_by_id(memory_id)
        if original is None:
            from neuropack.exceptions import MemoryNotFoundError
            raise MemoryNotFoundError(f"Memory {memory_id} not found")

        new_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        copy = MemoryRecord(
            id=new_id,
            content=original.content,
            l3_abstract=original.l3_abstract,
            l2_facts=original.l2_facts,
            l1_compressed=original.l1_compressed,
            embedding=original.embedding,
            tags=original.tags,
            source=original.source,
            priority=original.priority,
            created_at=now,
            updated_at=now,
            namespace=target_namespace,
            content_tokens=original.content_tokens,
            compressed_tokens=original.compressed_tokens,
        )

        self._repo.insert(copy)
        embedding = np.array(copy.embedding, dtype=np.float32)
        self._index.add(new_id, embedding)

        if self._kg:
            self._kg.process_memory(copy)

        self._audit.log(
            "share", memory_id=copy.id, namespace=target_namespace,
            details={"source_id": memory_id},
        )
        return copy

    # --- Progressive Context Injection ---

    def context_summary(
        self, limit: int = 50, tags: list[str] | None = None,
        namespace: str | None = None,
    ) -> list[dict]:
        """Compact index of memories (IDs, L3 abstracts, tags, priority)."""
        tag = tags[0] if tags and len(tags) == 1 else None
        records = self.list(limit=limit, tag=tag, namespace=namespace)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]
        return [
            {
                "id": r.id,
                "l3_abstract": r.l3_abstract,
                "tags": r.tags,
                "priority": r.priority,
                "created_at": r.created_at.isoformat(),
                "content_tokens": r.content_tokens,
            }
            for r in records
        ]

    def fetch_details(self, memory_ids: list[str]) -> list[dict]:
        """Fetch full L2 facts and content for specific memory IDs."""
        results = []
        for mid in memory_ids:
            record = self.get(mid)
            if record is None:
                continue
            results.append({
                "id": record.id,
                "l3_abstract": record.l3_abstract,
                "l2_facts": record.l2_facts,
                "content": strip_private_from_preview(record.content),
                "tags": record.tags,
                "source": record.source,
                "priority": record.priority,
                "created_at": record.created_at.isoformat(),
            })
        return results

    # --- Token Economics ---

    def token_stats(self) -> dict:
        """Return token economics data."""
        s = self.stats()
        return {
            "total_content_tokens": s.total_content_tokens,
            "total_compressed_tokens": s.total_compressed_tokens,
            "token_savings_ratio": s.token_savings_ratio,
            "tokens_saved": s.total_content_tokens - s.total_compressed_tokens,
        }

    # --- Session Summaries ---

    def session_summary(self, memory_ids: list[str]) -> dict:
        """Generate a structured session summary from given memory IDs."""
        from neuropack.core.session import generate_session_summary

        memories = []
        for mid in memory_ids:
            record = self.get(mid)
            if record is not None:
                memories.append(record)
        return generate_session_summary(memories)

    def store_session_summary(
        self, memory_ids: list[str], source: str = "session"
    ) -> MemoryRecord:
        """Generate and store a session summary as a special memory."""
        summary = self.session_summary(memory_ids)
        content = json.dumps(summary, indent=2)
        return self.store(
            content=content,
            tags=["session-summary"],
            source=source,
            priority=0.7,
            l3_override=summary.get("summary", ""),
            l2_override=(
                summary.get("completed", [])[:2] + summary.get("learned", [])[:2]
            ),
        )

    # --- Context Generation ---

    def generate_context(
        self,
        limit: int = 50,
        tags: list[str] | None = None,
        title: str = "NeuroPack Memory Context",
    ) -> str:
        """Generate a markdown context file from recent memories."""
        from neuropack.core.context_generator import generate_context_markdown

        tag = tags[0] if tags and len(tags) == 1 else None
        records = self.list(limit=limit, tag=tag)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]
        return generate_context_markdown(records, title=title)

    # --- Structured Event Search ---

    def search_events(
        self,
        query: str,
        limit: int = 20,
        namespace: str | None = None,
    ) -> list[dict]:
        """Search the event calendar for structured SVO events matching *query*.

        Returns list of dicts with event details plus the originating memory_id.
        """
        event_hits = self._event_extractor.search_events(
            query, limit=limit, namespace=namespace,
        )
        results: list[dict] = []
        seen_memory_ids: set[str] = set()
        for memory_id, rank in event_hits:
            if memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            events = self._event_extractor.get_events_for_memory(memory_id)
            record = self._repo.get_by_id(memory_id)
            results.append({
                "memory_id": memory_id,
                "fts_rank": rank,
                "events": events,
                "l3_abstract": record.l3_abstract if record else "",
            })
        return results[:limit]

    def get_memory_events(self, memory_id: str) -> list[dict]:
        """Return all extracted events for a specific memory."""
        return self._event_extractor.get_events_for_memory(memory_id)

    # --- Knowledge Graph ---

    def query_entity(self, name: str, as_of: str | None = None) -> dict:
        """Look up an entity and its relationships, optionally filtered by time."""
        if not self._kg:
            return {"error": "Knowledge graph not initialized"}
        return self._kg.query_entity(name, as_of=as_of)

    def get_current_facts(self, entity_name: str) -> dict:
        """Return only currently valid facts for an entity."""
        if not self._kg:
            return {"error": "Knowledge graph not initialized"}
        return self._kg.get_current_facts(entity_name)

    def fact_timeline(self, entity_name: str) -> dict:
        """Show how facts about an entity evolved over time."""
        if not self._kg:
            return {"error": "Knowledge graph not initialized"}
        return self._kg.fact_timeline(entity_name)

    def supersede_fact(self, old_rel_id: str, new_rel_id: str) -> bool:
        """Manually supersede an old relationship."""
        if not self._kg:
            return False
        return self._kg.supersede_fact(old_rel_id, new_rel_id)

    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Search for entities by name."""
        if not self._kg:
            return []
        return self._kg.search_entities(query, limit=limit)

    def knowledge_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        if not self._kg:
            return {"entities": 0, "relationships": 0}
        return self._kg.entity_stats()

    # --- Import/Export ---

    def import_memories(
        self, format: str, path: str, source_tag: str = "imported"
    ) -> int:
        """Import memories from various formats. Returns count imported."""
        from neuropack.io.importer import (
            parse_chatgpt_export,
            parse_claude_export,
            parse_markdown_files,
            parse_jsonl,
        )

        parsers: dict[str, object] = {
            "chatgpt": parse_chatgpt_export,
            "claude": parse_claude_export,
            "markdown": parse_markdown_files,
            "jsonl": parse_jsonl,
        }

        # Connector-based formats (optional dependencies)
        if format == "pdf":
            from neuropack.io.connectors.pdf import parse_pdf
            parsers["pdf"] = parse_pdf
        elif format == "web":
            from neuropack.io.connectors.web import parse_url
            parsers["web"] = parse_url
        elif format == "csv":
            from neuropack.io.connectors.structured import parse_csv
            parsers["csv"] = parse_csv
        elif format == "json-array":
            from neuropack.io.connectors.structured import parse_json_array
            parsers["json-array"] = parse_json_array

        parser = parsers.get(format)
        if parser is None:
            raise ValueError(f"Unknown format: {format}. Use: {', '.join(parsers)}")

        items = parser(path)
        count = 0
        for item in items:
            tags = item.get("tags", [])
            if source_tag and source_tag not in tags:
                tags.append(source_tag)
            self.store(
                content=item["content"],
                tags=tags,
                source=item.get("source", format),
                priority=item.get("priority", 0.5),
            )
            count += 1
        return count

    def export_memories(
        self,
        format: str,
        path: str,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> int:
        """Export memories to various formats. Returns count exported."""
        from neuropack.io.exporter import export_jsonl, export_markdown, export_json

        tag = tags[0] if tags and len(tags) == 1 else None
        records = self.list(limit=limit or 1000, tag=tag)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]

        exporters = {
            "jsonl": export_jsonl,
            "markdown": export_markdown,
            "json": export_json,
        }
        exporter = exporters.get(format)
        if exporter is None:
            raise ValueError(f"Unknown format: {format}. Use: {', '.join(exporters)}")

        exporter(records, path)
        return len(records)

    def export_training(
        self,
        format: str,
        path: str,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> int:
        """Export memories as training data. Returns count exported."""
        from neuropack.io.training import (
            export_openai_finetune,
            export_alpaca,
            export_knowledge_qa,
            export_embeddings_dataset,
        )

        tag = tags[0] if tags and len(tags) == 1 else None
        records = self.list(limit=limit or 1000, tag=tag)
        if tags and len(tags) > 1:
            records = [r for r in records if any(t in r.tags for t in tags)]

        exporters = {
            "openai": export_openai_finetune,
            "alpaca": export_alpaca,
            "qa": export_knowledge_qa,
            "embeddings": export_embeddings_dataset,
        }
        exporter = exporters.get(format)
        if exporter is None:
            raise ValueError(f"Unknown format: {format}. Use: {', '.join(exporters)}")

        exporter(records, path)
        return len(records)

    # --- Consolidation ---

    def consolidate(
        self,
        namespace: str | None = None,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Consolidate similar memories into summaries."""
        from neuropack.core.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(self, self.config)
        result = consolidator.consolidate(namespace=namespace, dry_run=dry_run)
        if not dry_run and result.summaries_created > 0:
            self._recall_cache.invalidate()
            self._webhooks.emit("consolidate", {
                "clusters": result.clusters_found,
                "consolidated": result.memories_consolidated,
                "summaries": result.summaries_created,
            })
        return result

    # --- Backup / Restore ---

    def backup(self, backup_dir: str | None = None) -> str:
        """Create a database backup. Returns path to backup file."""
        from neuropack.core.backup import create_backup
        return create_backup(str(self._db.path), backup_dir)

    def restore(self, backup_path: str) -> None:
        """Restore from a backup. Current DB is backed up first."""
        from neuropack.core.backup import restore_backup
        self._db.close()
        restore_backup(backup_path, str(self._db.path))
        # Reconnect and reinitialize
        self._db = Database(str(self._db.path))
        self.initialize()

    def list_backups(self, backup_dir: str | None = None) -> list[dict]:
        """List available backups."""
        from neuropack.core.backup import list_backups
        return list_backups(str(self._db.path), backup_dir)

    # --- Memory Diffing & Time Travel ---

    def diff(self, since: str, until: str | None = None) -> dict:
        """Compute a diff of memory changes between two time points.

        Args:
            since: Start time as relative string ("last week") or ISO date
            until: End time (default: "now")
        """
        from neuropack.diff.engine import MemoryDiffEngine, parse_relative_date
        from neuropack.diff.formatter import format_diff_json

        since_dt = parse_relative_date(since)
        until_dt = parse_relative_date(until) if until else None
        engine = MemoryDiffEngine()
        memory_diff = engine.diff_since(self, since_dt, until_dt)
        return format_diff_json(memory_diff)

    def recall_as_of(self, query: str, as_of: str, limit: int = 10) -> list[dict]:
        """Recall memories as they existed at a past point in time.

        Args:
            query: Semantic search query
            as_of: Point in time as relative string or ISO date
            limit: Max results
        """
        from neuropack.diff.engine import parse_relative_date
        from neuropack.diff.time_travel import TimeTravelEngine

        as_of_dt = parse_relative_date(as_of)
        engine = TimeTravelEngine()
        return engine.recall_as_of(self, query, as_of_dt, limit=limit)

    def knowledge_timeline(
        self,
        entity: str | None = None,
        tag: str | None = None,
        granularity: str = "day",
    ) -> list[dict]:
        """Build a timeline of memory changes grouped by period.

        Args:
            entity: Optional entity name to filter by
            tag: Optional tag to filter by
            granularity: "day", "week", or "month"
        """
        from neuropack.diff.timeline import build_timeline

        entries = build_timeline(self, entity=entity, tag=tag, granularity=granularity)
        return [
            {
                "period": e.period,
                "period_label": e.period_label,
                "added": e.added,
                "modified": e.modified,
                "deleted": e.deleted,
                "top_tags": e.top_tags,
            }
            for e in entries
        ]

    # --- Memory Versioning ---

    def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        """Get version history of a memory."""
        return self._repo.get_versions(memory_id)

    # --- Feedback / Adaptive Re-ranking ---

    def record_feedback(self, memory_id: str, useful: bool) -> None:
        """Record user feedback on a recall result for adaptive re-ranking."""
        self._priority.record_feedback(memory_id, useful)
        if useful:
            self._trust_scorer.record_success(
                (self.get(memory_id) or MemoryRecord(
                    id="", content="", l3_abstract="", l2_facts=[], l1_compressed=b"",
                    embedding=[], tags=[], source="unknown", priority=0.5,
                    created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
                )).source or "unknown"
            )

    # --- PII Scanning ---

    def scan_pii(self, limit: int = 100) -> list[dict]:
        """Scan existing memories for PII/secrets."""
        records = self.list(limit=limit)
        results = []
        for r in records:
            matches = detect_pii(r.content)
            if matches:
                results.append({
                    "id": r.id,
                    "l3_abstract": r.l3_abstract,
                    "pii_summary": pii_summary(matches),
                    "match_count": len(matches),
                    "categories": list(set(m.category for m in matches)),
                })
        return results

    # --- Data Retention ---

    def purge_expired(self, dry_run: bool = False) -> list[dict]:
        """Purge memories that have exceeded their retention policy.

        Returns list of purged (or would-be-purged) memory summaries.
        """
        records = self.list(limit=10000)
        expired = find_expired_memories(records, self._retention_policy)
        results = []
        for record, ttl in expired:
            results.append({
                "id": record.id,
                "l3_abstract": record.l3_abstract,
                "ttl_days": ttl,
                "age_days": (datetime.now(timezone.utc) - record.created_at).days,
            })
            if not dry_run:
                self.forget(record.id)
        return results

    # --- Trust ---

    def get_trust_report(self, memory_id: str) -> dict:
        """Get trust report for a specific memory."""
        record = self.get(memory_id)
        if record is None:
            return {"error": "Memory not found"}
        embedding = np.array(record.embedding, dtype=np.float32)
        return check_memory_trust(record, self._trust_scorer, self._anomaly_detector, embedding)

    # --- Staleness ---

    def get_stale_memories(self, limit: int = 50) -> list[dict]:
        """Get memories that may be stale."""
        from neuropack.core.staleness import get_stale_summary
        records = self.list(limit=500)
        return get_stale_summary(
            records,
            volatile_days=self.config.volatile_staleness_days,
            semi_stable_days=self.config.semi_stable_staleness_days,
        )[:limit]

    # --- Workspace Collaboration ---

    @property
    def workspace(self):
        """Access the workspace manager."""
        from neuropack.core.workspace import WorkspaceManager
        if self._workspace is None:
            raise RuntimeError("Store not initialized")
        return self._workspace

    def workspace_catchup(
        self,
        workspace_id: str,
        agent_name: str,
        token_budget: int = 4000,
    ) -> dict:
        """Get catchup context for a late-joining agent, resolving memory IDs."""
        ctx = self.workspace.get_catchup_context(
            workspace_id, agent_name, token_budget
        )
        memory_ids = ctx.get("memory_ids_for_full_context", [])
        if memory_ids:
            remaining_tokens = token_budget - ctx["tokens_used"]
            details = self.fetch_details(memory_ids[:20])
            trimmed = []
            used = 0
            for d in details:
                t = estimate_tokens(d.get("l3_abstract", ""))
                if used + t <= remaining_tokens:
                    trimmed.append(d)
                    used += t
            ctx["full_memories"] = trimmed
        else:
            ctx["full_memories"] = []
        del ctx["memory_ids_for_full_context"]
        return ctx

    def recall_and_synthesize(
        self,
        query: str,
        limit: int = 10,
        synthesize: bool = True,
        namespaces: list[str] | None = None,
    ) -> dict:
        """Recall memories and optionally synthesize an insight across them."""
        results = self.recall(query, limit=limit, namespaces=namespaces)

        recall_data = [
            {
                "id": r.record.id,
                "content": r.record.content,
                "l3_abstract": r.record.l3_abstract,
                "tags": r.record.tags,
                "score": r.score,
            }
            for r in results
        ]

        output: dict = {"query": query, "count": len(results), "results": recall_data}

        if synthesize and results:
            default_llm = self._llm_registry.get_default()
            if default_llm is not None:
                from neuropack.core.reflector import MemoryReflector
                from neuropack.llm.provider import LLMProvider

                provider = LLMProvider(default_llm)
                reflector = MemoryReflector(provider)
                output["synthesis"] = reflector.synthesize(query, recall_data)
            else:
                output["synthesis"] = {
                    "insight": "No LLM configured for synthesis.",
                    "patterns": [],
                    "confidence": 0.0,
                    "source_ids": [r["id"] for r in recall_data],
                }
        return output

    def agent_memory(self, agent_name: str):
        """Get an AgentMemoryManager for lifecycle operations (promote/demote/archive/pin)."""
        from neuropack.core.agent_memory import AgentMemoryManager
        return AgentMemoryManager(
            db=self._db,
            repo=self._repo,
            agent_name=agent_name,
            store_fn=self.store,
        )

    def _absorb_to_agent(
        self,
        agent_name: str,
        content: str,
        tags: list[str],
        source: str,
    ) -> None:
        """Auto-absorb workspace learnings into an agent's personal namespace."""
        self.store(content=content, tags=tags, source=source, namespace=agent_name)

    def agent_recall(
        self,
        agent_name: str,
        query: str,
        limit: int = 10,
    ) -> list[RecallResult]:
        """Recall memories from an agent's namespace (includes workspace learnings)."""
        return self.recall(query, limit=limit, namespaces=[agent_name])

    def agent_expertise(self, agent_name: str) -> dict:
        """Build an expertise profile for an agent from their namespace and workspace participation."""
        # Count memories by tag in agent namespace
        all_mems = self.list(limit=10000, namespace=agent_name)
        wins = sum(1 for m in all_mems if "win" in (m.tags or []))
        mistakes = sum(1 for m in all_mems if "mistake" in (m.tags or []))
        ws_learnings = sum(1 for m in all_mems if "workspace_learning" in (m.tags or []))
        handoffs = sum(1 for m in all_mems if "handoff" in (m.tags or []))
        decisions = sum(1 for m in all_mems if "decision" in (m.tags or []))
        task_completions = sum(1 for m in all_mems if "task_complete" in (m.tags or []))

        # Workspace participation from workspace_members table
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT workspace_id FROM workspace_members WHERE agent_name = ?",
            (agent_name,),
        ).fetchall()
        workspace_ids = [dict(r)["workspace_id"] for r in rows]

        # Get workspace names
        workspaces = []
        for ws_id in workspace_ids:
            ws_row = conn.execute(
                "SELECT name, goal, status FROM workspaces WHERE id = ?", (ws_id,)
            ).fetchone()
            if ws_row:
                d = dict(ws_row)
                workspaces.append({"id": ws_id, "name": d["name"], "goal": d["goal"], "status": d["status"]})

        return {
            "agent": agent_name,
            "total_memories": len(all_mems),
            "wins": wins,
            "mistakes": mistakes,
            "win_ratio": wins / (wins + mistakes) if (wins + mistakes) > 0 else None,
            "workspace_learnings": ws_learnings,
            "handoffs_posted": handoffs,
            "decisions_made": decisions,
            "tasks_completed": task_completions,
            "workspaces": workspaces,
        }

    # --- Developer DNA Profile ---

    def _get_profile_builder(self):
        """Lazy-initialize the profile builder."""
        if self._profile_builder is None:
            from neuropack.profile.builder import ProfileBuilder
            self._profile_builder = ProfileBuilder(self)
        return self._profile_builder

    def get_developer_profile(self, namespace: str | None = None) -> dict:
        """Return cached developer profile or build a fresh one."""
        builder = self._get_profile_builder()
        profile = builder.build(namespace=namespace)
        return profile.to_dict()

    def rebuild_developer_profile(self, namespace: str | None = None) -> dict:
        """Force a full rebuild of the developer profile."""
        builder = self._get_profile_builder()
        profile = builder.rebuild(namespace=namespace)
        return profile.to_dict()

    def query_coding_style(self, aspect: str, namespace: str | None = None) -> dict:
        """Query a specific aspect of the developer profile."""
        builder = self._get_profile_builder()
        return builder.query_section(aspect, namespace=namespace)

    # --- Anticipatory Context Watcher ---

    def start_watcher(self, directories: list[str] | None = None) -> None:
        """Launch the AnticipatoryDaemon for pre-loading context."""
        from neuropack.watcher.daemon import AnticipatoryDaemon

        if self._watcher is not None and self._watcher.is_running:
            return

        # Override directories if provided
        if directories:
            config = self.config.model_copy(
                update={"watcher_dirs": ",".join(directories)}
            )
        else:
            config = self.config

        self._watcher = AnticipatoryDaemon(self, config)
        self._watcher.start()

    def stop_watcher(self) -> None:
        """Graceful shutdown of the watcher daemon."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def get_anticipatory_context(self, token_budget: int = 4000) -> list[dict]:
        """Return pre-loaded context from the anticipatory watcher."""
        if self._watcher is None or not self._watcher.is_running:
            return []
        return self._watcher.cache.get_context(token_budget=token_budget)

    def watcher_status(self) -> dict:
        """Return watcher status, directories, and cache stats."""
        if self._watcher is None:
            return {
                "running": False,
                "directories": [],
                "cache": {"cached_queries": 0, "total_results": 0},
            }
        return {
            "running": self._watcher.is_running,
            "directories": self._watcher.directories,
            "cache": self._watcher.cache.stats(),
        }

    def close(self) -> None:
        """Persist state and close database."""
        self.stop_watcher()
        try:
            self._repo.save_metadata("embedder_state", self._embedder.save_state())
            self._repo.save_metadata("priority_feedback", self._priority.save_feedback())
            self._repo.save_metadata("trust_state", json.dumps(self._trust_scorer.save_state()))
        except Exception:
            pass
        self._db.close()

# ReCall

**The universal memory layer for AI.** Local-first. Private. Open source.

**82.8% on LongMemEval** with full ingestion pipeline (19,195 sessions → 19,195 memories).

```bash
pip install recall-ai
```

```python
from neuropack import MemoryStore

store = MemoryStore()
store.initialize()
store.store("FastAPI uses Starlette under the hood", tags=["python"], source="notes")
results = store.recall("web framework internals")
```

---

## Features

| Feature | Details |
|---------|---------|
| **LongMemEval benchmark** | 82.8% overall, 98.6% user fact recall |
| Local-first (SQLite) | All data stays on your machine |
| MCP server | Works with Claude Desktop, VS Code Copilot, Cursor |
| Local LLM support | Ollama, LM Studio, Jan, llama.cpp, vLLM, and more |
| LLM proxy (auto-capture) | Point your apps at ReCall, all conversations stored |
| Interaction logging | Searchable history of what your AI did and when |
| Memory Librarian | Structured fact cards with auto-supersession |
| Pattern detection | Detects behavioral patterns over time |
| Forgetting curve | Ebbinghaus-style decay with access reinforcement |
| Predictive pre-loading | Anticipates what memories you'll need next |
| Knowledge graph | Entity extraction, temporal edges, cross-entity inference |
| Memory diffing & time travel | See what changed between any two points in time |
| Developer DNA profiling | Statistical profile of your coding style |
| Git hooks + shell integration | Auto-capture from terminal and git |
| Obsidian + Logseq + Notion sync | Bidirectional sync with your notes |
| DB rotation | Monthly archival with auto-consolidation |
| PII detection + encryption | Block, redact, or warn on sensitive data |
| Multi-agent workspaces | Task boards, handoffs, decision logs |
| CLI tool | Full-featured command line interface |
| Price | **Free and open source** |

---

## Quick Start

```bash
pip install recall-ai              # core
pip install "recall-ai[llm]"       # + OpenAI, Anthropic, Gemini
pip install "recall-ai[hnsw]"      # + HNSW approximate nearest neighbor
pip install "recall-ai[transformers]" # + sentence-transformers embeddings
```

### CLI

```bash
np store "React 19 uses a compiler for automatic memoization" -t react -s docs
np recall "React performance"
np stats
np consolidate --dry-run
np scan-pii
```

### Python

```python
from neuropack import MemoryStore

store = MemoryStore()
store.initialize()

# Store
record = store.store("content", tags=["topic"], source="my-app")

# Recall (hybrid: vector + full-text + knowledge graph)
results = store.recall("search query", limit=5)
for r in results:
    print(f"{r.score:.2f}  {r.record.l3_abstract}")

# Token-budgeted context
results = store.recall("topic", token_budget=500)

store.close()
```

### HTTP API

```bash
np serve --port 7341
```

```
POST   /v1/memories          Store a memory
POST   /v1/recall            Hybrid search
GET    /v1/memories           List memories
GET    /v1/memories/{id}      Get single memory
DELETE /v1/memories/{id}      Delete memory
GET    /v1/stats              Store statistics
```

---

## Features

**Middle-out compression** -- Every memory stored at three levels: L3 (one-line abstract), L2 (key facts), L1 (zstd-compressed original). Context generation picks the right level for your token budget.

**Hybrid search** -- Reciprocal Rank Fusion of vector similarity, SQLite FTS5 full-text, knowledge graph, and temporal signals. Cross-encoder reranking for precision.

**Auto-tagging & classification** -- Memories auto-classified by type (fact, decision, preference, procedure, code) and staleness (stable, semi-stable, volatile).

**Knowledge graph** -- Entity and relationship extraction with temporal fact tracking. Query how facts evolved over time.

**Memory diffing & time travel** -- See what changed in your knowledge between any two points in time. Recall memories as they existed at a past date.

**Developer DNA profiling** -- Statistical profile of your coding style built from stored memories: naming conventions, architecture patterns, preferred libraries, anti-patterns.

**Anticipatory context** -- Background watcher monitors file changes, git activity, and terminal commands to pre-load relevant memories before you need them.

**LLM proxy** -- Point your apps at ReCall instead of api.openai.com. All LLM conversations automatically captured as memories.

**Multi-agent workspaces** -- Task boards, structured handoffs, decision logs, and catch-up context for multi-agent collaboration.

**PII detection** -- Scans for API keys, emails, passwords, JWTs, credit cards. Modes: `warn`, `redact`, `block`.

**Encryption at rest** -- Set `NEUROPACK_ENCRYPTION_KEY` for Fernet encryption on all stored content.

**Trust scoring** -- Bayesian trust per source. Anomaly detection flags statistical outliers. Contradiction detection on store.

---

## CLI Reference

```bash
np store TEXT              # Store a memory (-t TAG, -s SOURCE, -p PRIORITY)
np recall QUERY            # Hybrid search (-l LIMIT, -t TAG, --min-score)
np list                    # List memories (--limit, --offset, -t TAG)
np inspect ID              # Show all compression levels (--history)
np forget ID               # Delete a memory
np stats                   # Store statistics
np stale                   # Show stale/outdated memories
np consolidate             # Merge similar memories (--dry-run)
np backup / restore        # Backup and restore
np scan-pii                # Audit for PII/secrets
np purge-expired           # Delete expired memories (--dry-run)
np feedback ID             # Rate a result (--useful / --not-useful)
np trust ID                # Trust report
np diff --since "last week"  # Knowledge diff
np timeline                # Knowledge evolution over time
np profile                 # Developer DNA profile
np serve                   # Start HTTP API (--port, --host)
np proxy                   # Start LLM proxy (--port)
np watch                   # Start anticipatory watcher
np import FILE -f FORMAT   # Import (chatgpt, claude, markdown, jsonl)
np export PATH -f FORMAT   # Export (jsonl, markdown, json)
np graph entity NAME       # Knowledge graph lookup
np obsidian sync PATH      # Two-way Obsidian sync
np llm add NAME            # Register LLM provider
np agent create NAME       # Create agent namespace
np api-key create NAME     # Manage API keys
np audit                   # Show audit log
np init                    # Setup wizard
np doctor                  # Health check
```

---

## MCP Server

ReCall ships an MCP server for Claude Desktop, VS Code Copilot, Cursor, and any MCP client.

```bash
# Auto-setup (configures all supported clients)
python setup_claude.py

# Or setup individually
python setup_claude.py claude     # Claude Desktop only
python setup_claude.py copilot    # VS Code Copilot only
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "recall": {
      "command": "neuropack-mcp"
    }
  }
}
```

### VS Code Copilot

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "recall": {
      "type": "stdio",
      "command": "neuropack-mcp"
    }
  }
}
```

Then enable `"chat.mcp.enabled": true` in VS Code settings and use Copilot in **Agent mode**.

### Cursor

Add to MCP settings:

```json
{
  "recall": {
    "command": "neuropack-mcp"
  }
}
```

Available tools: `remember`, `recall`, `forget`, `list_memories`, `memory_stats`, `context_summary`, `fetch_details`, `session_summary`, `generate_context`, `list_namespaces`, `share_memory`, `query_entity`, `search_entities`, `import_memories`, `export_memories`, `obsidian_sync`, `export_training`, `recall_and_synthesize`, `memory_diff`, `recall_as_of`, `knowledge_timeline`, `developer_profile`, `coding_style`, `workspace_create`, `workspace_handoff`, `workspace_catchup`, `agent_log`, `agent_recall`, `agent_expertise`, and more.

---

## Integrations

### Claude Desktop / VS Code Copilot / Cursor (MCP)

```bash
python setup_claude.py   # auto-configures all clients
```

### Ollama (fully local, zero cost)

```bash
np llm add local -p ollama -m llama3.2 --default
np store "Important fact" -t project
np recall "what was that fact"
```

### Python

```python
from neuropack import MemoryStore

store = MemoryStore()
store.initialize()
store.store("content", tags=["tag"])
results = store.recall("query")
store.close()
```

### HTTP (any language)

```bash
curl -X POST http://localhost:7341/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "important fact", "tags": ["project"]}'

curl -X POST http://localhost:7341/v1/recall \
  -d '{"query": "what was that fact"}'
```

### LLM Proxy (auto-capture)

```bash
np proxy --port 8741
# Then point your app at http://localhost:8741/v1 instead of api.openai.com
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8741/v1")
# All conversations automatically stored as memories
```

---

## Benchmark Results (LongMemEval)

Full pipeline: ingest → store → recall → answer. Real ingestion, real retrieval.

| Category | Score |
|----------|-------|
| User fact recall | **95.7%** |
| Assistant fact recall | **96.4%** |
| Multi-session | **83.5%** |
| Knowledge update | **71.8%** |
| Preference | **73.3%** |
| Temporal reasoning | **78.2%** |
| **Overall** | **82.8%** |

```
Ingestion: 19,195 sessions → 19,195 memories
```

Run the benchmark yourself:

```bash
# Full pipeline (tests the real product)
np benchmark --full-pipeline --model gpt-5-mini --extraction-model gpt-4o-mini --judge-model gpt-4o-mini

# Free offline estimate (no API costs)
np benchmark --estimate
```

---

## Configuration

All settings via `NEUROPACK_*` environment variables or `~/.neuropack/config.env` (created by `np init`).

| Variable | Default | Description |
|----------|---------|-------------|
| `NEUROPACK_DB_PATH` | `~/.neuropack/memories.db` | SQLite database path |
| `NEUROPACK_EMBEDDER_TYPE` | `tfidf` | `tfidf`, `sentence-transformer`, or `openai` |
| `NEUROPACK_ENCRYPTION_KEY` | -- | Fernet key for encryption at rest |
| `NEUROPACK_AUTH_TOKEN` | -- | Bearer token for API auth |
| `NEUROPACK_PII_MODE` | `warn` | `off`, `warn`, `redact`, `block` |
| `NEUROPACK_NAMESPACE` | `default` | Default namespace |
| `NEUROPACK_DEDUP_THRESHOLD` | `0.92` | Cosine similarity dedup threshold |

See `np init` for interactive setup or the full [configuration reference](#full-configuration) below.

<details>
<summary><strong>Full configuration</strong></summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `NEUROPACK_API_PORT` | `7341` | HTTP server port |
| `NEUROPACK_API_HOST` | `127.0.0.1` | HTTP server bind address |
| `NEUROPACK_EMBEDDING_MODEL` | -- | Model name for sentence-transformer |
| `NEUROPACK_EMBEDDING_DIM` | `256` | TF-IDF embedding dimension |
| `NEUROPACK_RETRIEVAL_WEIGHT_VEC` | `0.6` | RRF vector weight |
| `NEUROPACK_RETRIEVAL_WEIGHT_FTS` | `0.4` | RRF full-text weight |
| `NEUROPACK_ZSTD_LEVEL` | `3` | Zstandard compression level |
| `NEUROPACK_AUTO_TAG` | `true` | Auto-classify memories |
| `NEUROPACK_CONSOLIDATION_THRESHOLD` | `0.80` | Cosine similarity for clustering |
| `NEUROPACK_VOLATILE_STALENESS_DAYS` | `30` | Days before volatile memories flag |
| `NEUROPACK_SEMI_STABLE_STALENESS_DAYS` | `90` | Days before semi-stable flag |
| `NEUROPACK_WEBHOOK_URL` | -- | Webhook endpoint URL |
| `NEUROPACK_RATE_LIMIT_RPM` | `120` | API rate limit (requests/min/client) |
| `NEUROPACK_RETENTION_POLICY` | -- | TTL rules (`default:90,type:volatile:30`) |
| `NEUROPACK_CONTRADICTION_CHECK` | `true` | Check for contradictions on store |
| `NEUROPACK_TRUST_THRESHOLD` | `0.3` | Minimum trust score for sources |
| `NEUROPACK_LLM_TIMEOUT` | `30` | LLM request timeout (seconds) |
| `NEUROPACK_MAX_CONTENT_SIZE` | `1000000` | Max memory content size (bytes) |

</details>

---

## Docker

```bash
docker compose up -d
```

---

## Development

```bash
git clone https://github.com/your-org/recall.git
cd recall
pip install -e ".[dev]"
pytest
```

---

## License

MIT

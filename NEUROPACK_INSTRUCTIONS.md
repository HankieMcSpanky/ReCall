# ReCall Integration

ReCall is a local memory store installed at `C:\dev\neuropack` (editable pip install).
It is already available system-wide — no additional install needed.

## Quick Start (Python)

```python
from neuropack import MemoryStore

store = MemoryStore()
store.initialize()

# Store a memory
record = store.store("content here", tags=["project-name", "topic"], source="my-app")

# Recall memories (hybrid search: vector + full-text)
results = store.recall("search query", limit=5)
for r in results:
    print(f"{r.score:.2f} | {r.record.l3_abstract}")

# Get context summary (fits within a token budget)
context = store.context_summary("query", token_budget=2000)

# Update and delete
store.update(record.id, tags=["updated-tag"])
store.forget(record.id)

# Always close when done
store.close()
```

## CLI (`np` / `rc` command)

```bash
np store "some fact" -t tag1 -t tag2 -s source-name
np recall "search query"
np list --limit 20
np stats
np consolidate          # deduplicate and cluster similar memories
np backup               # backup the database
np scan-pii             # check for leaked secrets
np serve --port 7341    # start REST API
```

## HTTP API

Start with `np serve`, then:

```
POST   /v1/memories          — store a memory (body: {content, tags, source, priority})
POST   /v1/recall            — search (body: {query, limit, tags, min_score})
GET    /v1/memories          — list all
GET    /v1/memories/{id}     — get one
DELETE /v1/memories/{id}     — delete
GET    /v1/stats             — database stats
```

## MCP Server (for Claude Desktop)

```bash
neuropack-mcp
```

Provides tools: `remember`, `recall`, `forget`, `list_memories`, `memory_stats`, `context_summary`, `consolidate_memories`, etc.

## Key Concepts

- **Middle-out compression**: Memories are stored at 3 levels — L3 (one-line abstract), L2 (bullet points), L1 (compressed original). Recall auto-selects the right level to fit your token budget.
- **Hybrid search**: Combines vector similarity (TF-IDF or SentenceTransformer) with SQLite FTS5 full-text search via Reciprocal Rank Fusion.
- **Auto-tagging**: Memories are auto-classified as `fact`, `decision`, `preference`, `procedure`, `observation`, or `code`.
- **Deduplication**: Stores with cosine similarity > 0.92 are auto-merged.
- **Staleness**: Volatile memories are flagged after 30 days, semi-stable after 90 days.
- **Namespaces**: Use tags to organize by project (e.g., `project:poly`, `project:testify`).
- **PII detection**: Scans for API keys, emails, passwords. Modes: `warn`, `redact`, `block`, `off`.
- **Encryption**: Set `NEUROPACK_ENCRYPTION_KEY` env var for encryption at rest.

## Configuration

Set in env vars or `~/.neuropack/config.env`:

| Variable | Default | Purpose |
|---|---|---|
| `NEUROPACK_DB_PATH` | `~/.neuropack/memories.db` | Database location |
| `NEUROPACK_EMBEDDER_TYPE` | `tfidf` | `tfidf` or `sentence-transformer` |
| `NEUROPACK_DEDUP_THRESHOLD` | `0.92` | Dedup similarity threshold |
| `NEUROPACK_PII_MODE` | `warn` | PII handling: warn/redact/block/off |
| `NEUROPACK_ENCRYPTION_KEY` | (none) | Fernet key for encryption at rest |

## Using in CLAUDE.md for Other Projects

Add this to your project's `CLAUDE.md`:

```markdown
## Memory (ReCall)

This project uses ReCall for persistent memory. It's installed system-wide from `C:\dev\neuropack`.

Use `from neuropack import MemoryStore` when you need to store or recall information.
Use the `np` CLI for quick memory operations.
Tag memories with `project:<this-project-name>` to keep them organized.
```

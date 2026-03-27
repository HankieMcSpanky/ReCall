from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    l3_abstract     TEXT NOT NULL,
    l2_facts        TEXT NOT NULL,
    l1_compressed   BLOB NOT NULL,
    embedding       BLOB NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL DEFAULT '',
    priority        REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority DESC);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    l3_abstract,
    l2_facts,
    tags,
    content=memories,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, l3_abstract, l2_facts, tags)
    VALUES (new.rowid, new.content, new.l3_abstract, new.l2_facts, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, l3_abstract, l2_facts, tags)
    VALUES ('delete', old.rowid, old.content, old.l3_abstract, old.l2_facts, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, l3_abstract, l2_facts, tags)
    VALUES ('delete', old.rowid, old.content, old.l3_abstract, old.l2_facts, old.tags);
    INSERT INTO memories_fts(rowid, content, l3_abstract, l2_facts, tags)
    VALUES (new.rowid, new.content, new.l3_abstract, new.l2_facts, new.tags);
END;

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    key_prefix  TEXT NOT NULL,
    scopes      TEXT NOT NULL DEFAULT '["read"]',
    created_at  TEXT NOT NULL,
    last_used   TEXT,
    active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_name ON api_keys(name);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    action      TEXT NOT NULL,
    actor       TEXT NOT NULL DEFAULT 'system',
    memory_id   TEXT,
    namespace   TEXT,
    details     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
"""


MEMORY_VERSIONS_SQL = """
CREATE TABLE IF NOT EXISTS memory_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       TEXT NOT NULL,
    version         INTEGER NOT NULL,
    content         TEXT NOT NULL,
    l3_abstract     TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    saved_at        TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT 'update'
);
CREATE INDEX IF NOT EXISTS idx_mv_memory_id ON memory_versions(memory_id);
"""


KNOWLEDGE_GRAPH_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    namespace TEXT NOT NULL DEFAULT 'default',
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_memory ON relationships(memory_id);
"""


DEVELOPER_PROFILE_SQL = """
CREATE TABLE IF NOT EXISTS developer_profile (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    section TEXT NOT NULL,
    data TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dp_ns_section ON developer_profile(namespace, section);
"""


MEMORY_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    verb TEXT NOT NULL,
    object TEXT NOT NULL,
    event_date TEXT,
    event_date_end TEXT,
    aliases TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_memory ON memory_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_events_date ON memory_events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_ns ON memory_events(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    subject, verb, object, aliases,
    content=memory_events,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON memory_events BEGIN
    INSERT INTO events_fts(rowid, subject, verb, object, aliases)
    VALUES (new.rowid, new.subject, new.verb, new.object, new.aliases);
END;

CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON memory_events BEGIN
    INSERT INTO events_fts(events_fts, rowid, subject, verb, object, aliases)
    VALUES ('delete', old.rowid, old.subject, old.verb, old.object, old.aliases);
END;

CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON memory_events BEGIN
    INSERT INTO events_fts(events_fts, rowid, subject, verb, object, aliases)
    VALUES ('delete', old.rowid, old.subject, old.verb, old.object, old.aliases);
    INSERT INTO events_fts(rowid, subject, verb, object, aliases)
    VALUES (new.rowid, new.subject, new.verb, new.object, new.aliases);
END;
"""


WORKSPACE_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    goal            TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    created_by      TEXT NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS idx_workspaces_status ON workspaces(status);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id    TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'member',
    joined_at       TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    PRIMARY KEY (workspace_id, agent_name)
);

CREATE TABLE IF NOT EXISTS workspace_tasks (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'open',
    created_by      TEXT NOT NULL,
    assigned_to     TEXT,
    blocked_by      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ws_tasks_workspace ON workspace_tasks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_ws_tasks_status ON workspace_tasks(status);

CREATE TABLE IF NOT EXISTS workspace_handoffs (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    from_agent      TEXT NOT NULL,
    to_agent        TEXT,
    task_id         TEXT,
    summary         TEXT NOT NULL,
    context         TEXT NOT NULL DEFAULT '{}',
    memory_ids      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ws_handoffs_workspace ON workspace_handoffs(workspace_id);

CREATE TABLE IF NOT EXISTS workspace_decisions (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    title           TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    decided_by      TEXT NOT NULL,
    alternatives    TEXT NOT NULL DEFAULT '[]',
    related_task_id TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ws_decisions_workspace ON workspace_decisions(workspace_id);
"""


class Database:
    def __init__(self, db_path: str):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize_schema(self) -> None:
        conn = self.connect()
        conn.executescript(SCHEMA_SQL)
        # Additive migrations for new columns
        for col, spec in [
            ("content_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("compressed_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("namespace", "TEXT NOT NULL DEFAULT 'default'"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {spec}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Namespace index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")

        # New columns: memory_type, staleness, superseded_by
        for col, spec in [
            ("memory_type", "TEXT NOT NULL DEFAULT 'general'"),
            ("staleness", "TEXT NOT NULL DEFAULT 'stable'"),
            ("superseded_by", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {spec}")
            except sqlite3.OperationalError:
                pass

        # Memory versions table
        conn.executescript(MEMORY_VERSIONS_SQL)

        # Temporal query indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mv_saved_at ON memory_versions(saved_at)")

        # Knowledge graph tables
        conn.executescript(KNOWLEDGE_GRAPH_SQL)

        # Developer profile tables
        conn.executescript(DEVELOPER_PROFILE_SQL)

        # Structured event extraction tables
        conn.executescript(MEMORY_EVENTS_SQL)

        # Workspace collaboration tables
        conn.executescript(WORKSPACE_SQL)

        # Additive migrations for temporal KG columns (v7)
        for col, spec in [
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("superseded_by", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {spec}")
            except sqlite3.OperationalError:
                pass

        try:
            conn.execute(
                "ALTER TABLE entities ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
        except sqlite3.OperationalError:
            pass

        # Schema version tracking
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', '8')"
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

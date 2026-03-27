from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from neuropack.exceptions import FTSQueryError, MemoryNotFoundError
from neuropack.validation import sanitize_fts_query
from neuropack.storage.database import Database
from neuropack.types import MemoryRecord, MemoryVersion, StoreStats


class MemoryRepository:
    def __init__(self, db: Database, encryptor=None):
        self._db = db
        self._encryptor = encryptor  # Optional FieldEncryptor

    def _encrypt_field(self, value: str) -> str:
        """Encrypt a text field if encryptor is configured."""
        if self._encryptor is None:
            return value
        return self._encryptor.encrypt_text(value)

    def _decrypt_field(self, value: str) -> str:
        """Decrypt a text field if encryptor is configured."""
        if self._encryptor is None:
            return value
        try:
            return self._encryptor.decrypt_text(value)
        except Exception:
            # If decryption fails, return as-is (unencrypted data or wrong key)
            return value

    def _row_to_record(self, row: dict) -> MemoryRecord:
        last_accessed = None
        if row["last_accessed"]:
            last_accessed = datetime.fromisoformat(row["last_accessed"])

        content = self._decrypt_field(row["content"])
        l3_abstract = self._decrypt_field(row["l3_abstract"])
        l2_raw = self._decrypt_field(row["l2_facts"])
        l1_raw = bytes(row["l1_compressed"])
        if self._encryptor is not None:
            try:
                l1_raw = self._encryptor.decrypt_bytes(l1_raw)
            except Exception:
                pass

        return MemoryRecord(
            id=row["id"],
            content=content,
            l3_abstract=l3_abstract,
            l2_facts=json.loads(l2_raw),
            l1_compressed=l1_raw,
            embedding=np.frombuffer(bytes(row["embedding"]), dtype=np.float32).tolist(),
            tags=json.loads(row["tags"]),
            source=row["source"],
            priority=row["priority"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            namespace=row["namespace"] if "namespace" in row.keys() else "default",
            access_count=row["access_count"],
            last_accessed=last_accessed,
            content_tokens=row.get("content_tokens", 0) if isinstance(row, dict) else (row["content_tokens"] if "content_tokens" in row.keys() else 0),
            compressed_tokens=row.get("compressed_tokens", 0) if isinstance(row, dict) else (row["compressed_tokens"] if "compressed_tokens" in row.keys() else 0),
            memory_type=row["memory_type"] if "memory_type" in row.keys() else "general",
            staleness=row["staleness"] if "staleness" in row.keys() else "stable",
            superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
        )

    def insert(self, record: MemoryRecord) -> None:
        embedding_blob = np.array(record.embedding, dtype=np.float32).tobytes()

        # Encrypt fields if encryptor is configured
        content = self._encrypt_field(record.content)
        l3 = self._encrypt_field(record.l3_abstract)
        l2 = self._encrypt_field(json.dumps(record.l2_facts))
        l1 = record.l1_compressed
        if self._encryptor is not None:
            l1 = self._encryptor.encrypt_bytes(l1)

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, content, l3_abstract, l2_facts, l1_compressed, embedding,
                    tags, source, priority, created_at, updated_at, access_count, last_accessed,
                    content_tokens, compressed_tokens, namespace,
                    memory_type, staleness, superseded_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    content,
                    l3,
                    l2,
                    l1,
                    embedding_blob,
                    json.dumps(record.tags),
                    record.source,
                    record.priority,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.access_count,
                    record.last_accessed.isoformat() if record.last_accessed else None,
                    record.content_tokens,
                    record.compressed_tokens,
                    record.namespace,
                    record.memory_type,
                    record.staleness,
                    record.superseded_by,
                ),
            )

        # Fix FTS: triggers inserted ciphertext; rebuild FTS with plaintext
        if self._encryptor is not None:
            self._rebuild_fts()

    def _ensure_fts_triggers_disabled(self) -> None:
        """Drop content-sync triggers when encryption is active."""
        conn = self._db.connect()
        for trigger in ("memories_ai", "memories_ad", "memories_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.commit()

    def _rebuild_fts(self) -> None:
        """Rebuild FTS index with plaintext content (used when encryption is active)."""
        self._ensure_fts_triggers_disabled()
        conn = self._db.connect()
        # Use 'delete-all' command to clear FTS content table
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('delete-all')")
        # Re-insert all rows with decrypted content
        rows = conn.execute(
            "SELECT rowid, content, l3_abstract, l2_facts, tags FROM memories"
        ).fetchall()
        for row in rows:
            d = dict(row)
            pt_content = self._decrypt_field(d["content"])
            pt_l3 = self._decrypt_field(d["l3_abstract"])
            pt_l2 = self._decrypt_field(d["l2_facts"])
            conn.execute(
                """INSERT INTO memories_fts(rowid, content, l3_abstract, l2_facts, tags)
                   VALUES (?, ?, ?, ?, ?)""",
                (d["rowid"], pt_content, pt_l3, pt_l2, d["tags"]),
            )
        conn.commit()

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        conn = self._db.connect()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(dict(row))

    def update(self, memory_id: str, **fields: object) -> MemoryRecord:
        existing = self.get_by_id(memory_id)
        if existing is None:
            raise MemoryNotFoundError(f"Memory {memory_id} not found")

        allowed = {
            "content", "l3_abstract", "l2_facts", "l1_compressed",
            "embedding", "tags", "source", "priority",
            "content_tokens", "compressed_tokens", "namespace",
            "memory_type", "staleness", "superseded_by",
        }
        set_clauses = []
        params: list[object] = []

        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "l2_facts":
                set_clauses.append("l2_facts = ?")
                params.append(self._encrypt_field(json.dumps(value)))
            elif key == "tags":
                set_clauses.append("tags = ?")
                params.append(json.dumps(value))
            elif key == "embedding":
                set_clauses.append("embedding = ?")
                params.append(np.array(value, dtype=np.float32).tobytes())
            elif key == "l1_compressed":
                set_clauses.append("l1_compressed = ?")
                val = value
                if self._encryptor is not None:
                    val = self._encryptor.encrypt_bytes(value)
                params.append(val)
            elif key == "content":
                set_clauses.append("content = ?")
                params.append(self._encrypt_field(value))
            elif key == "l3_abstract":
                set_clauses.append("l3_abstract = ?")
                params.append(self._encrypt_field(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)

        if not set_clauses:
            return existing

        now = datetime.now(timezone.utc).isoformat()
        set_clauses.append("updated_at = ?")
        params.append(now)
        params.append(memory_id)

        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

        # Fix FTS: trigger inserted ciphertext; rebuild FTS with plaintext
        if self._encryptor is not None:
            has_encrypted_fields = any(k in fields for k in ("content", "l3_abstract", "l2_facts"))
            if has_encrypted_fields:
                self._rebuild_fts()

        updated = self.get_by_id(memory_id)
        assert updated is not None
        return updated

    def delete(self, memory_id: str) -> bool:
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def list_all(
        self,
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
        tag: str | None = None,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        conn = self._db.connect()
        query = "SELECT * FROM memories"
        params: list[object] = []
        conditions: list[str] = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(dict(row)) for row in rows]

    def count(self) -> int:
        conn = self._db.connect()
        row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return dict(row)["cnt"]

    def stats(self, namespace: str | None = None) -> StoreStats:
        conn = self._db.connect()
        ns_filter = ""
        params: list[object] = []
        if namespace:
            ns_filter = " WHERE namespace = ?"
            params = [namespace]

        row = conn.execute(f"""
            SELECT
                COUNT(*) as cnt,
                COALESCE(SUM(LENGTH(content) + LENGTH(l1_compressed)), 0) as total_size,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM memories{ns_filter}
        """, params).fetchone()
        d = dict(row)

        avg_ratio = 0.0
        if d["cnt"] > 0:
            ratio_row = conn.execute(f"""
                SELECT AVG(CAST(LENGTH(content) AS REAL) / MAX(LENGTH(l1_compressed), 1))
                as avg_ratio FROM memories{ns_filter}
            """, params).fetchone()
            avg_ratio = dict(ratio_row)["avg_ratio"] or 0.0

        # Token stats
        token_row = conn.execute(f"""
            SELECT
                COALESCE(SUM(content_tokens), 0) as total_ct,
                COALESCE(SUM(compressed_tokens), 0) as total_cpt
            FROM memories{ns_filter}
        """, params).fetchone()
        td = dict(token_row)
        total_ct = td["total_ct"]
        total_cpt = td["total_cpt"]
        savings = round(1.0 - (total_cpt / total_ct), 4) if total_ct > 0 else 0.0

        return StoreStats(
            total_memories=d["cnt"],
            total_size_bytes=d["total_size"],
            avg_compression_ratio=avg_ratio,
            oldest=datetime.fromisoformat(d["oldest"]) if d["oldest"] else None,
            newest=datetime.fromisoformat(d["newest"]) if d["newest"] else None,
            total_content_tokens=total_ct,
            total_compressed_tokens=total_cpt,
            token_savings_ratio=savings,
        )

    def fts_search(
        self, query: str, limit: int = 20, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """FTS5 search. Returns list of (memory_id, bm25_rank)."""
        import sqlite3

        safe_query = sanitize_fts_query(query)
        conn = self._db.connect()
        try:
            if namespace:
                rows = conn.execute(
                    """SELECT m.id, fts.rank
                       FROM memories_fts fts
                       JOIN memories m ON m.rowid = fts.rowid
                       WHERE memories_fts MATCH ? AND m.namespace = ?
                       ORDER BY fts.rank
                       LIMIT ?""",
                    (safe_query, namespace, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT m.id, fts.rank
                       FROM memories_fts fts
                       JOIN memories m ON m.rowid = fts.rowid
                       WHERE memories_fts MATCH ?
                       ORDER BY fts.rank
                       LIMIT ?""",
                    (safe_query, limit),
                ).fetchall()
        except sqlite3.OperationalError as e:
            raise FTSQueryError(query, str(e))
        return [(dict(r)["id"], dict(r)["rank"]) for r in rows]

    def get_all_embeddings(
        self, namespace: str | None = None
    ) -> list[tuple[str, np.ndarray]]:
        """Load all (id, embedding) pairs for vector index rebuild."""
        conn = self._db.connect()
        if namespace:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id, embedding FROM memories").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            emb = np.frombuffer(bytes(d["embedding"]), dtype=np.float32).copy()
            result.append((d["id"], emb))
        return result

    def touch(self, memory_id: str) -> None:
        """Increment access_count and update last_accessed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, memory_id),
            )

    def record_access(self, memory_id: str) -> None:
        """Record an access event for a memory (alias for touch).

        Increments access_count and updates last_accessed timestamp.
        This is intentionally lightweight to avoid slowing down retrieval.
        """
        self.touch(memory_id)

    def get_access_stats(self, memory_id: str) -> dict:
        """Return access statistics for a memory.

        Returns:
            Dict with access_count and last_accessed_at, or empty dict if not found.
        """
        conn = self._db.connect()
        row = conn.execute(
            "SELECT access_count, last_accessed FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return {}
        d = dict(row)
        last_accessed_at = None
        if d["last_accessed"]:
            last_accessed_at = datetime.fromisoformat(d["last_accessed"])
        return {
            "access_count": d["access_count"],
            "last_accessed_at": last_accessed_at,
        }

    def save_metadata(self, key: str, value: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

    def load_metadata(self, key: str) -> str | None:
        conn = self._db.connect()
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return dict(row)["value"]

    def list_namespaces(self) -> list[dict]:
        """List all namespaces with memory counts."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT namespace, COUNT(*) as count FROM memories GROUP BY namespace ORDER BY namespace"
        ).fetchall()
        return [{"namespace": dict(r)["namespace"], "count": dict(r)["count"]} for r in rows]

    def count_by_namespace(self, namespace: str) -> int:
        """Count memories in a specific namespace."""
        conn = self._db.connect()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        return dict(row)["cnt"]

    # --- Memory Versioning ---

    def save_version(self, record: MemoryRecord, reason: str = "update") -> None:
        """Save a snapshot of a memory before it gets modified."""
        conn = self._db.connect()
        # Get next version number
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 as next_ver FROM memory_versions WHERE memory_id = ?",
            (record.id,),
        ).fetchone()
        next_ver = dict(row)["next_ver"]
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as tx:
            tx.execute(
                """INSERT INTO memory_versions (memory_id, version, content, l3_abstract, tags, saved_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (record.id, next_ver, record.content, record.l3_abstract,
                 json.dumps(record.tags), now, reason),
            )

    def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        """Get all versions of a memory, newest first."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY version DESC",
            (memory_id,),
        ).fetchall()
        return [
            MemoryVersion(
                memory_id=dict(r)["memory_id"],
                version=dict(r)["version"],
                content=dict(r)["content"],
                l3_abstract=dict(r)["l3_abstract"],
                tags=json.loads(dict(r)["tags"]),
                saved_at=datetime.fromisoformat(dict(r)["saved_at"]),
                reason=dict(r)["reason"],
            )
            for r in rows
        ]

    # --- Temporal queries ---

    def list_created_between(
        self, since: str, until: str, namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """List memories created in [since, until]."""
        conn = self._db.connect()
        conditions = ["created_at >= ?", "created_at <= ?"]
        params: list[object] = [since, until]
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at ASC",
            params,
        ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def list_updated_between(
        self, since: str, until: str, namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """List memories updated in [since, until] (includes created)."""
        conn = self._db.connect()
        conditions = ["updated_at >= ?", "updated_at <= ?"]
        params: list[object] = [since, until]
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY updated_at ASC",
            params,
        ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def get_deleted_between(self, since: str, until: str) -> list[dict]:
        """Get deletion audit entries in [since, until]."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT id, timestamp, memory_id, details FROM audit_log "
            "WHERE action = 'delete' AND timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC",
            (since, until),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_version_at(self, memory_id: str, as_of: datetime) -> MemoryVersion | None:
        """Get the version of a memory that was current at as_of.

        Returns the latest version saved before or at as_of, which represents
        the state of the memory just before it was changed.
        """
        conn = self._db.connect()
        as_of_iso = as_of.isoformat()
        row = conn.execute(
            "SELECT * FROM memory_versions "
            "WHERE memory_id = ? AND saved_at <= ? "
            "ORDER BY saved_at DESC, version DESC LIMIT 1",
            (memory_id, as_of_iso),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        return MemoryVersion(
            memory_id=d["memory_id"],
            version=d["version"],
            content=d["content"],
            l3_abstract=d["l3_abstract"],
            tags=json.loads(d["tags"]),
            saved_at=datetime.fromisoformat(d["saved_at"]),
            reason=d["reason"],
        )

    # --- Staleness queries ---

    def get_stale_memories(
        self, staleness: str, older_than_days: int, limit: int = 50
    ) -> list[MemoryRecord]:
        """Find memories of given staleness category older than N days."""
        conn = self._db.connect()
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE staleness = ? AND julianday('now') - julianday(updated_at) > ?
               ORDER BY updated_at ASC LIMIT ?""",
            (staleness, older_than_days, limit),
        ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def get_all_with_embeddings(
        self, namespace: str | None = None, exclude_superseded: bool = True,
    ) -> list[tuple[MemoryRecord, np.ndarray]]:
        """Get all memories with their embedding arrays. Used by consolidation."""
        conn = self._db.connect()
        query = "SELECT * FROM memories"
        params: list[object] = []
        conditions: list[str] = []
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        if exclude_superseded:
            conditions.append("superseded_by IS NULL")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            record = self._row_to_record(dict(row))
            emb = np.frombuffer(bytes(dict(row)["embedding"]), dtype=np.float32).copy()
            results.append((record, emb))
        return results

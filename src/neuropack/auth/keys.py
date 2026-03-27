"""API key management with scoped access control."""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone

from neuropack.storage.database import Database


class APIKeyManager:
    """Manages named API keys with scope-based access control."""

    VALID_SCOPES = {"read", "write", "admin"}
    KEY_PREFIX = "np_"

    def __init__(self, db: Database):
        self._db = db

    def create_key(self, name: str, scopes: list[str] | None = None) -> str:
        """Create a new API key. Returns the raw key (only shown once)."""
        scopes = scopes or ["read"]
        invalid = set(scopes) - self.VALID_SCOPES
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Valid: {self.VALID_SCOPES}")

        # Check for duplicate name
        conn = self._db.connect()
        existing = conn.execute(
            "SELECT id FROM api_keys WHERE name = ? AND active = 1", (name,)
        ).fetchone()
        if existing:
            raise ValueError(f"API key with name '{name}' already exists")

        # Generate key
        token = secrets.token_urlsafe(32)
        raw_key = f"{self.KEY_PREFIX}{token}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:8]
        now = datetime.now(timezone.utc).isoformat()

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO api_keys (name, key_hash, key_prefix, scopes, created_at, active)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (name, key_hash, key_prefix, json.dumps(scopes), now),
            )

        return raw_key

    def validate_key(self, raw_key: str) -> dict | None:
        """Validate a raw API key. Returns {name, scopes} or None."""
        if not raw_key.startswith(self.KEY_PREFIX):
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        conn = self._db.connect()
        row = conn.execute(
            "SELECT name, scopes, last_used FROM api_keys WHERE key_hash = ? AND active = 1",
            (key_hash,),
        ).fetchone()

        if row is None:
            return None

        d = dict(row)

        # Update last_used
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE key_hash = ?",
                (now, key_hash),
            )

        return {"name": d["name"], "scopes": json.loads(d["scopes"])}

    def list_keys(self) -> list[dict]:
        """List all active API keys (without hashes)."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT name, key_prefix, scopes, created_at, last_used FROM api_keys WHERE active = 1"
        ).fetchall()
        return [
            {
                "name": dict(r)["name"],
                "key_prefix": dict(r)["key_prefix"],
                "scopes": json.loads(dict(r)["scopes"]),
                "created_at": dict(r)["created_at"],
                "last_used": dict(r)["last_used"],
            }
            for r in rows
        ]

    def revoke_key(self, name: str) -> bool:
        """Revoke an API key by name. Returns True if found and revoked."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE name = ? AND active = 1",
                (name,),
            )
            return cursor.rowcount > 0

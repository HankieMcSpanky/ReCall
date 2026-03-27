"""Data retention policies: TTL-based auto-purge for memories.

Supports per-memory TTL (via tags or memory_type), per-namespace policies,
and global defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from neuropack.types import MemoryRecord


@dataclass
class RetentionPolicy:
    """A retention policy that determines when memories expire."""
    # Default TTL in days (0 = never expire)
    default_ttl_days: int = 0
    # Per-type TTL overrides
    type_ttl: dict[str, int] | None = None
    # Per-tag TTL overrides (tag -> days)
    tag_ttl: dict[str, int] | None = None
    # Per-namespace TTL overrides
    namespace_ttl: dict[str, int] | None = None

    def effective_ttl(self, record: MemoryRecord) -> int:
        """Compute effective TTL in days for a record. 0 means never expire."""
        # Tag overrides take highest priority (shortest TTL wins)
        if self.tag_ttl:
            tag_ttls = [self.tag_ttl[t] for t in record.tags if t in self.tag_ttl]
            if tag_ttls:
                return min(tag_ttls)

        # Namespace override
        if self.namespace_ttl and record.namespace in self.namespace_ttl:
            return self.namespace_ttl[record.namespace]

        # Type override
        if self.type_ttl and record.memory_type in self.type_ttl:
            return self.type_ttl[record.memory_type]

        return self.default_ttl_days


def find_expired_memories(
    records: list[MemoryRecord],
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> list[tuple[MemoryRecord, int]]:
    """Find memories that have expired according to the retention policy.

    Returns list of (record, ttl_days) tuples for expired memories.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    expired = []
    for record in records:
        ttl = policy.effective_ttl(record)
        if ttl <= 0:
            continue  # Never expires

        expiry = record.created_at + timedelta(days=ttl)
        if now >= expiry:
            expired.append((record, ttl))

    return expired


def parse_retention_config(config_str: str) -> RetentionPolicy:
    """Parse a retention policy from a config string.

    Format: "default:90,type:volatile:30,tag:temp:7,ns:scratch:14"
    """
    policy = RetentionPolicy()
    if not config_str.strip():
        return policy

    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")

        if tokens[0] == "default" and len(tokens) == 2:
            policy.default_ttl_days = int(tokens[1])
        elif tokens[0] == "type" and len(tokens) == 3:
            if policy.type_ttl is None:
                policy.type_ttl = {}
            policy.type_ttl[tokens[1]] = int(tokens[2])
        elif tokens[0] == "tag" and len(tokens) == 3:
            if policy.tag_ttl is None:
                policy.tag_ttl = {}
            policy.tag_ttl[tokens[1]] = int(tokens[2])
        elif tokens[0] == "ns" and len(tokens) == 3:
            if policy.namespace_ttl is None:
                policy.namespace_ttl = {}
            policy.namespace_ttl[tokens[1]] = int(tokens[2])

    return policy

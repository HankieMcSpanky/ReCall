from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AnticipationCache:
    """Cache for pre-loaded anticipatory context results."""

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[list, datetime, int]] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def put(self, query: str, results: list, timestamp: datetime) -> None:
        """Store results for a query."""
        with self._lock:
            self._entries[query] = (results, timestamp, 0)

    def get_context(self, token_budget: int = 4000) -> list[dict]:
        """Return most relevant pre-loaded results, deduped, fit within token budget."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_expired(now)

            # Collect all cached results, sorted by recency and access count
            all_results: list[tuple[dict, datetime, int]] = []
            seen_ids: set[str] = set()
            for query, (results, ts, access_count) in self._entries.items():
                # Increment access count
                self._entries[query] = (results, ts, access_count + 1)
                self._hits += 1
                for r in results:
                    rid = r.get("id", "")
                    if rid and rid in seen_ids:
                        continue
                    if rid:
                        seen_ids.add(rid)
                    all_results.append((r, ts, access_count))

            if not all_results:
                self._misses += 1
                return []

            # Sort by timestamp (newest first), then access count (most accessed first)
            all_results.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Fit within token budget using rough estimate
            budget_remaining = token_budget
            selected: list[dict] = []
            for result, _ts, _ac in all_results:
                # Rough token estimate: ~4 chars per token
                content = result.get("l3_abstract", "") or result.get("content_preview", "")
                tokens = max(len(content) // 4, 1)
                if budget_remaining - tokens < 0 and selected:
                    break
                budget_remaining -= tokens
                selected.append(result)

            return selected

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_expired(now)
            total_results = sum(len(r) for r, _, _ in self._entries.values())
            total_requests = self._hits + self._misses
            return {
                "cached_queries": len(self._entries),
                "total_results": total_results,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total_requests, 2) if total_requests > 0 else 0.0,
                "ttl_seconds": self._ttl,
            }

    def _evict_expired(self, now: datetime) -> None:
        """Remove expired entries. Must be called under lock."""
        expired = [
            q for q, (_, ts, _) in self._entries.items()
            if (now - ts).total_seconds() > self._ttl
        ]
        for q in expired:
            del self._entries[q]

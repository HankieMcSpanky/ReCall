"""Query result caching for recall operations."""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any


class RecallCache:
    """LRU + TTL cache for recall query results.

    Keys are derived from (query, namespace, tags, limit, min_score).
    Invalidated on store/delete/update operations.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 60.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._generation = 0  # Bumped on any mutation

    def _make_key(self, query: str, **kwargs: Any) -> str:
        """Create a cache key from query parameters."""
        parts = [f"g={self._generation}", f"q={query}"]
        for k in sorted(kwargs):
            v = kwargs[k]
            if v is not None:
                parts.append(f"{k}={v}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, **kwargs: Any) -> Any | None:
        """Get cached result or None if miss/expired."""
        key = self._make_key(query, **kwargs)
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def put(self, query: str, value: Any, **kwargs: Any) -> None:
        """Cache a result."""
        key = self._make_key(query, **kwargs)
        self._cache[key] = (time.monotonic(), value)
        self._cache.move_to_end(key)
        # Evict oldest if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """Invalidate all cached results (called on store/delete/update)."""
        self._generation += 1
        self._cache.clear()

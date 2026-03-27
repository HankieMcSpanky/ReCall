"""Sliding window rate limiter middleware for the API server."""
from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter. Limits requests per minute per client.

    Client is identified by API key (Authorization header) or IP address.
    """

    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Prefer API key if present
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return f"key:{auth[7:20]}"  # First 13 chars of key
        # Fall back to IP
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def _is_allowed(self, client: str) -> bool:
        """Check if request is within rate limit."""
        now = time.monotonic()
        cutoff = now - self._window

        # Remove expired entries
        timestamps = self._requests[client]
        self._requests[client] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client]) >= self._rpm:
            return False

        self._requests[client].append(now)
        return True

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static mobile assets
        path = request.url.path
        if path in ("/health", "/", "/mobile", "/mobile/manifest.json", "/mobile/sw.js", "/mobile/icon.svg"):
            return await call_next(request)

        client = self._client_key(request)
        if not self._is_allowed(client):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self._rpm} requests per minute",
                },
            )

        return await call_next(request)

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def create_auth_dependency(auth_token: str):
    """Factory: returns a dependency that validates API keys or legacy bearer tokens.
    If auth_token is empty and no API key manager is configured, auth is disabled."""

    async def verify_token(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> dict[str, Any] | None:
        """Returns auth context dict {name, scopes} or None if auth disabled."""
        # Get API key manager from app state (may not exist)
        key_manager = getattr(request.app.state, "api_key_manager", None)

        # Check if any API keys exist
        has_api_keys = False
        if key_manager is not None:
            try:
                has_api_keys = len(key_manager.list_keys()) > 0
            except Exception:
                pass

        if not auth_token and not has_api_keys:
            return None  # Auth disabled

        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials

        # Try API key first
        if key_manager is not None and token.startswith("np_"):
            result = key_manager.validate_key(token)
            if result is not None:
                return result

        # Fall back to legacy bearer token
        if auth_token and hmac.compare_digest(token, auth_token):
            return {"name": "_legacy_token", "scopes": ["read", "write", "admin"]}

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_token


def require_scope(scope: str):
    """Dependency factory: ensures the authenticated user has the required scope."""

    async def _check(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> dict[str, Any] | None:
        auth_dep = request.app.state.auth_dependency
        auth_ctx = await auth_dep(request, credentials)
        if auth_ctx is None:
            return None  # Auth disabled
        if scope not in auth_ctx.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' required",
            )
        return auth_ctx

    return _check

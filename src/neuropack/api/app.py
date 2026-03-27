from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuropack.api.auth import create_auth_dependency
from neuropack.api.middleware import RequestSizeLimitMiddleware
from neuropack.api.routes import public_router, router
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


def create_app(config: NeuropackConfig | None = None) -> FastAPI:
    config = config or NeuropackConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = MemoryStore(config)
        store.initialize()
        app.state.store = store
        app.state.config = config

        # API key manager
        from neuropack.auth.keys import APIKeyManager

        app.state.api_key_manager = APIKeyManager(store._db)

        yield
        store.close()

    app = FastAPI(
        title="NeuroPack",
        description="Local AI memory store with middle-out compression",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for Chrome extension (safe: server binds to 127.0.0.1 only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_size=config.max_content_size)

    # Rate limiting
    if config.rate_limit_rpm > 0:
        from neuropack.api.rate_limit import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware, requests_per_minute=config.rate_limit_rpm)

    # Auth dependency on protected routes (passed via include_router to avoid shared state)
    auth_dep = create_auth_dependency(config.auth_token)
    app.state.auth_dependency = auth_dep

    app.include_router(public_router)
    app.include_router(router, dependencies=[Depends(auth_dep)])

    return app

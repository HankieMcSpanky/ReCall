"""FastAPI middleware and context manager for capturing LLM calls."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Generator

from neuropack.core.store import MemoryStore
from neuropack.proxy.interceptor import LLMInterceptor

logger = logging.getLogger(__name__)

# Well-known LLM API host patterns
_LLM_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
)


class NeuroPatchMiddleware:
    """ASGI middleware that tracks LLM proxy calls passing through a FastAPI app.

    This middleware inspects outgoing-style request paths to known LLM endpoints
    (``/v1/chat/completions``, ``/v1/messages``) and captures them into NeuroPack.

    Usage::

        from fastapi import FastAPI
        from neuropack.proxy.middleware import NeuroPatchMiddleware

        app = FastAPI()
        app.add_middleware(NeuroPatchMiddleware, store=store)
    """

    def __init__(self, app: Any, store: MemoryStore, tags: list[str] | None = None) -> None:
        self.app = app
        self._interceptor = LLMInterceptor(store=store, tags=tags)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only intercept known LLM proxy paths
        if path in ("/v1/chat/completions", "/v1/messages"):
            body_parts: list[bytes] = []
            response_parts: list[bytes] = []
            status_code = 200
            start = time.monotonic()

            async def capture_receive() -> dict:
                msg = await receive()
                if msg.get("type") == "http.request":
                    body_parts.append(msg.get("body", b""))
                return msg

            async def capture_send(message: dict) -> None:
                if message.get("type") == "http.response.start":
                    nonlocal status_code
                    status_code = message.get("status", 200)
                elif message.get("type") == "http.response.body":
                    response_parts.append(message.get("body", b""))
                await send(message)

            await self.app(scope, capture_receive, capture_send)

            # After the response is complete, capture the call
            if status_code == 200:
                try:
                    import json

                    body_bytes = b"".join(body_parts)
                    resp_bytes = b"".join(response_parts)

                    if body_bytes and resp_bytes:
                        body = json.loads(body_bytes)
                        resp = json.loads(resp_bytes)
                        duration_ms = (time.monotonic() - start) * 1000

                        if path == "/v1/chat/completions":
                            self._capture_openai(body, resp, duration_ms)
                        elif path == "/v1/messages":
                            self._capture_anthropic(body, resp, duration_ms)
                except Exception:
                    logger.debug("NeuroPatchMiddleware: failed to capture LLM call", exc_info=True)
        else:
            await self.app(scope, receive, send)

    def _capture_openai(self, body: dict, resp: dict, duration_ms: float) -> None:
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        response_text = ""
        usage = None

        choices = resp.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")

        usage_data = resp.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        self._interceptor.capture(
            provider="openai",
            model=model,
            messages=messages,
            response=response_text,
            usage=usage,
            duration_ms=duration_ms,
        )

    def _capture_anthropic(self, body: dict, resp: dict, duration_ms: float) -> None:
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        system = body.get("system", "")

        full_messages: list[dict] = []
        if system:
            if isinstance(system, str):
                full_messages.append({"role": "system", "content": system})
            else:
                full_messages.append({"role": "system", "content": str(system)})
        full_messages.extend(messages)

        response_text = ""
        content_blocks = resp.get("content", [])
        parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        response_text = "\n".join(parts)

        usage = None
        usage_data = resp.get("usage")
        if usage_data:
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        self._interceptor.capture(
            provider="anthropic",
            model=model,
            messages=full_messages,
            response=response_text,
            usage=usage,
            duration_ms=duration_ms,
        )


@contextlib.contextmanager
def capture_llm_call(
    store: MemoryStore | None = None,
    provider: str = "unknown",
    model: str = "unknown",
    tags: list[str] | None = None,
) -> Generator[_CaptureContext, None, None]:
    """Context manager for wrapping individual LLM calls.

    Usage::

        from neuropack.proxy.middleware import capture_llm_call

        with capture_llm_call(store=store, provider="openai", model="gpt-4o") as ctx:
            ctx.messages = [{"role": "user", "content": "Hello"}]
            response = openai_client.chat.completions.create(...)
            ctx.response = response.choices[0].message.content
            ctx.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    """
    if store is None:
        store = MemoryStore()
        store.initialize()

    interceptor = LLMInterceptor(store=store, tags=tags)
    ctx = _CaptureContext(provider=provider, model=model)
    start = time.monotonic()

    try:
        yield ctx
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        if ctx.messages and ctx.response:
            interceptor.capture(
                provider=ctx.provider,
                model=ctx.model,
                messages=ctx.messages,
                response=ctx.response,
                usage=ctx.usage,
                duration_ms=duration_ms,
            )


class _CaptureContext:
    """Context object for capture_llm_call."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.messages: list[dict] = []
        self.response: str = ""
        self.usage: dict | None = None

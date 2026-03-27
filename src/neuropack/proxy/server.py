"""Proxy Server — OpenAI/Anthropic-compatible proxy that logs all calls to NeuroPack."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore
from neuropack.proxy.interceptor import LLMInterceptor

logger = logging.getLogger(__name__)

# Provider base URLs
OPENAI_BASE = "https://api.openai.com"
ANTHROPIC_BASE = "https://api.anthropic.com"


class ProxyServer:
    """A FastAPI app that acts as an OpenAI/Anthropic-compatible proxy server.

    Forwards requests to the real API, captures conversations, and returns responses.
    Point your apps at ``http://localhost:PORT`` instead of ``api.openai.com``.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: str = "auto",
        target_url: str = "",
        tags: list[str] | None = None,
        config: NeuropackConfig | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._target_url = target_url
        self._config = config or NeuropackConfig()
        self._interceptor = LLMInterceptor(
            store=store,
            log_prompts=self._config.proxy_log_prompts,
            log_responses=self._config.proxy_log_responses,
            tags=tags,
        )
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="NeuroPack LLM Proxy",
            description="Transparent proxy that captures LLM calls into NeuroPack memory",
        )

        @app.get("/health")
        async def health() -> dict:
            return {
                "status": "ok",
                "calls_captured": self._interceptor.call_count,
                "total_tokens": self._interceptor.total_tokens,
                "provider": self._provider,
            }

        @app.post("/v1/chat/completions")
        async def openai_proxy(request: Request) -> Response:
            return await self._handle_openai(request)

        @app.post("/v1/messages")
        async def anthropic_proxy(request: Request) -> Response:
            return await self._handle_anthropic(request)

        return app

    async def _handle_openai(self, request: Request) -> Response:
        """Forward an OpenAI-format request, capture the conversation, return response."""
        import httpx

        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        stream = body.get("stream", False)

        target_base = self._target_url or OPENAI_BASE
        url = f"{target_base}/v1/chat/completions"

        # Forward headers (especially Authorization)
        forward_headers = {}
        if "authorization" in request.headers:
            forward_headers["Authorization"] = request.headers["authorization"]
        forward_headers["Content-Type"] = "application/json"

        start = time.monotonic()

        if stream:
            return await self._stream_openai(url, forward_headers, body, messages, model, start)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=forward_headers)

        duration_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            response_text = ""
            usage = None
            try:
                choices = data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                usage_data = data.get("usage")
                if usage_data:
                    usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                    }
            except Exception:
                logger.debug("Failed to extract OpenAI response data", exc_info=True)

            self._interceptor.capture(
                provider="openai",
                model=model,
                messages=messages,
                response=response_text,
                usage=usage,
                duration_ms=duration_ms,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _stream_openai(
        self,
        url: str,
        headers: dict,
        body: dict,
        messages: list[dict],
        model: str,
        start: float,
    ) -> StreamingResponse:
        """Handle streaming OpenAI response — yield chunks while collecting full response."""
        import httpx

        chunks: list[str] = []

        async def generate():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    chunks.append(content)
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
                            yield f"{line}\n\n"
                        elif line:
                            yield f"{line}\n\n"

            # After stream completes, capture the full conversation
            duration_ms = (time.monotonic() - start) * 1000
            response_text = "".join(chunks)
            if response_text:
                self._interceptor.capture(
                    provider="openai",
                    model=model,
                    messages=messages,
                    response=response_text,
                    usage=None,
                    duration_ms=duration_ms,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def _handle_anthropic(self, request: Request) -> Response:
        """Forward an Anthropic-format request, capture the conversation, return response."""
        import httpx

        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        system = body.get("system", "")
        stream = body.get("stream", False)

        # Build full message list including system
        full_messages: list[dict] = []
        if system:
            if isinstance(system, str):
                full_messages.append({"role": "system", "content": system})
            else:
                full_messages.append({"role": "system", "content": str(system)})
        full_messages.extend(messages)

        target_base = self._target_url or ANTHROPIC_BASE
        url = f"{target_base}/v1/messages"

        # Forward headers
        forward_headers: dict[str, str] = {"Content-Type": "application/json"}
        if "x-api-key" in request.headers:
            forward_headers["x-api-key"] = request.headers["x-api-key"]
        if "anthropic-version" in request.headers:
            forward_headers["anthropic-version"] = request.headers["anthropic-version"]
        elif "anthropic-version" not in forward_headers:
            forward_headers["anthropic-version"] = "2023-06-01"

        start = time.monotonic()

        if stream:
            return await self._stream_anthropic(
                url, forward_headers, body, full_messages, model, start
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=forward_headers)

        duration_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            response_text = ""
            usage = None
            try:
                content_blocks = data.get("content", [])
                parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                response_text = "\n".join(parts)

                usage_data = data.get("usage")
                if usage_data:
                    input_tokens = usage_data.get("input_tokens", 0)
                    output_tokens = usage_data.get("output_tokens", 0)
                    usage = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
            except Exception:
                logger.debug("Failed to extract Anthropic response data", exc_info=True)

            self._interceptor.capture(
                provider="anthropic",
                model=model,
                messages=full_messages,
                response=response_text,
                usage=usage,
                duration_ms=duration_ms,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _stream_anthropic(
        self,
        url: str,
        headers: dict,
        body: dict,
        messages: list[dict],
        model: str,
        start: float,
    ) -> StreamingResponse:
        """Handle streaming Anthropic response — yield events while collecting full response."""
        import httpx

        chunks: list[str] = []

        async def generate():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type", "")
                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    text = delta.get("text", "")
                                    if text:
                                        chunks.append(text)
                            except (json.JSONDecodeError, KeyError):
                                pass
                            yield f"{line}\n\n"
                        elif line.startswith("event: "):
                            yield f"{line}\n\n"
                        elif line:
                            yield f"{line}\n\n"

            # After stream completes, capture
            duration_ms = (time.monotonic() - start) * 1000
            response_text = "".join(chunks)
            if response_text:
                self._interceptor.capture(
                    provider="anthropic",
                    model=model,
                    messages=messages,
                    response=response_text,
                    usage=None,
                    duration_ms=duration_ms,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )


def create_proxy_app(
    config: NeuropackConfig | None = None,
    provider: str = "auto",
    target_url: str = "",
    tags: list[str] | None = None,
) -> FastAPI:
    """Create a proxy FastAPI app with its own MemoryStore."""
    cfg = config or NeuropackConfig()
    store = MemoryStore(cfg)
    store.initialize()
    server = ProxyServer(
        store=store,
        provider=provider,
        target_url=target_url,
        tags=tags,
        config=cfg,
    )
    return server.app

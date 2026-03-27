"""LLM Interceptor — captures LLM API calls and stores them into NeuroPack memory."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from neuropack.core.store import MemoryStore

logger = logging.getLogger(__name__)


def _format_messages(messages: list[dict]) -> str:
    """Format a list of chat messages into a readable conversation string."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part content (e.g. vision messages)
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        parts.append("[image]")
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


class LLMInterceptor:
    """Intercepts LLM API calls and stores conversations into NeuroPack memory."""

    def __init__(
        self,
        store: MemoryStore,
        log_prompts: bool = True,
        log_responses: bool = True,
        tags: list[str] | None = None,
    ) -> None:
        self._store = store
        self._log_prompts = log_prompts
        self._log_responses = log_responses
        self._custom_tags = list(tags) if tags else []
        self._call_count = 0
        self._total_tokens = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def capture(
        self,
        provider: str,
        model: str,
        messages: list[dict],
        response: str,
        usage: dict | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Store a captured LLM conversation into NeuroPack memory."""
        try:
            parts: list[str] = []

            if self._log_prompts:
                parts.append(_format_messages(messages))

            if self._log_responses:
                parts.append(f"[assistant]: {response}")

            if not parts:
                return

            content = "\n".join(parts)

            # Build tags
            tags = [
                "llm-call",
                f"provider-{provider}",
                f"model-{model}",
            ]
            tags.extend(self._custom_tags)

            # Build source
            source = f"llm-proxy:{provider}"

            # Build metadata as a compact JSON line appended to content
            metadata: dict[str, Any] = {}
            if usage:
                metadata["usage"] = usage
                total = usage.get("total_tokens", 0)
                if total:
                    self._total_tokens += total
            if duration_ms is not None:
                metadata["duration_ms"] = round(duration_ms, 1)
            metadata["model"] = model
            metadata["provider"] = provider

            if metadata:
                content += f"\n\n---\nMetadata: {json.dumps(metadata)}"

            self._store.store(
                content=content,
                tags=tags,
                source=source,
                priority=0.3,
            )

            self._call_count += 1

        except Exception:
            # Never break the user's app if NeuroPack storage fails
            logger.debug("Failed to capture LLM call", exc_info=True)

    def wrap_openai(self, client: Any) -> Any:
        """Monkey-patch an OpenAI client to intercept chat.completions.create calls.

        Works with both sync and async clients. Returns the patched client.
        """
        interceptor = self

        # Detect if this is an async client
        is_async = _is_async_openai(client)

        if is_async:
            original_create = client.chat.completions.create

            async def async_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                model = kwargs.get("model", args[0] if args else "unknown")
                stream = kwargs.get("stream", False)

                start = time.monotonic()

                if stream:
                    return _AsyncOpenAIStreamWrapper(
                        await original_create(*args, **kwargs),
                        interceptor=interceptor,
                        model=model,
                        messages=list(messages),
                        start_time=start,
                    )

                result = await original_create(*args, **kwargs)
                duration_ms = (time.monotonic() - start) * 1000

                response_text = _extract_openai_response(result)
                usage = _extract_openai_usage(result)

                interceptor.capture(
                    provider="openai",
                    model=model,
                    messages=list(messages),
                    response=response_text,
                    usage=usage,
                    duration_ms=duration_ms,
                )
                return result

            client.chat.completions.create = async_create_wrapper
        else:
            original_create = client.chat.completions.create

            def sync_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                model = kwargs.get("model", args[0] if args else "unknown")
                stream = kwargs.get("stream", False)

                start = time.monotonic()

                if stream:
                    return _SyncOpenAIStreamWrapper(
                        original_create(*args, **kwargs),
                        interceptor=interceptor,
                        model=model,
                        messages=list(messages),
                        start_time=start,
                    )

                result = original_create(*args, **kwargs)
                duration_ms = (time.monotonic() - start) * 1000

                response_text = _extract_openai_response(result)
                usage = _extract_openai_usage(result)

                interceptor.capture(
                    provider="openai",
                    model=model,
                    messages=list(messages),
                    response=response_text,
                    usage=usage,
                    duration_ms=duration_ms,
                )
                return result

            client.chat.completions.create = sync_create_wrapper

        return client

    def wrap_anthropic(self, client: Any) -> Any:
        """Monkey-patch an Anthropic client to intercept messages.create calls.

        Works with both sync and async clients. Returns the patched client.
        """
        interceptor = self

        is_async = _is_async_anthropic(client)

        if is_async:
            original_create = client.messages.create

            async def async_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "unknown")
                system = kwargs.get("system", "")
                stream = kwargs.get("stream", False)

                # Build message list with system prompt
                full_messages = []
                if system:
                    if isinstance(system, str):
                        full_messages.append({"role": "system", "content": system})
                    else:
                        full_messages.append({"role": "system", "content": str(system)})
                full_messages.extend(messages)

                start = time.monotonic()

                if stream:
                    return _AsyncAnthropicStreamWrapper(
                        await original_create(*args, **kwargs),
                        interceptor=interceptor,
                        model=model,
                        messages=full_messages,
                        start_time=start,
                    )

                result = await original_create(*args, **kwargs)
                duration_ms = (time.monotonic() - start) * 1000

                response_text = _extract_anthropic_response(result)
                usage = _extract_anthropic_usage(result)

                interceptor.capture(
                    provider="anthropic",
                    model=model,
                    messages=full_messages,
                    response=response_text,
                    usage=usage,
                    duration_ms=duration_ms,
                )
                return result

            client.messages.create = async_create_wrapper
        else:
            original_create = client.messages.create

            def sync_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "unknown")
                system = kwargs.get("system", "")
                stream = kwargs.get("stream", False)

                full_messages = []
                if system:
                    if isinstance(system, str):
                        full_messages.append({"role": "system", "content": system})
                    else:
                        full_messages.append({"role": "system", "content": str(system)})
                full_messages.extend(messages)

                start = time.monotonic()

                if stream:
                    return _SyncAnthropicStreamWrapper(
                        original_create(*args, **kwargs),
                        interceptor=interceptor,
                        model=model,
                        messages=full_messages,
                        start_time=start,
                    )

                result = original_create(*args, **kwargs)
                duration_ms = (time.monotonic() - start) * 1000

                response_text = _extract_anthropic_response(result)
                usage = _extract_anthropic_usage(result)

                interceptor.capture(
                    provider="anthropic",
                    model=model,
                    messages=full_messages,
                    response=response_text,
                    usage=usage,
                    duration_ms=duration_ms,
                )
                return result

            client.messages.create = sync_create_wrapper

        return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_async_openai(client: Any) -> bool:
    """Check if an OpenAI client is the async variant."""
    cls_name = type(client).__name__
    return "Async" in cls_name


def _is_async_anthropic(client: Any) -> bool:
    """Check if an Anthropic client is the async variant."""
    cls_name = type(client).__name__
    return "Async" in cls_name


def _extract_openai_response(result: Any) -> str:
    """Extract text content from an OpenAI ChatCompletion response."""
    try:
        choices = result.choices
        if choices:
            message = choices[0].message
            return message.content or ""
    except Exception:
        pass
    return ""


def _extract_openai_usage(result: Any) -> dict | None:
    """Extract token usage from an OpenAI ChatCompletion response."""
    try:
        usage = result.usage
        if usage:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
    except Exception:
        pass
    return None


def _extract_anthropic_response(result: Any) -> str:
    """Extract text content from an Anthropic Messages response."""
    try:
        content_blocks = result.content
        parts = []
        for block in content_blocks:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
    except Exception:
        pass
    return ""


def _extract_anthropic_usage(result: Any) -> dict | None:
    """Extract token usage from an Anthropic Messages response."""
    try:
        usage = result.usage
        if usage:
            return {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0),
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Stream wrappers
# ---------------------------------------------------------------------------


class _SyncOpenAIStreamWrapper:
    """Wraps an OpenAI sync streaming response to collect and capture content."""

    def __init__(
        self,
        stream: Any,
        interceptor: LLMInterceptor,
        model: str,
        messages: list[dict],
        start_time: float,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._chunks: list[str] = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._collect_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def _collect_chunk(self, chunk: Any) -> None:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                self._chunks.append(delta.content)
        except Exception:
            pass

    def _finalize(self) -> None:
        if not self._chunks:
            return
        duration_ms = (time.monotonic() - self._start_time) * 1000
        response = "".join(self._chunks)
        self._interceptor.capture(
            provider="openai",
            model=self._model,
            messages=self._messages,
            response=response,
            usage=None,
            duration_ms=duration_ms,
        )
        self._chunks.clear()

    # Proxy common stream attributes
    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncOpenAIStreamWrapper:
    """Wraps an OpenAI async streaming response to collect and capture content."""

    def __init__(
        self,
        stream: Any,
        interceptor: LLMInterceptor,
        model: str,
        messages: list[dict],
        start_time: float,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._chunks: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._collect_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)
        return False

    def _collect_chunk(self, chunk: Any) -> None:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                self._chunks.append(delta.content)
        except Exception:
            pass

    def _finalize(self) -> None:
        if not self._chunks:
            return
        duration_ms = (time.monotonic() - self._start_time) * 1000
        response = "".join(self._chunks)
        self._interceptor.capture(
            provider="openai",
            model=self._model,
            messages=self._messages,
            response=response,
            usage=None,
            duration_ms=duration_ms,
        )
        self._chunks.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _SyncAnthropicStreamWrapper:
    """Wraps an Anthropic sync streaming response to collect and capture content."""

    def __init__(
        self,
        stream: Any,
        interceptor: LLMInterceptor,
        model: str,
        messages: list[dict],
        start_time: float,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._chunks: list[str] = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._collect_event(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def _collect_event(self, event: Any) -> None:
        try:
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        self._chunks.append(delta.text)
        except Exception:
            pass

    def _finalize(self) -> None:
        if not self._chunks:
            return
        duration_ms = (time.monotonic() - self._start_time) * 1000
        response = "".join(self._chunks)
        self._interceptor.capture(
            provider="anthropic",
            model=self._model,
            messages=self._messages,
            response=response,
            usage=None,
            duration_ms=duration_ms,
        )
        self._chunks.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncAnthropicStreamWrapper:
    """Wraps an Anthropic async streaming response to collect and capture content."""

    def __init__(
        self,
        stream: Any,
        interceptor: LLMInterceptor,
        model: str,
        messages: list[dict],
        start_time: float,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._chunks: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
            self._collect_event(event)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)
        return False

    def _collect_event(self, event: Any) -> None:
        try:
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        self._chunks.append(delta.text)
        except Exception:
            pass

    def _finalize(self) -> None:
        if not self._chunks:
            return
        duration_ms = (time.monotonic() - self._start_time) * 1000
        response = "".join(self._chunks)
        self._interceptor.capture(
            provider="anthropic",
            model=self._model,
            messages=self._messages,
            response=response,
            usage=None,
            duration_ms=duration_ms,
        )
        self._chunks.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

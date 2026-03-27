"""Convenience functions for wrapping LLM clients with NeuroPack interception."""

from __future__ import annotations

from typing import Any

from neuropack.core.store import MemoryStore
from neuropack.proxy.interceptor import LLMInterceptor


def _get_or_create_store(store: MemoryStore | None) -> MemoryStore:
    """Return the provided store or create a default one."""
    if store is not None:
        return store
    s = MemoryStore()
    s.initialize()
    return s


def proxy_openai(
    client: Any,
    store: MemoryStore | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Wrap an OpenAI client instance so all calls are automatically captured.

    Usage::

        from neuropack.proxy import proxy_openai
        import openai

        client = proxy_openai(openai.OpenAI())
        # Now all calls are automatically captured into NeuroPack
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """
    s = _get_or_create_store(store)
    interceptor = LLMInterceptor(store=s, tags=tags)
    return interceptor.wrap_openai(client)


def proxy_anthropic(
    client: Any,
    store: MemoryStore | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Wrap an Anthropic client instance so all calls are automatically captured.

    Usage::

        from neuropack.proxy import proxy_anthropic
        import anthropic

        client = proxy_anthropic(anthropic.Anthropic())
        # Now all calls are automatically captured into NeuroPack
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """
    s = _get_or_create_store(store)
    interceptor = LLMInterceptor(store=s, tags=tags)
    return interceptor.wrap_anthropic(client)


def proxy_all(
    store: MemoryStore | None = None,
    tags: list[str] | None = None,
) -> None:
    """Monkey-patch both openai and anthropic modules globally.

    After calling this, any new OpenAI() or Anthropic() client created anywhere
    in the process will have its ``chat.completions.create`` or
    ``messages.create`` method automatically intercepted.

    Usage::

        from neuropack.proxy import proxy_all
        proxy_all()

        # Now any client created in any library will be captured
        import openai
        client = openai.OpenAI()
        client.chat.completions.create(...)  # <-- automatically captured
    """
    s = _get_or_create_store(store)
    interceptor = LLMInterceptor(store=s, tags=tags)

    # Patch OpenAI
    try:
        import openai

        _original_openai_init = openai.OpenAI.__init__

        def _patched_openai_init(self_inner: Any, *args: Any, **kwargs: Any) -> None:
            _original_openai_init(self_inner, *args, **kwargs)
            interceptor.wrap_openai(self_inner)

        openai.OpenAI.__init__ = _patched_openai_init

        # Also patch async client
        if hasattr(openai, "AsyncOpenAI"):
            _original_async_openai_init = openai.AsyncOpenAI.__init__

            def _patched_async_openai_init(self_inner: Any, *args: Any, **kwargs: Any) -> None:
                _original_async_openai_init(self_inner, *args, **kwargs)
                interceptor.wrap_openai(self_inner)

            openai.AsyncOpenAI.__init__ = _patched_async_openai_init

    except ImportError:
        pass

    # Patch Anthropic
    try:
        import anthropic

        _original_anthropic_init = anthropic.Anthropic.__init__

        def _patched_anthropic_init(self_inner: Any, *args: Any, **kwargs: Any) -> None:
            _original_anthropic_init(self_inner, *args, **kwargs)
            interceptor.wrap_anthropic(self_inner)

        anthropic.Anthropic.__init__ = _patched_anthropic_init

        # Also patch async client
        if hasattr(anthropic, "AsyncAnthropic"):
            _original_async_anthropic_init = anthropic.AsyncAnthropic.__init__

            def _patched_async_anthropic_init(self_inner: Any, *args: Any, **kwargs: Any) -> None:
                _original_async_anthropic_init(self_inner, *args, **kwargs)
                interceptor.wrap_anthropic(self_inner)

            anthropic.AsyncAnthropic.__init__ = _patched_async_anthropic_init

    except ImportError:
        pass

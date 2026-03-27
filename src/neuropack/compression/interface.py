from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AbstractCompressor(Protocol):
    """Protocol for L3 compressors that produce a one-line abstract."""

    def compress_l3(self, text: str) -> str: ...


@runtime_checkable
class SemanticCompressor(Protocol):
    """Protocol for L2 compressors that extract key facts."""

    def compress_l2(self, text: str) -> list[str]: ...

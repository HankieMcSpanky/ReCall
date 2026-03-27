from __future__ import annotations

import zstandard

from neuropack.compression.extractive import ExtractiveCompressor
from neuropack.compression.interface import AbstractCompressor, SemanticCompressor
from neuropack.types import CompressedMemory


class MiddleOutCompressor:
    """Middle-out compression engine producing L3/L2/L1 from raw text."""

    def __init__(
        self,
        l3_compressor: AbstractCompressor | None = None,
        l2_compressor: SemanticCompressor | None = None,
        zstd_level: int = 3,
    ):
        self._extractive = ExtractiveCompressor()
        self._l3 = l3_compressor or self._extractive
        self._l2 = l2_compressor or self._extractive
        self._zstd = zstandard.ZstdCompressor(level=zstd_level)
        self._zstd_decompressor = zstandard.ZstdDecompressor()

    def compress(
        self,
        text: str,
        l3_override: str | None = None,
        l2_override: list[str] | None = None,
        l1_source: str | None = None,
    ) -> CompressedMemory:
        """Compress text into three levels.

        If l3_override/l2_override are provided (caller-provided), use them directly.
        Otherwise, fall back to the configured compressors.
        l1_source: if set, L1 compresses this text instead of the input text.
        Used by privacy tags to store full content in L1 while L3/L2 use stripped text.
        """
        l3 = l3_override if l3_override is not None else self._l3.compress_l3(text)
        l2 = l2_override if l2_override is not None else self._l2.compress_l2(text)
        l1_text = l1_source if l1_source is not None else text
        l1 = self._zstd.compress(l1_text.encode("utf-8"))
        return CompressedMemory(l3=l3, l2=l2, l1=l1)

    def decompress_l1(self, data: bytes) -> str:
        """Decompress L1 zstd bytes back to original text."""
        return self._zstd_decompressor.decompress(data).decode("utf-8")
